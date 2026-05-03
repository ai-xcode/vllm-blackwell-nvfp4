#!/usr/bin/env python3
"""nvfp4_ui.py — NiceGUI front-end for the NVFP4 conversion + bench workflow.

Wraps the three CLI tools in this repo:
  - convert_to_nvfp4.py  (BF16/FP16 → NVFP4)
  - start-nvfp4.sh       (vLLM serve with the curand fix)
  - bench_tps.py         (single-stream tok/s against any OpenAI-compat endpoint)

Override paths via env vars:
  NVFP4_MODELS_DIR   — where to find source + NVFP4 model dirs
  NVFP4_CONVERT_VENV — venv with nvidia-modelopt (for conversion)
  NVFP4_SERVE_VENV   — venv with vLLM (for serving + bench)
  NVFP4_UI_PORT      — UI port (default 8770)

Requires: nicegui (>= 1.4 or 3.x).
"""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path

from nicegui import ui

# ── config ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()

MODELS_DIR = Path(os.environ.get(
    "NVFP4_MODELS_DIR",
    str(Path.home() / "vLLM_Servers" / "models_awq")))
CONVERT_VENV = Path(os.environ.get(
    "NVFP4_CONVERT_VENV",
    str(Path.home() / "nvfp4_conversion" / "venv")))
SERVE_VENV = Path(os.environ.get(
    "NVFP4_SERVE_VENV",
    str(Path.home() / "vLLM_Servers" / "vllm_env")))
UI_PORT = int(os.environ.get("NVFP4_UI_PORT", "8770"))

CONVERT_PY = HERE / "convert_to_nvfp4.py"
START_SH = HERE / "start-nvfp4.sh"
BENCH_PY = HERE / "bench_tps.py"


# ── helpers ───────────────────────────────────────────────────────────────
def gpu_state() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2)
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({
                "idx": parts[0],
                "used_mb": int(parts[1]),
                "free_mb": int(parts[2]),
                "util": int(parts[3]),
            })
    return rows


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def list_dirs(root: Path, with_nvfp4: bool) -> list[str]:
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p.name.endswith("-NVFP4") == with_nvfp4):
            out.append(str(p))
    return out


# ── shared state ──────────────────────────────────────────────────────────
class AppState:
    def __init__(self) -> None:
        self.convert_proc: asyncio.subprocess.Process | None = None
        self.serve_proc: asyncio.subprocess.Process | None = None
        self.serve_port: int = 8011
        self.serve_model: str = ""

state = AppState()


# ── UI ────────────────────────────────────────────────────────────────────
with ui.header().classes("items-center justify-between"):
    ui.label("NVFP4 Conversion Center").classes("text-2xl font-bold")
    ui.label(f":{UI_PORT}").classes("text-xs opacity-60")

with ui.row().classes("w-full items-stretch gap-4 q-mt-md"):
    with ui.card().classes("flex-1"):
        ui.label("GPU state").classes("text-sm font-semibold opacity-70")
        gpu_label = ui.label("…").classes("text-sm font-mono whitespace-pre")
    with ui.card().classes("flex-1"):
        ui.label("Serving").classes("text-sm font-semibold opacity-70")
        serve_label = ui.label("…").classes("text-sm font-mono whitespace-pre")


def _refresh_status() -> None:
    gpus = gpu_state()
    if gpus:
        rows = []
        for g in gpus:
            used = g["used_mb"] / 1024
            total = (g["used_mb"] + g["free_mb"]) / 1024
            rows.append(f"GPU{g['idx']}  {used:5.1f}/{total:5.1f} GB  · util {g['util']}%")
        gpu_label.text = "\n".join(rows)
    else:
        gpu_label.text = "nvidia-smi not available"

    p = state.serve_proc
    if p and p.returncode is None:
        live = port_in_use(state.serve_port)
        live_str = "LIVE" if live else "starting…"
        serve_label.text = (
            f"{state.serve_model}\n"
            f":{state.serve_port}  {live_str}\n"
            f"pid {p.pid}"
        )
    else:
        any_8011 = port_in_use(8011)
        serve_label.text = (
            f"(none from this UI)\n"
            f":8011 {'occupied' if any_8011 else 'free'}"
        )


ui.timer(2.0, _refresh_status)


with ui.tabs().classes("w-full") as tabs:
    tab_convert = ui.tab("Convert", icon="autorenew")
    tab_serve = ui.tab("Serve", icon="play_circle")
    tab_bench = ui.tab("Bench", icon="speed")

with ui.tab_panels(tabs, value=tab_convert).classes("w-full"):

    # ── Convert ──────────────────────────────────────────────────
    with ui.tab_panel(tab_convert):
        ui.label("Convert a BF16/FP16 HF model to NVFP4").classes("text-lg")
        ui.label(f"Source candidates from: {MODELS_DIR}").classes("text-xs opacity-60")

        src_options = list_dirs(MODELS_DIR, with_nvfp4=False)

        def _autofill(_e=None) -> None:
            v = src_select.value
            if v and not out_input.value.strip():
                out_input.value = v + "-NVFP4"

        with ui.row().classes("w-full items-end gap-2"):
            src_select = ui.select(
                options=src_options or [""],
                with_input=True,
                label="Source (BF16/FP16 dir)",
                on_change=_autofill,
            ).classes("flex-1")
            ui.button(icon="refresh",
                      on_click=lambda: setattr(src_select, "options",
                                               list_dirs(MODELS_DIR, with_nvfp4=False))
                      ).props("flat dense").tooltip("rescan source dir")

        with ui.row().classes("w-full items-end gap-2"):
            out_input = ui.input(label="Output dir (must not exist)").classes("flex-1")
            calib_select = ui.select([32, 64, 128, 256, 512], value=256,
                                     label="Calib samples")
            device_select = ui.select(
                {"cuda:0": "GPU 0", "cuda:1": "GPU 1", "auto": "auto (multi-GPU)"},
                value="cuda:0", label="Device")

        convert_log = ui.log(max_lines=2000).classes(
            "h-96 w-full text-xs font-mono")

        async def _do_convert() -> None:
            if state.convert_proc and state.convert_proc.returncode is None:
                ui.notify("conversion already running", type="warning"); return
            src = (src_select.value or "").strip()
            out = (out_input.value or "").strip()
            if not src or not Path(src).is_dir():
                ui.notify("source is not a valid directory", type="negative"); return
            if not out:
                ui.notify("output dir required", type="negative"); return
            if Path(out).exists():
                ui.notify(f"output already exists: {out}", type="negative"); return

            py = CONVERT_VENV / "bin" / "python"
            if not py.is_file():
                ui.notify(f"conversion venv python not found at {py}",
                          type="negative"); return

            cmd = [str(py), str(CONVERT_PY),
                   "--source", src,
                   "--output", out,
                   "--calib-samples", str(calib_select.value),
                   "--device", device_select.value]
            convert_log.clear()
            convert_log.push(f"$ {' '.join(cmd)}\n")
            convert_btn.props("loading")
            try:
                state.convert_proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT)
                assert state.convert_proc.stdout is not None
                async for line in state.convert_proc.stdout:
                    convert_log.push(line.decode(errors="replace").rstrip())
                rc = await state.convert_proc.wait()
                if rc == 0:
                    convert_log.push("\n✓ done (exit 0)")
                    ui.notify("conversion finished", type="positive")
                else:
                    convert_log.push(f"\n✗ exit code {rc}")
                    ui.notify(f"conversion failed (exit {rc})", type="negative")
            except Exception as e:
                convert_log.push(f"error: {e}")
                ui.notify(f"error: {e}", type="negative")
            finally:
                convert_btn.props(remove="loading")
                state.convert_proc = None

        convert_btn = ui.button("Start conversion", icon="autorenew",
                                on_click=_do_convert) \
                        .props("color=primary unelevated")

    # ── Serve ────────────────────────────────────────────────────
    with ui.tab_panel(tab_serve):
        ui.label("Serve an NVFP4 model in vLLM").classes("text-lg")
        ui.label(f"NVFP4 models in {MODELS_DIR}:").classes("text-xs opacity-60")

        with ui.row().classes("w-full items-end gap-2"):
            nvfp4_options = list_dirs(MODELS_DIR, with_nvfp4=True)
            serve_select = ui.select(options=nvfp4_options or [""],
                                     label="Model dir").classes("flex-1")
            ui.button(icon="refresh",
                      on_click=lambda: setattr(serve_select, "options",
                                               list_dirs(MODELS_DIR, with_nvfp4=True))
                      ).props("flat dense").tooltip("rescan NVFP4 dirs")

        with ui.row().classes("w-full items-end gap-2"):
            port_input = ui.number(label="Port", value=8011, min=8000, max=9000)
            util_label = ui.label("util 0.45")
            util_slider = ui.slider(min=0.10, max=0.90, step=0.05, value=0.45,
                                    on_change=lambda e: setattr(
                                        util_label, "text",
                                        f"util {e.value:.2f}")
                                    ).classes("flex-1")
            max_len_input = ui.number(label="max-model-len", value=4096, min=512)

        serve_log = ui.log(max_lines=500).classes("h-64 w-full text-xs font-mono")

        async def _do_serve() -> None:
            if state.serve_proc and state.serve_proc.returncode is None:
                ui.notify("already serving — stop first", type="warning"); return
            model = (serve_select.value or "").strip()
            if not model or not Path(model).is_dir():
                ui.notify("pick a valid NVFP4 model dir", type="negative"); return
            port = int(port_input.value)
            if port_in_use(port):
                ui.notify(f"port {port} already in use", type="negative"); return

            cmd = [str(START_SH), model,
                   "--port", str(port),
                   "--util", f"{util_slider.value:.2f}",
                   "--max-len", str(int(max_len_input.value))]
            serve_log.clear()
            serve_log.push(f"$ {' '.join(cmd)}\n")
            try:
                state.serve_proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT)
                state.serve_port = port
                state.serve_model = Path(model).name
                ui.notify(f"starting on :{port} (cold flashinfer cache: ~1–2 min)",
                          type="info")
                assert state.serve_proc.stdout is not None
                async for line in state.serve_proc.stdout:
                    serve_log.push(line.decode(errors="replace").rstrip())
                rc = await state.serve_proc.wait()
                serve_log.push(f"\nexited with code {rc}")
            except Exception as e:
                serve_log.push(f"error: {e}")
            finally:
                state.serve_proc = None

        async def _do_stop() -> None:
            p = state.serve_proc
            if not p or p.returncode is not None:
                ui.notify("not running", type="warning"); return
            try:
                p.terminate()
                try:
                    await asyncio.wait_for(p.wait(), timeout=5)
                except asyncio.TimeoutError:
                    p.kill()
                # vLLM workers don't always exit with their parent
                subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"],
                               check=False, timeout=5)
                ui.notify("stopped", type="positive")
            except Exception as e:
                ui.notify(f"stop error: {e}", type="negative")

        with ui.row().classes("gap-2 q-mt-sm"):
            ui.button("Start", icon="play_arrow", on_click=_do_serve) \
                .props("color=primary unelevated")
            ui.button("Stop", icon="stop", on_click=_do_stop) \
                .props("color=negative outline")

    # ── Bench ────────────────────────────────────────────────────
    with ui.tab_panel(tab_bench):
        ui.label("Single-stream tok/s benchmark").classes("text-lg")
        ui.label("Hits any OpenAI-compatible endpoint (vLLM, ollama, llama.cpp/server).") \
            .classes("text-xs opacity-60")

        with ui.row().classes("w-full items-end gap-2"):
            url_input = ui.input(label="Base URL",
                                 value="http://127.0.0.1:8011/v1") \
                          .classes("flex-1")
            runs_select = ui.select([3, 5, 10, 20], value=10, label="Runs")
            warmup_select = ui.select([0, 1, 2], value=1, label="Warmup")
            mt_input = ui.number(label="max-tokens", value=256, min=16, max=2048)

        bench_log = ui.log(max_lines=200).classes("h-48 w-full text-xs font-mono")
        bench_summary = ui.label("").classes("font-mono text-sm whitespace-pre q-mt-sm")

        async def _do_bench() -> None:
            url = (url_input.value or "").strip()
            if not url:
                ui.notify("URL required", type="negative"); return
            py = SERVE_VENV / "bin" / "python"
            if not py.is_file():
                py = Path(sys.executable)
            cmd = [str(py), str(BENCH_PY),
                   "--url", url,
                   "--runs", str(int(runs_select.value)),
                   "--warmup", str(int(warmup_select.value)),
                   "--max-tokens", str(int(mt_input.value))]
            bench_log.clear()
            bench_summary.text = ""
            bench_log.push(f"$ {' '.join(cmd)}\n")
            bench_btn.props("loading")
            captured: list[str] = []
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT)
                assert proc.stdout is not None
                async for line in proc.stdout:
                    s = line.decode(errors="replace").rstrip()
                    bench_log.push(s)
                    captured.append(s)
                rc = await proc.wait()
                if rc != 0:
                    ui.notify(f"bench exited with code {rc}", type="negative"); return
                bench_summary.text = "\n".join(l for l in captured if l.startswith("# "))
                ui.notify("bench done", type="positive")
            except Exception as e:
                bench_log.push(f"error: {e}")
                ui.notify(f"error: {e}", type="negative")
            finally:
                bench_btn.props(remove="loading")

        bench_btn = ui.button("Run bench", icon="speed", on_click=_do_bench) \
                      .props("color=primary unelevated")


with ui.footer().classes("bg-transparent text-xs opacity-50"):
    ui.label("nvfp4_ui · part of vllm-blackwell-nvfp4 · MIT")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="127.0.0.1",
        port=UI_PORT,
        title="NVFP4 Conversion Center",
        reload=False,
        show=False,
    )
