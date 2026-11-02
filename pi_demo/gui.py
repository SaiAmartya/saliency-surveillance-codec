#!/usr/bin/env python3
"""
Demo Console — all-in-one tkinter wrapper for the demo scripts.

Two tabs:
  * Live Saliency Demo  — launches scripts/live_demo.py with the saliency
                          backend / camera / bg-samples / auto-recal flags
                          surfaced as form controls.
  * Compression Pipeline — pick any video, set CRF + saliency + bg_mode
                          (including the rolling-mode parameters), click
                          Run, watch the log stream live, and when the
                          three-stage pipeline finishes (ffmpeg stage →
                          ablation_bgfg → compare_clip --internals) click
                          "Open Comparison Video" to launch the result.

Subprocess + threading model:
  - All heavy work runs in a daemon worker thread that shells out via
    subprocess.Popen to the existing CLI scripts. The GUI is a thin
    command-builder + log streamer, not a re-implementation.
  - Subprocess stdout/stderr is read line-by-line in the worker, pushed
    into a queue.Queue, and drained on the tk main thread via after().
    This is the standard pattern — UI never blocks on encode work, and
    UI mutations only happen on the main thread.

Run from the project root:
    python gui.py
"""
from __future__ import annotations

import os
import platform
import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

ROOT = Path(__file__).resolve().parent
DATA_REAL = ROOT / "data" / "real"
ENCODED   = ROOT / "results" / "ablation_bgfg" / "encoded"
COMPARES  = ROOT / "results" / "comparisons"

SALIENCY_CHOICES = ["spectral", "finegrained", "yolo", "yolo+spectral"]
CRF_CHOICES      = ["22", "28", "34"]
PANEL_SCALES     = ["0.25", "0.5", "0.75", "1.0"]

# Comparison video name we ship in the bundle (when launcher mode triggers).
BUNDLED_COMPARISON_VIDEO = "clip_virat_crf22.mp4"
BUNDLED_DEMO_SCRIPT      = "demo_pi.py"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def open_externally(path: Path) -> None:
    """Open a file with the OS default handler."""
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", str(path)])
    elif system == "Linux":
        subprocess.Popen(["xdg-open", str(path)])
    elif system == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]


def derive_clip_id(video_path: Path) -> str:
    """Turn a video filename into a script-friendly clip_<id> string."""
    stem = video_path.stem.lower()
    safe = "".join(c if c.isalnum() else "_" for c in stem).strip("_")
    return safe if safe.startswith("clip_") else f"clip_{safe}"


def _is_raspberry_pi() -> bool:
    """True if this machine looks like a Raspberry Pi.

    Reads /proc/device-tree/model (exists only on Pi/embedded ARM and
    contains the literal string 'Raspberry Pi' on Raspberry Pi OS), then
    falls back to /proc/cpuinfo for older or oddly-configured installs.
    Both files are absent on macOS / Windows, so this returns False there.
    """
    for candidate in (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")):
        try:
            if "raspberry pi" in candidate.read_text(errors="ignore").lower():
                return True
        except OSError:
            pass
    try:
        if "raspberry" in Path("/proc/cpuinfo").read_text(errors="ignore").lower():
            return True
    except OSError:
        pass
    return False


def _has_local_bundle() -> bool:
    """True if gui.py is colocated with demo_pi.py AND the comparison video.

    This is the signal that we're running inside the shipped pi_demo bundle.
    Triggers launcher mode even on non-Pi hardware (e.g. a developer testing
    the partner-facing UI from their Mac inside the bundle directory).
    """
    return (ROOT / BUNDLED_DEMO_SCRIPT).exists() and (ROOT / BUNDLED_COMPARISON_VIDEO).exists()


def should_use_launcher_mode() -> bool:
    """Either signal flips launcher mode on; otherwise full developer GUI."""
    return _is_raspberry_pi() or _has_local_bundle()


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------

class DemoConsole:
    POLL_MS = 100

    def __init__(self, root: tk.Tk, launcher_mode: bool = False) -> None:
        self.root = root
        self.launcher_mode = launcher_mode

        if self.launcher_mode:
            self.root.title("Surveillance Codec — Pi Demo Console")
            self.root.geometry("1000x760")
            self.root.minsize(880, 640)
        else:
            self.root.title("Surveillance Codec — Demo Console")
            self.root.geometry("1000x760")
            self.root.minsize(880, 640)

        self.log_queue: "queue.Queue[object]" = queue.Queue()
        self.current_proc: subprocess.Popen | None = None
        self.worker_thread: threading.Thread | None = None
        self.last_comparison_path: Path | None = None
        self._action_buttons: list = []   # populated by panel builders; toggled by _set_running

        self._build_style()
        self._build_layout()
        self.root.after(self.POLL_MS, self._poll_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- UI scaffolding ----

    def _build_style(self) -> None:
        style = ttk.Style()
        # Native theme on macOS, fall through on Linux/Windows
        try:
            style.theme_use("aqua")
        except tk.TclError:
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Helvetica", 12, "bold"))

    def _build_layout(self) -> None:
        # Title bar (both modes)
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=14, pady=(12, 6))
        title_text = (
            "Surveillance Codec — Pi Demo Console"
            if self.launcher_mode else "Surveillance Codec — Demo Console"
        )
        ttk.Label(header, text=title_text, style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text=" saliency-aware bg/fg compression for surveillance footage",
            foreground="#555",
        ).pack(side="left", padx=(8, 0))

        # Middle: notebook tabs in both modes.
        #   * launcher (Pi / bundled) mode → Live Saliency Demo + the hardcoded
        #     Compression Pipeline replay (demo_pi.py).
        #   * full developer mode          → Live Saliency Demo + the full
        #     Compression Pipeline builder.
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=False, padx=14, pady=4)
        self.live_tab = ttk.Frame(nb)
        self.comp_tab = ttk.Frame(nb)
        nb.add(self.live_tab, text="  Live Saliency Demo  ")
        if self.launcher_mode:
            nb.add(self.comp_tab, text="  Compression Pipeline Demo  ")
            self._build_live_tab()
            self._build_launcher_panel(self.comp_tab)
        else:
            nb.add(self.comp_tab, text="  Compression Pipeline  ")
            self._build_live_tab()
            self._build_comp_tab()

        # Log + status bar (both modes)
        self._build_log_pane()
        self._build_status_bar()

    # ---- Launcher panel (Pi-mode / bundled-mode) ----

    def _build_launcher_panel(self, parent: tk.Misc | None = None) -> None:
        frame = ttk.Frame(parent if parent is not None else self.root, padding=20)
        frame.pack(fill="both", expand=True, padx=14, pady=4)

        ttk.Label(
            frame,
            text=(
                "Click below to run the compression pipeline demo. The log "
                "streams progress as each stage runs, and the side-by-side "
                "comparison video opens in your default player when ready."
            ),
            wraplength=680,
            foreground="#555",
            justify="left",
        ).pack(anchor="w", pady=(0, 16))

        self.launcher_btn = ttk.Button(
            frame, text="Run Compression Pipeline Demo",
            command=self.run_bundled_demo,
        )
        self.launcher_btn.pack(pady=4, ipadx=12, ipady=4)
        self._action_buttons.append(self.launcher_btn)

        # Defensive: surface a friendly note if the bundle is incomplete.
        missing = []
        if not (ROOT / BUNDLED_DEMO_SCRIPT).exists():
            missing.append(BUNDLED_DEMO_SCRIPT)
        if not (ROOT / BUNDLED_COMPARISON_VIDEO).exists():
            missing.append(BUNDLED_COMPARISON_VIDEO)
        if missing:
            ttk.Label(
                frame,
                text=("Missing required file(s) in this folder: "
                      + ", ".join(missing)
                      + ". Re-extract the bundle and try again."),
                foreground="#c0392b",
                wraplength=680,
            ).pack(anchor="w", pady=(12, 0))
            self.launcher_btn.config(state="disabled")

    # ---- Tab 1: Live demo ----

    def _build_live_tab(self) -> None:
        frame = ttk.Frame(self.live_tab, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text=(
                "Launches scripts/live_demo.py in a separate OpenCV window. "
                "Step out of frame during the calibration banner, then step "
                "back in to see the saliency mask. Press 'q' in the window to quit."
            ),
            wraplength=820,
            foreground="#555",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        self.live_camera = tk.IntVar(value=0)
        self.live_saliency = tk.StringVar(value="yolo+spectral")
        self.live_bg_samples = tk.IntVar(value=60)
        self.live_auto_recal = tk.BooleanVar(value=True)

        ttk.Label(frame, text="Camera index:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Spinbox(frame, from_=0, to=8, width=6, textvariable=self.live_camera).grid(
            row=1, column=1, sticky="w", padx=4, pady=4,
        )

        ttk.Label(frame, text="Saliency backend:").grid(row=1, column=2, sticky="e", padx=(20, 4), pady=4)
        ttk.Combobox(
            frame, values=SALIENCY_CHOICES, textvariable=self.live_saliency,
            state="readonly", width=18,
        ).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Bg calibration frames:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ttk.Spinbox(frame, from_=15, to=300, width=6, textvariable=self.live_bg_samples).grid(
            row=2, column=1, sticky="w", padx=4, pady=4,
        )

        ttk.Checkbutton(
            frame,
            text="Enable rolling-median auto-recalibration (recommended)",
            variable=self.live_auto_recal,
        ).grid(row=2, column=2, columnspan=2, sticky="w", padx=(20, 4), pady=4)

        ttk.Separator(frame, orient="horizontal").grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=14,
        )

        self.launch_live_btn = ttk.Button(
            frame, text="Launch Live Demo Window", command=self.launch_live_demo,
        )
        self.launch_live_btn.grid(row=4, column=0, columnspan=4, pady=4)
        self._action_buttons.append(self.launch_live_btn)

        ttk.Label(
            frame,
            text="(Camera permission required. macOS will prompt the first time.)",
            foreground="#888", font=("Helvetica", 10),
        ).grid(row=5, column=0, columnspan=4, pady=(4, 0))

    # ---- Tab 2: Compression pipeline ----

    def _build_comp_tab(self) -> None:
        frame = ttk.Frame(self.comp_tab, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text=(
                "Pick a video, set the flags, click Run. The pipeline will "
                "(1) stage your video, (2) run ablation_bgfg.py to encode the "
                "baseline H.265 and ours_bgfg, and (3) build the 2-row "
                "side-by-side comparison via compare_clip.py --internals. "
                "Watch the log below to see progress."
            ),
            wraplength=820,
            foreground="#555",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        # --- File picker ---
        self.video_path = tk.StringVar(value="")
        self.clip_id    = tk.StringVar(value="")

        ttk.Label(frame, text="Input video:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        path_row = ttk.Frame(frame)
        path_row.grid(row=1, column=1, columnspan=3, sticky="ew", padx=4, pady=4)
        path_row.columnconfigure(1, weight=1)
        ttk.Button(path_row, text="Choose file…", command=self.pick_video).grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(path_row, textvariable=self.video_path, state="readonly").grid(
            row=0, column=1, sticky="ew",
        )

        ttk.Label(frame, text="Clip ID:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.clip_id, width=28).grid(
            row=2, column=1, sticky="w", padx=4, pady=4,
        )
        ttk.Label(
            frame, text="(used as data/real/<clip_id>.mp4)",
            foreground="#888", font=("Helvetica", 10),
        ).grid(row=2, column=2, columnspan=2, sticky="w", padx=4)

        # --- Encoder flags ---
        self.crf = tk.StringVar(value="22")
        self.comp_saliency = tk.StringVar(value="yolo+spectral")
        ttk.Label(frame, text="CRF:").grid(row=3, column=0, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, values=CRF_CHOICES, textvariable=self.crf,
                     state="readonly", width=6).grid(row=3, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Saliency backend:").grid(row=3, column=2, sticky="e", padx=(20, 4), pady=4)
        ttk.Combobox(frame, values=SALIENCY_CHOICES, textvariable=self.comp_saliency,
                     state="readonly", width=18).grid(row=3, column=3, sticky="w", padx=4, pady=4)

        # --- Background mode ---
        bg_box = ttk.LabelFrame(frame, text="Background reference", style="Section.TLabelframe", padding=10)
        bg_box.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(12, 6))
        bg_box.columnconfigure(3, weight=1)

        self.bg_mode = tk.StringVar(value="static")
        ttk.Radiobutton(
            bg_box, text="Static (1 clip-wide median — recommended for stable scenes)",
            variable=self.bg_mode, value="static", command=self._refresh_rolling_state,
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(
            bg_box, text="Rolling (re-median every N seconds — recommended for long clips with drift)",
            variable=self.bg_mode, value="rolling", command=self._refresh_rolling_state,
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=4, pady=2)

        self.bg_interval = tk.DoubleVar(value=4.0)
        self.bg_window   = tk.DoubleVar(value=6.0)

        ttk.Label(bg_box, text="Recal interval (s):").grid(row=2, column=0, sticky="e", padx=(28, 4), pady=4)
        self.bg_interval_spin = ttk.Spinbox(
            bg_box, from_=1.0, to=600.0, increment=1.0, width=8,
            textvariable=self.bg_interval, format="%.1f",
        )
        self.bg_interval_spin.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(bg_box, text="Window (s):").grid(row=2, column=2, sticky="e", padx=(20, 4), pady=4)
        self.bg_window_spin = ttk.Spinbox(
            bg_box, from_=1.0, to=600.0, increment=1.0, width=8,
            textvariable=self.bg_window, format="%.1f",
        )
        self.bg_window_spin.grid(row=2, column=3, sticky="w", padx=4, pady=4)

        # --- Comparison layout ---
        self.panel_scale = tk.StringVar(value="0.5")
        ttk.Label(frame, text="Comparison panel scale:").grid(row=5, column=0, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, values=PANEL_SCALES, textvariable=self.panel_scale,
                     state="readonly", width=6).grid(row=5, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(
            frame, text="(0.5 → 2880×1080 final grid; safe for laptops & Raspberry Pi)",
            foreground="#888", font=("Helvetica", 10),
        ).grid(row=5, column=2, columnspan=2, sticky="w", padx=4)

        # --- Action buttons ---
        actions = ttk.Frame(frame)
        actions.grid(row=6, column=0, columnspan=4, pady=(16, 6), sticky="ew")
        self.run_pipeline_btn = ttk.Button(
            actions, text="Run Full Pipeline", command=self.run_pipeline,
        )
        self.run_pipeline_btn.pack(side="left", padx=(0, 8))
        self._action_buttons.append(self.run_pipeline_btn)
        self.open_comparison_btn = ttk.Button(
            actions, text="Open Comparison Video", command=self.open_comparison, state="disabled",
        )
        self.open_comparison_btn.pack(side="left")
        # Not added to _action_buttons — its enabled/disabled state is driven
        # by pipeline-completion, not by the generic running/idle toggle.

        self._refresh_rolling_state()

    # ---- Shared log pane ----

    def _build_log_pane(self) -> None:
        log_frame = ttk.LabelFrame(self.root, text="Pipeline log", style="Section.TLabelframe", padding=8)
        log_frame.pack(fill="both", expand=True, padx=14, pady=(8, 4))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap="word", height=14,
            font=("Menlo", 11), background="#1d1f21", foreground="#e6e6e6",
            insertbackground="#e6e6e6", borderwidth=0,
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.config(state="disabled")

        # ANSI-ish accent tags so important lines pop
        self.log_text.tag_config("cmd",   foreground="#82aaff")   # blue for commands
        self.log_text.tag_config("ok",    foreground="#9ece6a")   # green for success
        self.log_text.tag_config("warn",  foreground="#e0af68")   # yellow for warnings
        self.log_text.tag_config("err",   foreground="#f7768e")   # red for errors
        self.log_text.tag_config("dim",   foreground="#888888")   # gray for status

    def _build_status_bar(self) -> None:
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=14, pady=(0, 12))

        self.status_var = tk.StringVar(value="idle")
        ttk.Label(status_frame, textvariable=self.status_var,
                  foreground="#666").pack(side="left")

        ttk.Button(status_frame, text="Clear log", command=self.clear_log).pack(side="right", padx=4)
        self.stop_btn = ttk.Button(
            status_frame, text="Stop running job", command=self.stop_current_job, state="disabled",
        )
        self.stop_btn.pack(side="right", padx=4)

    # ---- Log queue plumbing ----

    def _poll_log_queue(self) -> None:
        """Drain pending log lines onto the text widget (main thread only)."""
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__cb__":
                    item[1]()
                elif isinstance(item, tuple) and len(item) == 2:
                    self._append_log(item[0], tag=item[1])
                else:
                    self._append_log(str(item))
        except queue.Empty:
            pass
        self.root.after(self.POLL_MS, self._poll_log_queue)

    def _append_log(self, msg: str, tag: str | None = None) -> None:
        self.log_text.config(state="normal")
        if tag:
            self.log_text.insert("end", msg + "\n", tag)
        else:
            self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _log_q(self, msg: str, tag: str | None = None) -> None:
        """Thread-safe log enqueue. Workers must use this, never _append_log."""
        self.log_queue.put((msg, tag) if tag else msg)

    def _cb_q(self, callback) -> None:
        """Enqueue a callable to run on the main thread."""
        self.log_queue.put(("__cb__", callback))

    def clear_log(self) -> None:
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    # ---- UI state ----

    def _refresh_rolling_state(self) -> None:
        is_rolling = self.bg_mode.get() == "rolling"
        state = "normal" if is_rolling else "disabled"
        self.bg_interval_spin.config(state=state)
        self.bg_window_spin.config(state=state)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for btn in self._action_buttons:
            try:
                btn.config(state=state)
            except tk.TclError:
                pass
        self.stop_btn.config(state="normal" if running else "disabled")
        self._set_status("running…" if running else "idle")

    # ---- Subprocess runner ----

    def _stream_subprocess(self, cmd: list[str], cwd: Path = ROOT) -> int:
        """Run a subprocess synchronously (called from worker thread).

        Streams stdout+stderr line-by-line into the log queue. Returns the
        exit code. Sets self.current_proc so the Stop button can kill it.

        Force-unbuffers any Python child process — without this, Python
        detects its stdout is a pipe (not a TTY) and switches to ~4KB
        block buffering, which means short status lines sit in a buffer
        for minutes before the GUI ever sees them.
        """
        self._log_q(f"$ {' '.join(shlex.quote(c) for c in cmd)}", tag="cmd")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.current_proc = subprocess.Popen(
                cmd, cwd=str(cwd), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except FileNotFoundError as e:
            self._log_q(f"[error] {e}", tag="err")
            return 127

        assert self.current_proc.stdout is not None
        for line in self.current_proc.stdout:
            self._log_q(line.rstrip("\n"))
        self.current_proc.wait()
        rc = self.current_proc.returncode
        self.current_proc = None
        tag = "ok" if rc == 0 else "err"
        self._log_q(f"[exit {rc}]", tag=tag)
        return rc

    def _start_worker(self, target) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning(
                "Job in progress",
                "Another job is currently running. Stop it first or wait for it to finish.",
            )
            return
        self.worker_thread = threading.Thread(target=target, daemon=True)
        self.worker_thread.start()

    def stop_current_job(self) -> None:
        if self.current_proc is not None and self.current_proc.poll() is None:
            self._log_q("[stopping current subprocess…]", tag="warn")
            try:
                self.current_proc.terminate()
            except Exception as e:
                self._log_q(f"  (terminate failed: {e})", tag="warn")

    # ---- Launcher action (bundled mode) ----

    def run_bundled_demo(self) -> None:
        """Shell out to demo_pi.py (next to this script) and stream its output."""
        script = ROOT / BUNDLED_DEMO_SCRIPT
        video  = ROOT / BUNDLED_COMPARISON_VIDEO
        if not script.exists():
            messagebox.showerror("Missing demo script", f"Could not find {script}")
            return
        if not video.exists():
            messagebox.showerror("Missing comparison video", f"Could not find {video}")
            return
        self.last_comparison_path = video
        self._set_running(True)

        def worker():
            try:
                rc = self._stream_subprocess([sys.executable, "-u", str(script)])
                if rc == 0:
                    self._cb_q(lambda: self._set_status("done"))
                else:
                    self._cb_q(lambda: self._set_status("demo exited with non-zero status"))
            except Exception as e:
                self._log_q(f"[error] {e}", tag="err")
                self._cb_q(lambda: self._set_status("error — see log"))
            finally:
                self._cb_q(lambda: self._set_running(False))

        self._start_worker(worker)

    # ---- Tab 1 action ----

    def launch_live_demo(self) -> None:
        script = ROOT / "scripts" / "live_demo.py"
        if not script.exists():
            messagebox.showerror("Missing script", f"Could not find {script}")
            return
        cmd = [
            sys.executable, "-u", str(script),
            "--camera",     str(self.live_camera.get()),
            "--saliency",   self.live_saliency.get(),
            "--bg-samples", str(self.live_bg_samples.get()),
        ]
        if not self.live_auto_recal.get():
            cmd.append("--no-auto-recal")

        # Fire-and-forget — live_demo manages its own OpenCV window.
        self._log_q("Launching live demo window…", tag="dim")
        self._log_q(f"$ {' '.join(shlex.quote(c) for c in cmd)}", tag="cmd")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            subprocess.Popen(cmd, cwd=str(ROOT), env=env)
        except Exception as e:
            self._log_q(f"[error] {e}", tag="err")

    # ---- Tab 2 actions ----

    def pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v"),
                ("All files", "*.*"),
            ],
            initialdir=str(Path.home()),
        )
        if not path:
            return
        self.video_path.set(path)
        self.clip_id.set(derive_clip_id(Path(path)))
        self.open_comparison_btn.config(state="disabled")

    def open_comparison(self) -> None:
        if self.last_comparison_path is None or not self.last_comparison_path.exists():
            messagebox.showinfo("No comparison yet", "Run the pipeline first.")
            return
        open_externally(self.last_comparison_path)

    def run_pipeline(self) -> None:
        src_path = self.video_path.get().strip()
        clip_id  = self.clip_id.get().strip()
        if not src_path:
            messagebox.showinfo("Pick a video", "Choose an input video first.")
            return
        if not clip_id or not all(c.isalnum() or c == "_" for c in clip_id):
            messagebox.showerror(
                "Bad clip ID",
                "Clip ID must be non-empty and contain only letters, digits, underscores.",
            )
            return
        src = Path(src_path)
        if not src.exists():
            messagebox.showerror("Missing input", f"Video does not exist: {src}")
            return

        # Snapshot UI state into locals so the worker doesn't touch tk vars.
        crf          = self.crf.get()
        saliency     = self.comp_saliency.get()
        bg_mode      = self.bg_mode.get()
        bg_interval  = self.bg_interval.get()
        bg_window    = self.bg_window.get()
        panel_scale  = self.panel_scale.get()

        staged       = DATA_REAL / f"{clip_id}.mp4"
        compare_out  = COMPARES / f"{clip_id}_crf{crf}.mp4"

        self._set_running(True)
        self.open_comparison_btn.config(state="disabled")
        self.last_comparison_path = None

        def worker():
            try:
                DATA_REAL.mkdir(parents=True, exist_ok=True)
                ENCODED.mkdir(parents=True, exist_ok=True)
                COMPARES.mkdir(parents=True, exist_ok=True)

                # ---- Stage source ----
                self._log_q("", )
                self._log_q(f"[1/3] Staging source → data/real/{clip_id}.mp4 …", tag="dim")
                rc = self._stream_subprocess([
                    "ffmpeg", "-y", "-i", str(src), "-c:v", "copy", "-an", str(staged),
                ])
                if rc != 0:
                    raise RuntimeError("ffmpeg stage step failed")

                # ---- Run ablation_bgfg ----
                self._log_q("", )
                self._log_q(f"[2/3] Running ablation_bgfg.py …", tag="dim")
                cmd = [
                    sys.executable, "-u", str(ROOT / "scripts" / "ablation_bgfg.py"),
                    "--clips",    clip_id,
                    "--crfs",     crf,
                    "--saliency", saliency,
                    "--bg-mode",  bg_mode,
                    "--every",    "15",
                ]
                if bg_mode == "rolling":
                    cmd += [
                        "--bg-recal-interval", f"{bg_interval}",
                        "--bg-window",         f"{bg_window}",
                    ]
                rc = self._stream_subprocess(cmd)
                if rc != 0:
                    raise RuntimeError("ablation_bgfg step failed")

                # ---- Build comparison grid ----
                self._log_q("", )
                self._log_q(f"[3/3] Building 2-row comparison grid (--internals) …", tag="dim")
                cmd = [
                    sys.executable, "-u", str(ROOT / "scripts" / "compare_clip.py"),
                    "--clip",        clip_id,
                    "--crf",         crf,
                    "--internals",
                    "--panel-scale", panel_scale,
                    "--saliency",    saliency,
                    "--bg-mode",     bg_mode,
                ]
                if bg_mode == "rolling":
                    cmd += [
                        "--bg-recal-interval", f"{bg_interval}",
                        "--bg-window",         f"{bg_window}",
                    ]
                rc = self._stream_subprocess(cmd)
                if rc != 0:
                    raise RuntimeError("compare_clip step failed")

                self.last_comparison_path = compare_out
                self._log_q("", )
                self._log_q(f"[done] Comparison video → {compare_out}", tag="ok")
                self._cb_q(lambda: self.open_comparison_btn.config(state="normal"))
                self._cb_q(lambda: self._set_status(f"finished — {compare_out.name}"))

            except Exception as e:
                self._log_q(f"[pipeline error] {e}", tag="err")
                self._cb_q(lambda: self._set_status("error — see log"))
            finally:
                self._cb_q(lambda: self._set_running(False))

        self._start_worker(worker)

    # ---- Lifecycle ----

    def _on_close(self) -> None:
        if self.current_proc is not None and self.current_proc.poll() is None:
            if messagebox.askyesno(
                "Job still running",
                "A subprocess is still running. Quit anyway? It will be terminated.",
            ):
                try:
                    self.current_proc.terminate()
                except Exception:
                    pass
                self.root.destroy()
        else:
            self.root.destroy()


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

def main() -> int:
    root = tk.Tk()
    DemoConsole(root, launcher_mode=should_use_launcher_mode())
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
