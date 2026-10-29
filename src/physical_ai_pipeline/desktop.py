"""FrameWise native desktop workspace. No local web server or browser required."""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from physical_ai_pipeline.desktop_state import RESOLUTIONS, DesktopState, duration_label, parse_options, size_label
from physical_ai_pipeline.laptop import CleaningCancelled, clean_video

BG = "#f3f4f6"
PAPER = "#ffffff"
INK = "#20252d"
MUTED = "#68717e"
LINE = "#dde1e7"
ACCENT = "#275dc5"
_LOG = logging.getLogger(__name__)


def open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"This file or folder is no longer available:\n{path}")
    if sys.platform == "win32":
        os.startfile(str(path))
    else:
        subprocess.Popen(
            ["open" if sys.platform == "darwin" else "xdg-open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class FrameWiseApp:
    def __init__(self, root: tk.Tk, state_path: Path | None = None) -> None:
        self.root = root
        self.state_path = state_path or Path.cwd() / ".framewise" / "desktop.json"
        self.saved = DesktopState.load(self.state_path, Path.cwd() / "output")
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.loading = False
        self.closing = False
        self.source: Path | None = None
        self.source_info: dict[str, Any] | None = None
        self.preview_token = 0
        self.last_output: Path | None = None
        self.preview_image: tk.PhotoImage | None = None
        self.preview_size: tuple[int, int] | None = None
        self.controls: list[ttk.Widget] = []
        self.output = tk.StringVar(value=self.saved.output)
        self.resolution = tk.StringVar(value=self.saved.resolution)
        self.blur = tk.StringVar(value=self.saved.blur)
        self.duplicates = tk.BooleanVar(value=self.saved.duplicates)
        self.idle = tk.BooleanVar(value=self.saved.idle)
        self.audio = tk.BooleanVar(value=self.saved.audio)
        self.threads = tk.StringVar(value=self.saved.threads)
        self.filename = tk.StringVar(value="No video selected")
        self.file_detail = tk.StringVar(value="MP4, MOV, MKV, AVI and WebM")
        self.status = tk.StringVar(value="Ready")
        self.progress_detail = tk.StringVar(value="Choose a video to begin.")
        self.error = tk.StringVar()
        self.result = tk.StringVar(value="")
        self.metrics = [tk.StringVar(value="—") for _ in range(3)]
        self._configure()
        self._build()
        self._history_rows()
        root.bind("<Control-o>", lambda _event: self.choose_source())
        root.bind("<Control-Return>", lambda _event: self.start())
        root.bind("<Escape>", lambda _event: self.stop())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self._poll)

    @property
    def running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _configure(self) -> None:
        self.root.title("FrameWise")
        self.root.geometry("1180x980")
        self.root.minsize(980, 740)
        self.root.configure(bg=BG)
        families = {name.lower(): name for name in tkfont.families(self.root)}
        font = next(
            (families[name] for name in ("inter", "dejavu sans", "liberation sans", "arial") if name in families),
            "TkDefaultFont",
        )
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(name).configure(family=font, size=10)
        self.root.option_add("*TCombobox*Listbox.font", (font, 10))
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("Paper.TFrame", background=PAPER)
        s.configure("TLabel", background=BG, foreground=INK, font=(font, 10))
        s.configure("Paper.TLabel", background=PAPER)
        s.configure("Muted.TLabel", foreground=MUTED)
        s.configure("PaperMuted.TLabel", background=PAPER, foreground=MUTED, font=(font, 9))
        s.configure("Title.TLabel", font=(font, 20, "bold"))
        s.configure("Section.TLabel", background=PAPER, font=(font, 12, "bold"))
        s.configure("Value.TLabel", background=PAPER, font=(font, 11, "bold"))
        s.configure("Error.TLabel", foreground="#ad3535", background=BG, font=(font, 10))
        s.configure(
            "TButton",
            padding=(12, 8),
            background=PAPER,
            foreground=INK,
            bordercolor=LINE,
            lightcolor=PAPER,
            darkcolor=PAPER,
            font=(font, 10),
        )
        s.map(
            "TButton", background=[("active", "#edf1f7"), ("disabled", "#f1f2f4")], foreground=[("disabled", "#979da7")]
        )
        s.configure(
            "Primary.TButton",
            background=ACCENT,
            foreground=PAPER,
            bordercolor=ACCENT,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            font=(font, 10, "bold"),
        )
        s.map(
            "Primary.TButton",
            background=[("disabled", "#d7deeb"), ("active", "#204fa9")],
            foreground=[("disabled", "#7e8ca5")],
        )
        s.configure("TEntry", padding=8, fieldbackground=PAPER, bordercolor=LINE, foreground=INK)
        s.map("TEntry", bordercolor=[("focus", ACCENT)])
        s.configure("TCombobox", padding=7, fieldbackground=PAPER, foreground=INK, bordercolor=LINE)
        s.map(
            "TCombobox",
            fieldbackground=[("readonly", PAPER)],
            selectbackground=[("readonly", PAPER)],
            selectforeground=[("readonly", INK)],
        )
        s.configure("TCheckbutton", background=PAPER, foreground=INK, padding=(0, 4), font=(font, 10))
        s.map("TCheckbutton", background=[("active", PAPER)], foreground=[("disabled", "#979da7")])
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(16, 10), font=(font, 10))
        s.map("TNotebook.Tab", background=[("selected", PAPER)], foreground=[("selected", INK)])
        s.configure(
            "Horizontal.TProgressbar",
            background=ACCENT,
            troughcolor="#e5e8ee",
            borderwidth=0,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )
        s.configure(
            "Treeview",
            background=PAPER,
            fieldbackground=PAPER,
            foreground=INK,
            rowheight=36,
            borderwidth=0,
            font=(font, 10),
        )
        s.configure("Treeview.Heading", background="#edf0f4", foreground=MUTED, padding=10, font=(font, 9, "bold"))
        s.map("Treeview", background=[("selected", "#e4edfc")], foreground=[("selected", INK)])

    def _card(self, parent: ttk.Frame) -> ttk.Frame:
        border = tk.Frame(parent, bg=LINE, bd=0)
        border.pack(fill="both", expand=True, pady=(0, 14))
        card = ttk.Frame(border, style="Paper.TFrame", padding=20)
        card.pack(fill="both", expand=True, padx=1, pady=1)
        return card

    def _label(self, parent: Any, text: str, style: str = "Paper.TLabel", **kwargs: Any) -> ttk.Label:
        label = ttk.Label(parent, text=text, style=style, **kwargs)
        label.pack(anchor="w", pady=(0, 7))
        return label

    def _button(self, parent: Any, text: str, command: Any, **kwargs: Any) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command, **kwargs)
        self.controls.append(button)
        return button

    def _build(self) -> None:
        shell = ttk.Frame(self.root, padding=(28, 22, 28, 14))
        shell.pack(fill="both", expand=True)
        head = ttk.Frame(shell)
        head.pack(fill="x", pady=(0, 18))
        mark = tk.Canvas(head, width=38, height=38, bg=BG, highlightthickness=0)
        mark.pack(side="left", padx=(0, 12))
        mark.create_rectangle(1, 1, 37, 37, fill=INK, outline=INK)
        for x, y, dx, dy in ((10, 10, 1, 1), (28, 10, -1, 1), (10, 28, 1, -1), (28, 28, -1, -1)):
            mark.create_line([x, y + dy * 6, x, y, x + dx * 6, y], fill=PAPER, width=2)
        ttk.Label(head, text="FrameWise", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="VIDEO PREPARATION", style="Muted.TLabel").pack(side="left", padx=20)
        ttk.Button(head, text="Help", command=self.help).pack(side="right")
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill="both", expand=True)
        self.workspace = ttk.Frame(self.tabs, padding=(0, 18, 0, 0))
        history_page = ttk.Frame(self.tabs, padding=(0, 18, 0, 0))
        self.tabs.add(self.workspace, text="New export")
        self.tabs.add(history_page, text="Recent exports")
        scroller = ttk.Frame(self.workspace)
        scroller.pack(fill="both", expand=True)
        viewport = tk.Canvas(scroller, bg=BG, highlightthickness=0)
        viewport.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(scroller, orient="vertical", command=viewport.yview)
        scroll.pack(side="right", fill="y")
        viewport.configure(yscrollcommand=scroll.set)
        columns = ttk.Frame(viewport)
        content = viewport.create_window(0, 0, window=columns, anchor="nw")
        columns.bind("<Configure>", lambda _event: viewport.configure(scrollregion=viewport.bbox("all")))

        def resize_viewport(event: Any) -> None:
            viewport.itemconfigure(content, width=event.width)
            self.root.after_idle(lambda: viewport.yview_moveto(0))

        viewport.bind("<Configure>", resize_viewport)

        def scroll_wheel(event: Any) -> None:
            if self.workspace.winfo_viewable():
                viewport.yview_scroll(-1 if event.num == 4 or event.delta > 0 else 1, "units")

        self.root.bind("<MouseWheel>", scroll_wheel)
        self.root.bind("<Button-4>", scroll_wheel)
        self.root.bind("<Button-5>", scroll_wheel)
        columns.columnconfigure(0, weight=3)
        columns.columnconfigure(1, weight=2)
        columns.rowconfigure(0, weight=1)
        left = ttk.Frame(columns)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        right = ttk.Frame(columns)
        right.grid(row=0, column=1, sticky="nsew")
        source_card = self._card(left)
        source_head = ttk.Frame(source_card, style="Paper.TFrame")
        source_head.pack(fill="x", pady=(0, 12))
        ttk.Label(source_head, text="Source video", style="Section.TLabel").pack(side="left")
        self._button(source_head, "Choose video…", self.choose_source).pack(side="right")
        self._button(source_head, "Paste path", self.paste_source).pack(side="right", padx=8)
        self.preview = tk.Canvas(source_card, bg="#e9edf2", height=260, highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", lambda _event: self._draw_preview())
        ttk.Label(source_card, textvariable=self.filename, style="Value.TLabel").pack(anchor="w", pady=(14, 4))
        self.file_label = ttk.Label(
            source_card, textvariable=self.file_detail, style="PaperMuted.TLabel", wraplength=520
        )
        self.file_label.pack(anchor="w")
        facts = ttk.Frame(source_card, style="Paper.TFrame")
        facts.pack(fill="x", pady=(18, 0))
        for i, title in enumerate(("DURATION", "FRAME SIZE", "FRAME RATE")):
            block = ttk.Frame(facts, style="Paper.TFrame")
            block.pack(side="left", fill="x", expand=True)
            ttk.Label(block, text=title, style="PaperMuted.TLabel").pack(anchor="w")
            ttk.Label(block, textvariable=self.metrics[i], style="Value.TLabel").pack(anchor="w", pady=(4, 0))
        destination = self._card(left)
        self._label(destination, "Save location", "Section.TLabel")
        line = ttk.Frame(destination, style="Paper.TFrame")
        line.pack(fill="x", pady=(3, 8))
        self.output_entry = ttk.Entry(line, textvariable=self.output)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.controls.append(self.output_entry)
        self._button(line, "Browse…", self.choose_output).pack(side="right")
        self._label(destination, "A separate folder is created for every export.", "PaperMuted.TLabel")

        settings = self._card(right)
        self._label(settings, "Export settings", "Section.TLabel")
        self._label(settings, "H.264 video · MP4 container", "PaperMuted.TLabel")
        self._label(settings, "Output size")
        self.size_combo = ttk.Combobox(
            settings, values=list(RESOLUTIONS), textvariable=self.resolution, state="readonly", width=26
        )
        self.size_combo.pack(fill="x", pady=(0, 7))
        self.controls.append(self.size_combo)
        self._label(
            settings, "Aspect ratio is preserved. Videos are never upscaled.", "PaperMuted.TLabel", wraplength=330
        )
        ttk.Separator(settings).pack(fill="x", pady=10)
        self._label(settings, "Quality filters", "Section.TLabel")
        self._label(settings, "Exposure and corrupt-frame checks are always on.", "PaperMuted.TLabel", wraplength=340)
        blur_row = ttk.Frame(settings, style="Paper.TFrame")
        blur_row.pack(fill="x", pady=(3, 6))
        ttk.Label(blur_row, text="Blur threshold", style="Paper.TLabel").pack(side="left")
        blur_entry = ttk.Entry(blur_row, textvariable=self.blur, width=8, justify="right")
        blur_entry.pack(side="right")
        self.controls.append(blur_entry)
        self._label(settings, "Lower keeps more frames. Set 0 to turn off.", "PaperMuted.TLabel")
        for label, variable in (
            ("Remove near-duplicate frames", self.duplicates),
            ("Remove static stretches", self.idle),
            ("Include audio", self.audio),
        ):
            button = ttk.Checkbutton(settings, text=label, variable=variable)
            button.pack(anchor="w", pady=2)
            self.controls.append(button)
        self._label(
            settings, "Audio follows the cuts and is exported as mono AAC.", "PaperMuted.TLabel", wraplength=340
        )
        ttk.Separator(settings).pack(fill="x", pady=10)
        cpu = ttk.Frame(settings, style="Paper.TFrame")
        cpu.pack(fill="x", pady=(0, 7))
        ttk.Label(cpu, text="CPU threads", style="Paper.TLabel").pack(side="left")
        self.thread_combo = ttk.Combobox(
            cpu, values=["1", "2", "3", "4"], textvariable=self.threads, width=5, state="readonly"
        )
        self.thread_combo.pack(side="right")
        self.controls.append(self.thread_combo)
        self._label(settings, "2 threads is a good starting point for a laptop.", "PaperMuted.TLabel", wraplength=340)
        ttk.Label(
            right,
            text="Quality cleaning runs on your device.\nSemantic AI detection is not enabled in this mode.",
            style="Muted.TLabel",
            wraplength=390,
        ).pack(anchor="w", padx=2, pady=(0, 10))

        bottom = ttk.Frame(self.workspace)
        bottom.pack(side="bottom", fill="x", pady=(6, 3), before=scroller)
        ttk.Label(bottom, textvariable=self.error, style="Error.TLabel", wraplength=1050).pack(anchor="w", pady=(0, 6))
        status_row = ttk.Frame(bottom)
        status_row.pack(fill="x")
        ttk.Label(status_row, textvariable=self.status).pack(side="left")
        self.start_button = ttk.Button(
            status_row, text="Export video", style="Primary.TButton", command=self.start, state="disabled"
        )
        self.start_button.pack(side="right")
        self.stop_button = ttk.Button(status_row, text="Cancel", command=self.stop, state="disabled")
        self.stop_button.pack(side="right", padx=10)
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(12, 8))
        ttk.Label(bottom, textvariable=self.progress_detail, style="Muted.TLabel").pack(anchor="w")
        result_row = ttk.Frame(bottom)
        result_row.pack(fill="x", pady=(8, 0))
        ttk.Label(result_row, textvariable=self.result, style="Muted.TLabel", wraplength=540).pack(side="left")
        self.folder_button = ttk.Button(
            result_row, text="Open folder", command=lambda: self.open_result("folder"), state="disabled"
        )
        self.folder_button.pack(side="right")
        self.play_button = ttk.Button(
            result_row, text="Play result", command=lambda: self.open_result("video"), state="disabled"
        )
        self.play_button.pack(side="right", padx=8)
        self.report_button = ttk.Button(
            result_row, text="View report", command=lambda: self.open_result("report"), state="disabled"
        )
        self.report_button.pack(side="right")
        ttk.Label(
            shell,
            text="FrameWise 1.1  ·  Local quality workspace          Ctrl+O  Open video     Ctrl+Enter  Export     Esc  Cancel",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(15, 0))
        self._build_history(history_page)

    def _build_history(self, page: ttk.Frame) -> None:
        ttk.Label(page, text="Recent exports", style="Title.TLabel").pack(anchor="w", pady=(0, 7))
        ttk.Label(
            page, text="Your last 20 exports. Files stay in the output folders you chose.", style="Muted.TLabel"
        ).pack(anchor="w", pady=(0, 18))
        holder = ttk.Frame(page)
        holder.pack(fill="both", expand=True)
        self.history = ttk.Treeview(
            holder, columns=("source", "status", "frames", "time"), show="headings", selectmode="browse"
        )
        for key, label, width in (
            ("source", "Video", 420),
            ("status", "Status", 160),
            ("frames", "Frames kept", 160),
            ("time", "Processing time", 150),
        ):
            self.history.heading(key, text=label)
            self.history.column(key, width=width, minwidth=100)
        self.history.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(holder, orient="vertical", command=self.history.yview)
        scrollbar.pack(side="right", fill="y")
        self.history.configure(yscrollcommand=scrollbar.set)
        self.history.bind("<Double-1>", lambda _event: self.open_history())
        row = ttk.Frame(page)
        row.pack(fill="x", pady=12)
        ttk.Label(row, text="Double-click an export to open its folder.", style="Muted.TLabel").pack(side="left")
        ttk.Button(row, text="Open selected folder", command=self.open_history).pack(side="right")

    def _history_rows(self) -> None:
        self.history.delete(*self.history.get_children())
        for index, row in reversed(list(enumerate(self.saved.history))):
            self.history.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    Path(row["source"]).name,
                    "Exported" if row.get("status") == "completed" else "No frames kept",
                    f"{row.get('kept', 0):,} / {row.get('total', 0):,}",
                    f"{row.get('elapsed', 0):.1f} s",
                ),
            )

    def _draw_preview(self) -> None:
        self.preview.delete("all")
        width, height = self.preview.winfo_width(), self.preview.winfo_height()
        if self.source_info is not None and (width, height) != self.preview_size:
            import cv2
            import numpy as np

            pixels = cv2.imdecode(np.frombuffer(self.source_info["preview"], dtype=np.uint8), cv2.IMREAD_COLOR)
            assert pixels is not None  # This PNG was encoded successfully by the preview worker.
            h, w = pixels.shape[:2]
            scale = min(max(1, width - 12) / w, max(1, height - 12) / h, 1)
            pixels = cv2.resize(
                pixels, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
            )
            ok, encoded = cv2.imencode(".png", pixels)
            if ok:
                self.preview_image = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()))
                self.preview_size = (width, height)
        if self.preview_image is not None:
            self.preview.create_image(width / 2, height / 2, image=self.preview_image)
        else:
            self.preview.create_rectangle(
                width / 2 - 22, height / 2 - 32, width / 2 + 22, height / 2, outline="#9aa5b4", width=2
            )
            self.preview.create_polygon(
                width / 2 - 4,
                height / 2 - 25,
                width / 2 - 4,
                height / 2 - 7,
                width / 2 + 10,
                height / 2 - 16,
                fill="#9aa5b4",
                outline="",
            )
            self.preview.create_text(
                width / 2,
                height / 2 + 24,
                text="Opening preview…" if self.loading else "Choose a video to preview",
                fill=MUTED,
                font="TkDefaultFont",
            )

    def choose_source(self) -> None:
        if self.running or self.closing:
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.MP4 *.MOV *.MKV *.AVI *.WEBM"),
                ("All files", "*"),
            ],
        )
        if path:
            self.select_source(Path(path))

    def paste_source(self) -> None:
        if self.running:
            return
        try:
            path = self.root.clipboard_get().strip().strip('"')
            self.select_source(Path(path))
        except tk.TclError:
            self.error.set("Copy a video file path, then choose Paste path.")

    def select_source(self, path: Path) -> None:
        if self.running:
            return
        path = path.expanduser().resolve()
        if not path.is_file():
            self.error.set("The selected file could not be found. Choose an existing video.")
            return
        self.source, self.source_info = path, None
        self.last_output = None
        self.result.set("")
        self.error.set("")
        self.filename.set(path.name)
        self.file_detail.set(str(path))
        self.preview_image = None
        self.preview_size = None
        self.preview_token += 1
        self.loading = True
        self.start_button.configure(state="disabled")
        self._result_actions(False)
        self.status.set("Reading video…")
        self.progress_detail.set("Checking the video stream and loading a preview.")
        self.progress["value"] = 0
        for metric in self.metrics:
            metric.set("—")
        self._draw_preview()
        token = self.preview_token

        def load() -> None:
            try:
                import av
                import cv2

                from physical_ai_pipeline.decode.video_reader import probe

                info = probe(path)
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 1
                    frame = next(container.decode(video=0), None)
                    if frame is None:
                        raise ValueError("This file contains no readable video frames.")
                    scale = min(580 / frame.width, 255 / frame.height, 1)
                    frame = frame.reformat(
                        width=max(1, round(frame.width * scale)), height=max(1, round(frame.height * scale))
                    )
                    ok, encoded = cv2.imencode(".png", frame.to_ndarray(format="bgr24"))
                    if not ok:
                        raise ValueError("The video preview could not be decoded.")
                    data = {
                        "width": info.width,
                        "height": info.height,
                        "fps": info.fps,
                        "frames": info.frame_count,
                        "duration": info.duration_s,
                        "codec": info.codec,
                        "audio": bool(container.streams.audio),
                        "size": path.stat().st_size,
                        "preview": encoded.tobytes(),
                    }
                self.events.put(("loaded", (token, data)))
            except Exception as exc:
                _LOG.exception("Could not read video %s", path)
                self.events.put(("load_error", (token, str(exc))))

        threading.Thread(target=load, name="video-preview", daemon=True).start()

    def choose_output(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title="Choose output folder", mustexist=False)
        if path:
            self.output.set(path)

    def _save(self) -> None:
        self.saved.output, self.saved.resolution, self.saved.blur = (
            self.output.get(),
            self.resolution.get(),
            self.blur.get(),
        )
        self.saved.duplicates, self.saved.idle = self.duplicates.get(), self.idle.get()
        self.saved.audio, self.saved.threads = self.audio.get(), self.threads.get()
        try:
            self.saved.save(self.state_path)
        except OSError:
            _LOG.exception("Could not save desktop preferences")
            self.error.set("Export files are safe, but preferences could not be saved in this project folder.")

    def _busy(self, value: bool) -> None:
        for widget in self.controls:
            widget.state(["disabled"] if value else ["!disabled"])
        if not value:
            self.size_combo.configure(state="readonly")
            self.thread_combo.configure(state="readonly")
        self.start_button.configure(state="disabled" if value or self.source_info is None else "normal")
        self.stop_button.configure(state="normal" if value else "disabled")

    def _result_actions(self, available: bool, video: bool = False) -> None:
        for button in (self.folder_button, self.report_button):
            button.configure(state="normal" if available else "disabled")
        self.play_button.configure(state="normal" if available and video else "disabled")

    def start(self) -> None:
        if self.running or self.loading or self.closing:
            return
        self.error.set("")
        try:
            if self.source is None or self.source_info is None:
                raise ValueError("Choose a readable input video first.")
            if not self.output.get().strip():
                raise ValueError("Choose a folder for the exported files.")
            destination = Path(self.output.get()).expanduser().resolve()
            if destination.exists() and not destination.is_dir():
                raise ValueError("The save location is a file. Choose a folder instead.")
            options = parse_options(
                self.resolution.get(),
                self.blur.get(),
                self.duplicates.get(),
                self.idle.get(),
                self.audio.get(),
                self.threads.get(),
            )
        except ValueError as exc:
            self.error.set(str(exc))
            return
        self._save()
        self.last_output = None
        self.result.set("")
        self._result_actions(False)
        self.cancel.clear()
        self._busy(True)
        self.status.set("Preparing export…")
        self.progress_detail.set("Opening the source and creating a new export folder.")
        self.progress["value"] = 0
        source = self.source

        def run() -> None:
            try:
                folder = clean_video(
                    source,
                    destination,
                    options,
                    progress=lambda data: self.events.put(("progress", data)),
                    cancel=self.cancel,
                )
                report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                self.events.put(("done", (folder, report)))
            except CleaningCancelled:
                self.events.put(("cancelled", None))
            except Exception as exc:
                _LOG.exception("Export failed for %s", source)
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=run, name="framewise-export", daemon=False)
        self.worker.start()

    def stop(self) -> None:
        if self.running:
            self.cancel.set()
            self.stop_button.configure(state="disabled")
            self.status.set("Cancelling…")
            self.progress_detail.set("Finishing the current operation and removing incomplete video files.")

    def _handle(self, kind: str, data: Any) -> None:
        if kind in {"loaded", "load_error"}:
            token, value = data
            if token != self.preview_token:
                return
            self.loading = False
            if kind == "load_error":
                self.source_info = None
                self.status.set("Could not open video")
                self.error.set(value)
                self.progress_detail.set("Choose another file or check whether the video plays in your media player.")
            else:
                self.source_info = value
                self.preview_image = tk.PhotoImage(data=base64.b64encode(value["preview"]))
                self.metrics[0].set(duration_label(value["duration"]))
                self.metrics[1].set(f"{value['width']} × {value['height']}")
                self.metrics[2].set(f"{value['fps']:.2f} fps")
                self.file_detail.set(
                    f"{size_label(value['size'])}   ·   {value['codec'].upper()}   ·   "
                    + ("Audio track available" if value["audio"] else "No audio track")
                )
                self.status.set("Ready to export")
                self.progress_detail.set("Your source file stays unchanged. Select export settings, then Export video.")
                self.start_button.configure(state="normal")
            self._draw_preview()
            return
        if kind == "progress":
            if self.cancel.is_set():
                return
            processed, total, elapsed = data["processed"], data.get("total"), data["elapsed"]
            audio_phase = data["phase"] == "Saving audio"
            self.status.set("Saving audio…" if audio_phase else "Exporting video…")
            if total:
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.progress["value"] = min(
                    99, (90 + processed / total * 9) if audio_phase else processed / total * 90
                )
            elif str(self.progress["mode"]) != "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start()
            if audio_phase:
                self.progress_detail.set(
                    f"Matching audio to the retained frames   ·   {duration_label(elapsed)} elapsed"
                )
            else:
                speed = processed / max(elapsed, 0.01)
                eta = (
                    f"   ·   about {duration_label((total - processed) / speed)} remaining"
                    if total and speed and total > processed
                    else ""
                )
                self.progress_detail.set(
                    f"{processed:,} frames checked   ·   {data['kept']:,} kept   ·   {speed:.1f} fps" + eta
                )
            return
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self._busy(False)
        if kind == "done":
            folder, report = data
            self.last_output = folder
            self.progress["value"] = 100
            kept, total = report["kept_frames"], report["total_frames"]
            self.status.set("Export complete" if kept else "No frames passed the filters")
            self.progress_detail.set(
                f"{kept:,} of {total:,} frames kept   ·   {report['elapsed_seconds']:.1f} s processing time"
            )
            self.result.set(str(folder))
            if not kept:
                self.error.set(
                    "Only reports were saved. Lower the blur threshold or turn off optional filters and try again."
                )
            self._result_actions(True, video=bool(kept))
            self.saved.add_export(folder, report)
            self._save()
            self._history_rows()
        elif kind == "cancelled":
            self.status.set("Export cancelled")
            self.progress_detail.set(
                "Incomplete video files were removed. The run folder contains its audit and status report."
            )
        else:
            self.status.set("Export failed")
            self.error.set(data or "An unexpected error occurred.")
            self.progress_detail.set("Check the input, output folder and available disk space, then retry.")

    def _poll(self) -> None:
        try:
            while True:
                kind, data = self.events.get_nowait()
                self._handle(kind, data)
        except queue.Empty:
            pass
        except Exception:
            _LOG.exception("Desktop event failed")
            self.error.set("The interface could not update. See .framewise/framewise.log for details.")
        if self.closing and not self.running:
            self.root.destroy()
        else:
            self.root.after(100, self._poll)

    def open_history(self) -> None:
        selected = self.history.selection()
        if selected:
            self._open(Path(self.saved.history[int(selected[0])]["folder"]))

    def _open(self, path: Path) -> None:
        try:
            open_path(path)
        except OSError as exc:
            self.error.set(str(exc))
            messagebox.showerror("Could not open result", str(exc), parent=self.root)

    def open_result(self, target: str) -> None:
        if self.last_output is None:
            return
        if target == "report":
            try:
                report = json.loads((self.last_output / "report.json").read_text(encoding="utf-8"))
                reasons = report.get("dropped_by_reason", {})
                lines = [
                    "EXPORT SUMMARY",
                    "",
                    f"Input: {Path(report['source']).name}",
                    f"Kept: {report['kept_frames']:,} / {report['total_frames']:,} frames",
                    f"Processing time: {report['elapsed_seconds']:.1f} s",
                    f"Audio: {'Included (mono AAC)' if report['audio_preserved'] else 'Not included'}",
                    "",
                    "DROPPED FRAMES",
                ]
                lines.extend(f"{key.replace('_', ' ').title()}: {value:,}" for key, value in reasons.items())
                if not reasons:
                    lines.append("No frames were dropped.")
                lines.extend(
                    ["", "Quality checks only. Semantic AI checks were not performed.", "", str(self.last_output)]
                )
                messagebox.showinfo("Export report", "\n".join(lines), parent=self.root)
            except (OSError, ValueError, KeyError) as exc:
                self.error.set(f"Could not read the export report: {exc}")
        else:
            self._open(self.last_output / "cleaned.mp4" if target == "video" else self.last_output)

    def help(self) -> None:
        messagebox.showinfo(
            "Using FrameWise",
            "1. Choose a video, or paste its file path.\n"
            "2. Choose where to save the export and adjust the quality settings.\n"
            "3. Select Export video. Open the result or review the report when it finishes.\n\n"
            "Each export gets a new folder with cleaned.mp4, report.json and frames.jsonl.\n"
            "Lower blur thresholds retain more frames. A threshold of 0 disables blur removal.\n"
            "Static filtering adds CPU work. Audio is cut to match the video and exported as mono AAC.\n\n"
            "This workspace checks video quality. It does not run semantic AI models.\n"
            "Preferences, recent exports and diagnostic logs are stored in .framewise/.",
            parent=self.root,
        )

    def close(self) -> None:
        self.closing = True
        self.preview_token += 1
        self._save()
        if self.running:
            self.stop()
            self.status.set("Closing after cancellation…")


def main() -> int:
    state_dir = Path.cwd() / ".framewise"
    try:
        state_dir.mkdir(exist_ok=True)
        handler = RotatingFileHandler(state_dir / "framewise.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        _LOG.addHandler(handler)
        _LOG.setLevel(logging.INFO)
    except OSError:
        logging.basicConfig(level=logging.WARNING)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"FrameWise needs a desktop display: {exc}", file=sys.stderr)
        return 1
    FrameWiseApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
