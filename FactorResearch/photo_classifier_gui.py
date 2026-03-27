"""
photo_classifier_gui.py — Desktop UI for photo_classifier.py
Run with:  python photo_classifier_gui.py
Requires:  Python 3.9+ (tkinter is bundled with Python)
photos.py must be in the same folder (or update SCRIPT_PATH below).
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import sys
import os
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "photo_classifier.py"

# Detect venv Python in the same folder as this script
_venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

# ── Colour tokens (dark theme) ──────────────────────────────────────────────
BG        = "#0f0f1a"
BG2       = "#1a1a2e"
BG3       = "#252540"
ACCENT    = "#4ecca3"
ACCENT2   = "#e23e57"
TEXT      = "#e0e0f0"
TEXT2     = "#888899"
BORDER    = "#2a2a44"
GOOD      = "#4ecca3"
BAD       = "#e23e57"
WARN      = "#f0a500"
FONT_BODY = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 9)
FONT_H1   = ("Segoe UI Semibold", 13)
FONT_H2   = ("Segoe UI Semibold", 10)
FONT_TINY = ("Segoe UI", 8)

PROFILES = {
    "portrait":  dict(sharpness=20, brightness=10, face=25, eyes=15, composition=10, emotion=10, noise=5,  color=5),
    "group":     dict(sharpness=20, brightness=10, face=15, eyes=10, composition=15, emotion=10, noise=10, color=10),
    "landscape": dict(sharpness=25, brightness=15, face=0,  eyes=0,  composition=20, emotion=0,  noise=15, color=25),
}

WEIGHT_ORDER = ["sharpness","brightness","face","eyes","composition","emotion","noise","color"]
WEIGHT_COLORS = {
    "sharpness":"#BA7517","brightness":"#185FA5","face":"#D4537E","eyes":"#1D9E75",
    "composition":"#7F77DD","emotion":"#D85A30","noise":"#888780","color":"#3B6D11",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_frame(parent, **kw):
    return tk.Frame(parent, bg=BG2, **kw)

def label(parent, text, font=FONT_BODY, fg=TEXT, **kw):
    kw.setdefault("bg", parent["bg"])
    return tk.Label(parent, text=text, font=font, fg=fg, **kw)

def styled_entry(parent, textvariable, width=28):
    e = tk.Entry(parent, textvariable=textvariable, width=width,
                 bg=BG3, fg=TEXT, insertbackground=TEXT, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, font=FONT_BODY)
    return e

def styled_button(parent, text, command, color=ACCENT, width=14):
    btn = tk.Button(parent, text=text, command=command,
                    bg=color, fg=BG, font=FONT_H2,
                    activebackground=TEXT, activeforeground=BG,
                    relief="flat", cursor="hand2", width=width, pady=5)
    return btn

def divider(parent):
    return tk.Frame(parent, bg=BORDER, height=1)

def section_label(parent, text):
    f = tk.Frame(parent, bg=parent["bg"])
    label(f, text.upper(), font=FONT_TINY, fg=TEXT2).pack(side="left")
    tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(8,0), pady=6)
    return f


# ── Main App ─────────────────────────────────────────────────────────────────

class PhotoClassifierGUI:
    def __init__(self, root):
        self.root = root
        root.title("Photo Classifier")
        root.configure(bg=BG)
        root.geometry("860x700")
        root.minsize(720, 560)

        self._init_vars()
        self._build_ui()
        self._refresh_command()
        self._process = None

    # ── Variables ────────────────────────────────────────────────────────────

    def _init_vars(self):
        self.profile   = tk.StringVar(value="portrait")
        self.input_dir = tk.StringVar(value="input_photos")
        self.good_dir  = tk.StringVar(value="good_photos")
        self.bad_dir   = tk.StringVar(value="bad_photos")
        self.min_score = tk.IntVar(value=60)
        self.workers   = tk.IntVar(value=4)
        self.device    = tk.StringVar(value="cpu")
        self.dry_run   = tk.BooleanVar(value=False)
        self.report    = tk.BooleanVar(value=False)
        self.html_rep  = tk.BooleanVar(value=False)
        self.no_cache  = tk.BooleanVar(value=False)

        self.weights = {k: tk.IntVar(value=v)
                        for k, v in PROFILES["portrait"].items()}

        # trace for live command preview
        for v in [self.profile, self.input_dir, self.good_dir, self.bad_dir,
                  self.min_score, self.workers, self.device,
                  self.dry_run, self.report, self.html_rep, self.no_cache,
                  *self.weights.values()]:
            v.trace_add("write", lambda *_: self._refresh_command())

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Title bar ──
        title_bar = tk.Frame(self.root, bg=BG, pady=12)
        title_bar.pack(fill="x", padx=20)
        label(title_bar, "Photo Classifier", font=FONT_H1, fg=TEXT, bg=BG).pack(side="left")
        label(title_bar, f"v2  ·  {PYTHON}",
              fg=TEXT2, bg=BG).pack(side="left", padx=12)

        # ── Main pane: left config + right log ──
        pane = tk.PanedWindow(self.root, orient="horizontal",
                              bg=BG, sashwidth=6, sashrelief="flat")
        pane.pack(fill="both", expand=True, padx=12, pady=(0,12))

        left  = self._build_left(pane)
        right = self._build_right(pane)
        pane.add(left,  minsize=380, width=440)
        pane.add(right, minsize=300)

    def _build_left(self, parent):
        frame = tk.Frame(parent, bg=BG)

        nb = ttk.Notebook(frame)
        nb.pack(fill="both", expand=True, pady=(0,8))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",        background=BG,  borderwidth=0)
        style.configure("TNotebook.Tab",    background=BG2, foreground=TEXT2,
                        padding=[12,5], font=FONT_BODY)
        style.map("TNotebook.Tab",
                  background=[("selected", BG3)],
                  foreground=[("selected", TEXT)])

        nb.add(self._tab_profile(nb),    text=" Profile & Paths ")
        nb.add(self._tab_weights(nb),    text=" Scoring Weights ")
        nb.add(self._tab_thresholds(nb), text=" Thresholds ")
        nb.add(self._tab_processing(nb), text=" Processing ")

        # ── Command preview ──
        cmd_frame = make_frame(frame)
        cmd_frame.pack(fill="x")
        cmd_frame.configure(bd=0, highlightthickness=1,
                             highlightbackground=BORDER)
        top = tk.Frame(cmd_frame, bg=BG3, pady=5)
        top.pack(fill="x")
        label(top, "Generated command", fg=TEXT2, bg=BG3,
              font=FONT_TINY).pack(side="left", padx=10)
        styled_button(top, "Copy", self._copy_command,
                      color=BG3, width=6).pack(side="right", padx=6)
        self.cmd_text = tk.Text(cmd_frame, height=4, bg=BG2, fg=ACCENT,
                                font=FONT_MONO, relief="flat",
                                state="disabled", wrap="none",
                                insertbackground=ACCENT,
                                selectbackground=BG3)
        self.cmd_text.pack(fill="x", padx=8, pady=8)

        return frame

    def _scroll_frame(self, parent):
        """Scrollable inner frame for a tab."""
        canvas = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG2)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
        return inner

    # ── Tab: Profile & Paths ──────────────────────────────────────────────────

    def _tab_profile(self, parent):
        tab = tk.Frame(parent, bg=BG2)
        inner = self._scroll_frame(tab)
        p = inner; p.configure(padx=16, pady=12)

        section_label(p, "Scoring profile").pack(fill="x", pady=(0,8))

        profile_row = tk.Frame(p, bg=BG2)
        profile_row.pack(fill="x", pady=(0,16))
        for name, emoji, desc in [
            ("portrait",  "👤", "Single subject — face & emotion focused"),
            ("group",     "👥", "Multiple people — balanced weights"),
            ("landscape", "🌄", "No faces — sharpness & colour priority"),
        ]:
            self._profile_btn(profile_row, name, emoji, desc)

        section_label(p, "Folder paths").pack(fill="x", pady=(4,8))
        for lbl, var, tip in [
            ("Input folder",       self.input_dir, "Folder containing photos to evaluate"),
            ("Good photos output", self.good_dir,  "Where passing photos are copied"),
            ("Bad photos output",  self.bad_dir,   "Root for rejection sub-folders"),
        ]:
            self._path_row(p, lbl, var, tip)

        return tab

    def _profile_btn(self, parent, name, emoji, desc):
        def select():
            self.profile.set(name)
            for k, v in PROFILES[name].items():
                self.weights[k].set(v)
            for btn in parent._btns:
                btn.configure(
                    bg=BG3 if btn._name == name else BG2,
                    highlightbackground=ACCENT if btn._name == name else BORDER,
                )
        if not hasattr(parent, "_btns"):
            parent._btns = []
        f = tk.Frame(parent, bg=BG2, bd=0, highlightthickness=1,
                     highlightbackground=BORDER, cursor="hand2")
        f._name = name
        f.pack(side="left", expand=True, fill="x", padx=4)
        f.bind("<Button-1>", lambda e: select())
        label(f, emoji, bg=BG2, font=("Segoe UI", 16)).pack(pady=(10,0))
        label(f, name.capitalize(), bg=BG2, font=FONT_H2).pack()
        label(f, desc, bg=BG2, fg=TEXT2, font=FONT_TINY,
              wraplength=110).pack(padx=6, pady=(2,10))
        parent._btns.append(f)
        if name == self.profile.get():
            f.configure(bg=BG3, highlightbackground=ACCENT)

    def _path_row(self, parent, lbl_text, var, tip):
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill="x", pady=4)
        label(row, lbl_text, fg=TEXT2, font=FONT_TINY).pack(anchor="w")
        r2 = tk.Frame(row, bg=BG2)
        r2.pack(fill="x")
        e = styled_entry(r2, var)
        e.pack(side="left", fill="x", expand=True)
        def browse():
            d = filedialog.askdirectory(title=lbl_text)
            if d:
                var.set(d)
        tk.Button(r2, text="Browse", command=browse,
                  bg=BG3, fg=TEXT2, font=FONT_TINY, relief="flat",
                  cursor="hand2", padx=8).pack(side="left", padx=(6,0))
        label(row, tip, fg=TEXT2, font=FONT_TINY).pack(anchor="w")

    # ── Tab: Weights ──────────────────────────────────────────────────────────

    def _tab_weights(self, parent):
        tab = tk.Frame(parent, bg=BG2)
        inner = self._scroll_frame(tab)
        p = inner; p.configure(padx=16, pady=12)

        self.weight_total_lbl = label(p, "", fg=GOOD, font=FONT_H2)
        self.weight_total_lbl.pack(anchor="e", pady=(0,10))

        for k in WEIGHT_ORDER:
            self._weight_row(p, k)

        label(p, "Weights must sum to 100. Choosing a profile resets them.",
              fg=TEXT2, font=FONT_TINY, wraplength=380).pack(anchor="w", pady=(10,0))
        return tab

    def _weight_row(self, parent, key):
        color = WEIGHT_COLORS[key]
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill="x", pady=3)

        dot = tk.Canvas(row, width=10, height=10, bg=BG2, highlightthickness=0)
        dot.create_oval(1,1,9,9, fill=color, outline="")
        dot.pack(side="left", padx=(0,6))

        label(row, key.capitalize(), fg=TEXT2, width=12,
              anchor="w").pack(side="left")

        bar = tk.Scale(row, variable=self.weights[key],
                       from_=0, to=40, orient="horizontal",
                       bg=BG2, fg=TEXT, highlightthickness=0,
                       troughcolor=BG3, activebackground=color,
                       sliderrelief="flat", sliderlength=16,
                       length=200, showvalue=False)
        bar.pack(side="left", fill="x", expand=True)

        val_lbl = label(row, "0", fg=TEXT, font=FONT_MONO, width=3, anchor="e")
        val_lbl.pack(side="left", padx=(6,0))

        def update_label(*_):
            val_lbl.configure(text=str(self.weights[key].get()))
            self._update_weight_total()
        self.weights[key].trace_add("write", update_label)
        update_label()

    def _update_weight_total(self):
        total = sum(v.get() for v in self.weights.values())
        color = GOOD if total == 100 else BAD
        self.weight_total_lbl.configure(
            text=f"Total: {total} / 100", fg=color)

    # ── Tab: Thresholds ───────────────────────────────────────────────────────

    def _tab_thresholds(self, parent):
        tab = tk.Frame(parent, bg=BG2)
        inner = self._scroll_frame(tab)
        p = inner; p.configure(padx=16, pady=12)

        rows = [
            ("Min quality score", self.min_score, 0,   100, 1,  "/100"),
        ]
        section_label(p, "Quality gate").pack(fill="x", pady=(0,6))
        for args in rows:
            self._scale_row(p, *args)

        section_label(p, "Brightness").pack(fill="x", pady=(10,6))
        self.min_bright = tk.IntVar(value=50)
        self.max_bright = tk.IntVar(value=220)
        self._scale_row(p, "Min brightness", self.min_bright, 0, 255, 1, "")
        self._scale_row(p, "Max brightness", self.max_bright, 0, 255, 1, "")

        section_label(p, "Minimum resolution").pack(fill="x", pady=(10,6))
        self.min_w = tk.IntVar(value=640)
        self.min_h = tk.IntVar(value=480)
        self._scale_row(p, "Width (px)",  self.min_w, 320, 4000, 10, "px")
        self._scale_row(p, "Height (px)", self.min_h, 240, 3000, 10, "px")

        section_label(p, "Face count").pack(fill="x", pady=(10,6))
        self.t_min_faces = tk.IntVar(value=1)
        self.t_max_faces = tk.IntVar(value=10)
        self._scale_row(p, "Min faces", self.t_min_faces, 0,  50, 1, "")
        self._scale_row(p, "Max faces", self.t_max_faces, 1, 200, 1, "")

        section_label(p, "Duplicate detection").pack(fill="x", pady=(10,6))
        self.dup_thresh = tk.IntVar(value=8)
        self._scale_row(p, "Hash threshold", self.dup_thresh, 0, 20, 1, "")
        label(p, "Lower = stricter. 0 = exact duplicates only.",
              fg=TEXT2, font=FONT_TINY).pack(anchor="w", pady=(4,0))

        label(p, "Note: thresholds other than min-score require editing Config in photos.py.",
              fg=WARN, font=FONT_TINY, wraplength=380).pack(anchor="w", pady=(12,0))
        return tab

    def _scale_row(self, parent, lbl_text, var, lo, hi, step, unit):
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill="x", pady=3)
        label(row, lbl_text, fg=TEXT2, width=18, anchor="w").pack(side="left")
        bar = tk.Scale(row, variable=var, from_=lo, to=hi, resolution=step,
                       orient="horizontal", bg=BG2, fg=TEXT,
                       highlightthickness=0, troughcolor=BG3,
                       activebackground=ACCENT, sliderrelief="flat",
                       sliderlength=16, length=180, showvalue=False)
        bar.pack(side="left", fill="x", expand=True)
        val_lbl = label(row, f"{var.get()}{unit}", fg=TEXT,
                        font=FONT_MONO, width=7, anchor="e")
        val_lbl.pack(side="left", padx=(6,0))
        def upd(*_):
            val_lbl.configure(text=f"{var.get()}{unit}")
        var.trace_add("write", upd)

    # ── Tab: Processing ───────────────────────────────────────────────────────

    def _tab_processing(self, parent):
        tab = tk.Frame(parent, bg=BG2)
        inner = self._scroll_frame(tab)
        p = inner; p.configure(padx=16, pady=12)

        section_label(p, "Parallel workers").pack(fill="x", pady=(0,6))
        self._scale_row(p, "Workers", self.workers, 1, 16, 1, "")
        label(p, "DeepFace calls are serialised regardless — workers help with I/O.",
              fg=TEXT2, font=FONT_TINY).pack(anchor="w", pady=(4,0))

        section_label(p, "Compute device").pack(fill="x", pady=(14,6))
        dev_row = tk.Frame(p, bg=BG2)
        dev_row.pack(fill="x", pady=(0,14))
        for d in ("cpu", "cuda"):
            tk.Radiobutton(dev_row, text=d, variable=self.device, value=d,
                           bg=BG2, fg=TEXT, selectcolor=BG3,
                           activebackground=BG2, activeforeground=ACCENT,
                           font=FONT_MONO, cursor="hand2").pack(side="left", padx=8)

        section_label(p, "Run options").pack(fill="x", pady=(0,8))
        for var, lbl_text, tip in [
            (self.dry_run,  "Dry run",          "Analyse without copying any files"),
            (self.report,   "JSON report",       "Write report.json after processing"),
            (self.html_rep, "HTML gallery",      "Write an interactive report.html"),
            (self.no_cache, "Disable cache",     "Skip SQLite result caching"),
        ]:
            self._toggle_row(p, var, lbl_text, tip)

        return tab

    def _toggle_row(self, parent, var, lbl_text, tip):
        row = tk.Frame(parent, bg=BG2, pady=4)
        row.pack(fill="x")
        tk.Checkbutton(row, variable=var, text=lbl_text, onvalue=True, offvalue=False,
                       bg=BG2, fg=TEXT, selectcolor=BG3,
                       activebackground=BG2, activeforeground=ACCENT,
                       font=FONT_BODY, cursor="hand2").pack(side="left")
        label(row, f"— {tip}", fg=TEXT2, font=FONT_TINY).pack(side="left", padx=6)

    # ── Right panel: log output ───────────────────────────────────────────────

    def _build_right(self, parent):
        frame = tk.Frame(parent, bg=BG)

        # Status bar
        status_bar = tk.Frame(frame, bg=BG2, pady=6)
        status_bar.pack(fill="x", pady=(0,6))
        self.status_lbl = label(status_bar, "Ready", fg=TEXT2, bg=BG2)
        self.status_lbl.pack(side="left", padx=12)
        self.stop_btn = styled_button(status_bar, "Stop", self._stop_run,
                                      color=BAD, width=6)
        self.stop_btn.pack(side="right", padx=6)
        self.stop_btn.configure(state="disabled")

        self.run_btn = styled_button(status_bar, "Run", self._start_run,
                                     color=ACCENT, width=10)
        self.run_btn.pack(side="right", padx=6)

        # Summary cards
        cards = tk.Frame(frame, bg=BG)
        cards.pack(fill="x", pady=(0,6))
        self.card_total = self._stat_card(cards, "Total",  "—", TEXT)
        self.card_good  = self._stat_card(cards, "Good",   "—", GOOD)
        self.card_bad   = self._stat_card(cards, "Bad",    "—", BAD)
        self.card_dups  = self._stat_card(cards, "Dupes",  "—", WARN)

        # Log
        log_lbl = tk.Frame(frame, bg=BG)
        log_lbl.pack(fill="x")
        label(log_lbl, "OUTPUT LOG", fg=TEXT2, font=FONT_TINY, bg=BG).pack(side="left", pady=2)
        tk.Button(log_lbl, text="Clear", command=self._clear_log,
                  bg=BG, fg=TEXT2, font=FONT_TINY, relief="flat",
                  cursor="hand2").pack(side="right")

        self.log = scrolledtext.ScrolledText(
            frame, bg=BG2, fg=TEXT, font=FONT_MONO,
            relief="flat", state="disabled", wrap="none",
            insertbackground=TEXT,
        )
        self.log.pack(fill="both", expand=True)

        # Tag colours
        self.log.tag_config("good",  foreground=GOOD)
        self.log.tag_config("bad",   foreground=BAD)
        self.log.tag_config("warn",  foreground=WARN)
        self.log.tag_config("info",  foreground=TEXT2)
        self.log.tag_config("head",  foreground=ACCENT)

        return frame

    def _stat_card(self, parent, title, value, color):
        f = tk.Frame(parent, bg=BG2, padx=14, pady=8,
                     bd=0, highlightthickness=1, highlightbackground=BORDER)
        f.pack(side="left", expand=True, fill="x", padx=4)
        label(f, title, fg=TEXT2, font=FONT_TINY, bg=BG2).pack()
        lbl = label(f, value, fg=color, font=("Segoe UI Semibold", 18), bg=BG2)
        lbl.pack()
        return lbl

    # ── Command generation ────────────────────────────────────────────────────

    def _build_command(self):
        parts = [PYTHON, str(SCRIPT_PATH)]
        if self.input_dir.get() != "input_photos": parts += ["--input", self.input_dir.get()]
        if self.good_dir.get()  != "good_photos":  parts += ["--good",  self.good_dir.get()]
        if self.bad_dir.get()   != "bad_photos":   parts += ["--bad",   self.bad_dir.get()]
        if self.min_score.get() != 60:             parts += ["--min-score", str(self.min_score.get())]
        if self.workers.get()   != 4:              parts += ["--workers",   str(self.workers.get())]
        if self.profile.get()   != "portrait":     parts += ["--profile",   self.profile.get()]
        if self.device.get()    != "cpu":          parts += ["--device",    self.device.get()]
        if self.dry_run.get():   parts.append("--dry-run")
        if self.report.get():    parts.append("--report")
        if self.html_rep.get():  parts.append("--html-report")
        if self.no_cache.get():  parts.append("--no-cache")
        return parts

    def _refresh_command(self):
        cmd = self._build_command()
        display = " \\\n  ".join(cmd)
        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("end", display)
        self.cmd_text.configure(state="disabled")

    def _copy_command(self):
        cmd = " ".join(self._build_command())
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self._log("Command copied to clipboard.\n", "info")

    # ── Run / Stop ────────────────────────────────────────────────────────────

    def _start_run(self):
        if not SCRIPT_PATH.exists():
            messagebox.showerror("Not found",
                f"photos.py not found at:\n{SCRIPT_PATH}\n\n"
                "Put photos.py in the same folder as this GUI.")
            return

        input_path = Path(self.input_dir.get())
        if not input_path.exists():
            messagebox.showerror("Input missing",
                f"Input folder not found:\n{input_path}")
            return

        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_lbl.configure(text="Running…", fg=ACCENT)
        for c in [self.card_total, self.card_good, self.card_bad, self.card_dups]:
            c.configure(text="…")
        self._log("─" * 60 + "\n", "head")
        self._log(f"Starting: {' '.join(self._build_command())}\n", "head")

        cmd = self._build_command()
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=str(SCRIPT_PATH.parent),
        )
        threading.Thread(target=self._stream_output, daemon=True).start()

    def _stop_run(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log("\nProcess terminated by user.\n", "warn")
        self._run_finished()

    def _stream_output(self):
        summary = {"total": None, "good": None, "bad": None, "dups": None}
        for line in self._process.stdout:
            self.root.after(0, self._append_line, line, summary)
        self._process.wait()
        rc = self._process.returncode
        self.root.after(0, self._run_finished, rc, summary)

    def _append_line(self, line, summary):
        tag = "info"
        lo = line.lower()
        if "good" in lo and "score" in lo:  tag = "good"
        elif " bad" in lo and "score" in lo: tag = "bad"
        elif "error" in lo or "traceback" in lo: tag = "bad"
        elif "warning" in lo or "warn" in lo:    tag = "warn"
        elif "===" in line or "---" in line:     tag = "head"

        # Parse summary block
        if "Total processed" in line:
            try: summary["total"] = line.split(":")[1].strip()
            except: pass
        if "Good photos" in line:
            try: summary["good"] = line.split(":")[1].strip().split()[0]
            except: pass
        if "Bad  photos" in line:
            try:
                summary["bad"]  = line.split(":")[1].strip().split()[0]
                summary["dups"] = line.split("removed:")[1].strip().rstrip(")")
            except: pass

        self._log(line, tag)

    def _log(self, text, tag="info"):
        self.log.configure(state="normal")
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _run_finished(self, returncode=None, summary=None):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        if returncode == 0:
            self.status_lbl.configure(text="Finished ✓", fg=GOOD)
            self._log("Done.\n", "good")
        elif returncode is None:
            self.status_lbl.configure(text="Stopped", fg=WARN)
        else:
            self.status_lbl.configure(text=f"Exited ({returncode})", fg=BAD)
            self._log(f"Process exited with code {returncode}\n", "bad")

        if summary:
            if summary["total"]: self.card_total.configure(text=summary["total"])
            if summary["good"]:  self.card_good.configure(text=summary["good"])
            if summary["bad"]:   self.card_bad.configure(text=summary["bad"])
            if summary["dups"]:  self.card_dups.configure(text=summary["dups"])

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoClassifierGUI(root)
    root.mainloop()
