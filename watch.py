#!/usr/bin/env python3
"""Manage and open MAX30009 recording CSV pairs.

The manager scans the directory containing this script by default.  It finds
the calibrated ``.csv`` file that belongs to each ``.bioz.csv`` export,
optionally waits until both files have stopped changing, and lists complete
recordings in a small desktop window.  The directory is scanned at startup and again
only when the user clicks ``Refresh Now``.  Selecting a recording launches the
existing ``plots.py`` program with both CSV files from that recording.

Annotations are stored in ``max30009_recording_annotations.json`` beside the
script, so they remain available the next time the manager is opened.  The
program uses only Python's standard library; Tkinter is used for the desktop
window when it is available.

Run from the directory containing this file and the plotting program.  The
directory is scanned once at startup; click ``Refresh Now`` when you want to
look for newly exported files:

    python watch_max30009.py

If Tkinter is unavailable, a text menu is used instead:

    python watch_max30009.py --console
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


BIOZ_FILENAME_RE = re.compile(
    r"(?P<date>\d{8}|\d{4})[_-](?P<time>\d{6})\.bioz\.csv$",
    re.IGNORECASE,
)
ANNOTATIONS_FILENAME = "max30009_recording_annotations.json"


class AnnotationFileError(ValueError):
    """The annotation sidecar exists but does not have a valid format."""


@dataclass(frozen=True)
class RecordingPair:
    """The two files needed to open one MAX30009 recording."""

    display_identifier: str
    normalized_identifier: str
    bioz_path: Path
    calibrated_path: Path

    @property
    def key(self) -> str:
        """Return a stable key for the annotation sidecar."""

        return "||".join(
            (
                self.display_identifier,
                self.bioz_path.name,
                self.calibrated_path.name,
            )
        )


@dataclass
class StabilityState:
    signature: Optional[tuple[int, int]] = None
    unchanged_since: Optional[float] = None


def _recording_match(path: Path) -> Optional[re.Match[str]]:
    return BIOZ_FILENAME_RE.search(path.name)


def recording_identifier(path: Path) -> Optional[str]:
    """Return the normalized MMDD_HHMMSS token from a .bioz.csv filename."""

    match = _recording_match(path)
    if match is None:
        return None
    date_token = match.group("date")
    if len(date_token) == 8:
        date_token = date_token[-4:]
    return f"{date_token}_{match.group('time')}"


def display_recording_identifier(path: Path) -> Optional[str]:
    """Return the filename's YYYYMMDD_HHMMSS or MMDD_HHMMSS identifier."""

    match = _recording_match(path)
    if match is None:
        return None
    return f"{match.group('date')}_{match.group('time')}"


def is_bioz_csv(path: Path) -> bool:
    return path.is_file() and _recording_match(path) is not None


def list_bioz_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if is_bioz_csv(path)
    )


def find_companion_csv(
    directory: Path,
    bioz_path: Path,
    identifier: str,
) -> tuple[Optional[Path], str]:
    """Find the non-.bioz CSV containing the same MMDD_HHMMSS token."""

    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path != bioz_path
        and path.suffix.lower() == ".csv"
        and not path.name.lower().endswith(".bioz.csv")
        and identifier in path.name
    ]
    if len(candidates) == 1:
        return candidates[0], ""

    # Prefer the evaluation software's calibrated-result naming convention if
    # other CSVs with the same identifier are present.
    preferred = [
        path
        for path in candidates
        if "bioz-vs-time" in path.name.lower()
        or "cal-results" in path.name.lower()
    ]
    if len(preferred) == 1:
        return preferred[0], ""
    if not candidates:
        return None, f"waiting for companion CSV containing {identifier}"
    names = ", ".join(path.name for path in candidates)
    return None, f"waiting because companion CSV is ambiguous: {names}"


def file_is_stable(
    path: Path,
    states: dict[Path, StabilityState],
    *,
    stable_seconds: float,
) -> bool:
    """Return True after a file's size and mtime remain unchanged long enough."""

    try:
        stat = path.stat()
    except FileNotFoundError:
        states.pop(path, None)
        return False

    signature = (stat.st_size, stat.st_mtime_ns)
    now = time.monotonic()
    state = states.setdefault(path, StabilityState())
    if state.signature != signature:
        state.signature = signature
        state.unchanged_since = now
        return stable_seconds <= 0
    if state.unchanged_since is None:
        state.unchanged_since = now
        return stable_seconds <= 0
    return now - state.unchanged_since >= stable_seconds


def discover_recording_pairs(
    directory: Path,
    stability_states: dict[Path, StabilityState],
    *,
    stable_seconds: float,
) -> tuple[list[RecordingPair], list[str]]:
    """Return ready recording pairs and short descriptions of pending files."""

    ready: list[RecordingPair] = []
    pending: list[str] = []

    for bioz_path in list_bioz_files(directory):
        normalized_identifier = recording_identifier(bioz_path)
        display_identifier = display_recording_identifier(bioz_path)
        if normalized_identifier is None or display_identifier is None:
            continue

        companion_path, status = find_companion_csv(
            directory,
            bioz_path,
            normalized_identifier,
        )
        if companion_path is None:
            pending.append(f"{bioz_path.name}: {status}")
            continue

        bioz_stable = file_is_stable(
            bioz_path,
            stability_states,
            stable_seconds=stable_seconds,
        )
        companion_stable = file_is_stable(
            companion_path,
            stability_states,
            stable_seconds=stable_seconds,
        )
        if not bioz_stable or not companion_stable:
            pending.append(
                f"{bioz_path.name}: waiting for both files to finish writing"
            )
            continue

        ready.append(
            RecordingPair(
                display_identifier=display_identifier,
                normalized_identifier=normalized_identifier,
                bioz_path=bioz_path,
                calibrated_path=companion_path,
            )
        )

    return ready, pending


def load_annotations(path: Path) -> dict[str, str]:
    """Load and validate the annotation sidecar.

    The current format is::

        {"version": 1, "recordings": {"pair-key": {"annotation": "..."}}}

    A simple mapping from pair keys to strings is accepted for compatibility
    with early development versions.  Other malformed content is reported to
    the caller so the GUI can ask whether to exit or continue.
    """

    if not path.exists():
        return {}
    if not path.is_file():
        raise AnnotationFileError(f"the annotation path is not a regular file: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationFileError(
            f"Could not read the annotation file {path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise AnnotationFileError("the top-level value must be a JSON object")

    is_legacy = "recordings" not in payload and "version" not in payload
    if is_legacy:
        recordings = payload
    else:
        if payload.get("version") != 1:
            raise AnnotationFileError("the annotation file must have version 1")
        recordings = payload.get("recordings")

    if not isinstance(recordings, dict):
        raise AnnotationFileError("the 'recordings' value must be a JSON object")

    annotations: dict[str, str] = {}
    for key, value in recordings.items():
        if not isinstance(key, str):
            raise AnnotationFileError("recording keys must be strings")
        if is_legacy:
            if not isinstance(value, str):
                raise AnnotationFileError(
                    f"legacy annotation for {key!r} must be a string"
                )
            annotations[key] = value
            continue
        if not isinstance(value, dict) or not isinstance(value.get("annotation"), str):
            raise AnnotationFileError(
                f"recordings[{key!r}] must contain a string 'annotation' value"
            )
        annotations[key] = value["annotation"]
    return annotations


def save_annotations(path: Path, annotations: dict[str, str]) -> None:
    """Atomically save annotations beside the measurement files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    payload = {
        "version": 1,
        "recordings": {
            key: {"annotation": annotations[key]}
            for key in sorted(annotations)
        },
    }

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def launch_viewer(
    pair: RecordingPair,
    *,
    viewer: Path,
    output_directory: Path,
    working_directory: Path,
) -> None:
    """Launch the existing plotting program for a selected recording."""

    command = [
        sys.executable,
        str(viewer),
        str(pair.bioz_path),
        str(pair.calibrated_path),
        "--output-dir",
        str(output_directory),
    ]
    subprocess.Popen(command, cwd=str(working_directory))


def build_argument_parser() -> argparse.ArgumentParser:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Catalogue MAX30009 CSV pairs and open the plotting program; "
            "refreshes are manual."
        )
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=script_directory,
        help="directory to scan; defaults to the directory containing this script",
    )
    parser.add_argument(
        "--viewer",
        type=Path,
        default=script_directory / "plots.py",
        help="plotting script to launch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for viewer output; defaults to the scanned directory",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "annotation JSON file; defaults to "
            f"{ANNOTATIONS_FILENAME} in the scanned directory"
        ),
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=0.0,
        help=(
            "optional safety delay requiring both CSVs to remain unchanged "
            "before they are listed"
        ),
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="use the text menu instead of the Tkinter desktop window",
    )
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


class RecordingManagerGUI:
    """Tkinter interface for the recording catalogue."""

    def __init__(
        self,
        root: Any,
        tk_module: Any,
        ttk_module: Any,
        messagebox_module: Any,
        *,
        directory: Path,
        viewer: Path,
        output_directory: Path,
        annotations_path: Path,
        annotations: dict[str, str],
        stable_seconds: float,
    ) -> None:
        self.root = root
        self.tk = tk_module
        self.ttk = ttk_module
        self.messagebox = messagebox_module
        self.directory = directory
        self.viewer = viewer
        self.output_directory = output_directory
        self.annotations_path = annotations_path
        self.stable_seconds = stable_seconds

        self.annotations = dict(annotations)
        self.stability_states: dict[Path, StabilityState] = {}
        self.pairs: list[RecordingPair] = []
        self.pairs_by_key: dict[str, RecordingPair] = {}
        self.item_to_key: dict[str, str] = {}
        self.selected_key: Optional[str] = None
        self.annotation_editor_visible = False

        self.root.title("MAX30009 Recording Manager")
        self.root.geometry("950x520")
        self.root.minsize(720, 420)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh()

    def _build_widgets(self) -> None:
        main = self.ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        heading = self.ttk.Frame(main)
        heading.grid(row=0, column=0, sticky="ew")
        heading.columnconfigure(0, weight=1)
        self.ttk.Label(
            heading,
            text="MAX30009 recordings",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Button(
            heading,
            text="Refresh Now",
            command=self.refresh,
        ).grid(row=0, column=1, sticky="e")

        self.ttk.Label(
            main,
            text=(
                f"Folder: {self.directory}\n"
                "Only complete, stable .bioz.csv + calibrated .csv pairs are listed."
            ),
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 8))

        table_frame = self.ttk.Frame(main)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.tree = self.ttk.Treeview(
            table_frame,
            columns=("recording", "annotation"),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("recording", text="Recording")
        self.tree.heading("annotation", text="Annotation")
        self.tree.column("recording", width=185, anchor="w", stretch=False)
        self.tree.column("annotation", width=650, anchor="w", stretch=True)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = self.ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_selection_changed)
        self.tree.bind("<Return>", self._open_selected_event)
        self.tree.bind("<KP_Enter>", self._open_selected_event)

        self.action_frame = self.ttk.LabelFrame(
            main,
            text="Selected recording",
            padding=8,
        )
        self.action_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.action_frame.columnconfigure(0, weight=1)
        self.action_frame.columnconfigure(1, weight=1)

        self.display_button = self.ttk.Button(
            self.action_frame,
            text="Display Plot",
            command=self._display_plot,
        )
        self.display_button.grid(row=0, column=0, sticky="w")
        self.annotate_button = self.ttk.Button(
            self.action_frame,
            text="Annotate",
            command=self._show_annotation_editor,
        )
        self.annotate_button.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.annotation_display_var = self.tk.StringVar(value="Annotation: none")
        self.ttk.Label(
            self.action_frame,
            textvariable=self.annotation_display_var,
            justify="left",
            wraplength=850,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.annotation_editor = self.ttk.Frame(self.action_frame)
        self.annotation_editor.columnconfigure(1, weight=1)
        self.annotation_editor.rowconfigure(0, weight=1)
        self.annotation_editor.rowconfigure(1, weight=1)
        self.ttk.Label(self.annotation_editor, text="Description:").grid(
            row=0,
            column=0,
            rowspan=2,
            sticky="nw",
            padx=(0, 8),
        )
        self.annotation_text = self.tk.Text(
            self.annotation_editor,
            height=3,
            width=60,
            wrap="word",
        )
        self.annotation_text.grid(row=0, column=1, rowspan=2, sticky="nsew")
        self.ttk.Button(
            self.annotation_editor,
            text="Copy",
            command=self._copy_annotation,
        ).grid(row=0, column=2, sticky="n", padx=(8, 0))
        self.ttk.Button(
            self.annotation_editor,
            text="Paste",
            command=self._paste_annotation,
        ).grid(row=1, column=2, sticky="n", padx=(8, 0))
        self.ttk.Button(
            self.annotation_editor,
            text="Save Annotation",
            command=self._save_selected_annotation,
        ).grid(row=0, column=3, sticky="n", padx=(8, 0))
        self.ttk.Button(
            self.annotation_editor,
            text="Cancel",
            command=self._cancel_annotation,
        ).grid(row=1, column=3, sticky="n", padx=(8, 0))

        self.annotation_menu = self.tk.Menu(self.annotation_text, tearoff=False)
        self.annotation_menu.add_command(label="Cut", command=self._cut_annotation)
        self.annotation_menu.add_command(label="Copy", command=self._copy_annotation)
        self.annotation_menu.add_command(label="Paste", command=self._paste_annotation)
        self.annotation_menu.add_separator()
        self.annotation_menu.add_command(
            label="Select All",
            command=self._select_all_annotation,
        )
        self.annotation_text.bind("<Button-3>", self._show_annotation_context_menu)
        self.annotation_text.bind("<Control-c>", self._copy_annotation)
        self.annotation_text.bind("<Control-v>", self._paste_annotation)
        self.annotation_text.bind("<Control-x>", self._cut_annotation)
        self.annotation_text.bind("<Control-a>", self._select_all_annotation)
        self.annotation_text.bind("<Command-c>", self._copy_annotation)
        self.annotation_text.bind("<Command-v>", self._paste_annotation)
        self.annotation_text.bind("<Command-x>", self._cut_annotation)
        self.annotation_text.bind("<Command-a>", self._select_all_annotation)

        # Keep the selected-recording controls hidden until a row is selected.
        self.action_frame.grid_remove()

        self.details_var = self.tk.StringVar(value="No recording selected.")
        self.ttk.Label(
            main,
            textvariable=self.details_var,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.status_var = self.tk.StringVar(value="Scanning...")
        self.ttk.Label(
            main,
            textvariable=self.status_var,
            justify="left",
        ).grid(row=5, column=0, sticky="w", pady=(4, 0))

    def _selected_pair(self) -> Optional[RecordingPair]:
        if self.selected_key is None:
            return None
        return self.pairs_by_key.get(self.selected_key)

    def _hide_selected_controls(self) -> None:
        self.action_frame.grid_remove()
        self._hide_annotation_editor()

    def _hide_annotation_editor(self) -> None:
        if self.annotation_editor_visible:
            self.annotation_editor.grid_remove()
            self.annotation_editor_visible = False

    def _set_annotation_editor_text(self, annotation: str) -> None:
        self.annotation_text.delete("1.0", "end")
        self.annotation_text.insert("1.0", annotation)

    def _copy_annotation(self, _event: Any = None) -> str:
        try:
            selected = self.annotation_text.get("sel.first", "sel.last")
        except self.tk.TclError:
            return "break"
        self.root.clipboard_clear()
        self.root.clipboard_append(selected)
        return "break"

    def _cut_annotation(self, _event: Any = None) -> str:
        if self._copy_annotation() == "break":
            try:
                self.annotation_text.delete("sel.first", "sel.last")
            except self.tk.TclError:
                pass
        return "break"

    def _paste_annotation(self, _event: Any = None) -> str:
        try:
            pasted = self.root.clipboard_get()
        except self.tk.TclError:
            return "break"
        try:
            self.annotation_text.delete("sel.first", "sel.last")
        except self.tk.TclError:
            pass
        self.annotation_text.insert("insert", pasted)
        return "break"

    def _select_all_annotation(self, _event: Any = None) -> str:
        self.annotation_text.focus_set()
        self.annotation_text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _show_annotation_context_menu(self, event: Any) -> str:
        self.annotation_text.focus_set()
        try:
            self.annotation_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.annotation_menu.grab_release()
        return "break"

    def _on_selection_changed(self, _event: Any = None) -> None:
        selection = self.tree.selection()
        if not selection:
            self.selected_key = None
            self._hide_selected_controls()
            self.details_var.set("No recording selected.")
            return

        key = self.item_to_key.get(selection[0])
        pair = self.pairs_by_key.get(key) if key is not None else None
        if pair is None:
            return
        self.selected_key = pair.key
        self.action_frame.grid()
        self._hide_annotation_editor()
        annotation = self.annotations.get(pair.key, "")
        self.annotation_display_var.set(
            f"Annotation: {annotation}" if annotation else "Annotation: none"
        )
        self.details_var.set(
            f"{pair.display_identifier}\n"
            f"BioZ: {pair.bioz_path.name}\n"
            f"Calibrated results: {pair.calibrated_path.name}"
        )

    def _open_selected_event(self, _event: Any = None) -> str:
        self._display_plot()
        return "break"

    def _show_annotation_editor(self) -> None:
        pair = self._selected_pair()
        if pair is None:
            return
        self._set_annotation_editor_text(self.annotations.get(pair.key, ""))
        self.annotation_editor.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )
        self.annotation_editor_visible = True
        self.annotation_text.focus_set()

    def _cancel_annotation(self) -> None:
        self._hide_annotation_editor()

    def _save_selected_annotation(self) -> bool:
        pair = self._selected_pair()
        if pair is None:
            return False

        annotation_text = self.annotation_text.get("1.0", "end-1c")
        if annotation_text.strip():
            self.annotations[pair.key] = annotation_text
        else:
            self.annotations.pop(pair.key, None)
        try:
            save_annotations(self.annotations_path, self.annotations)
        except OSError as exc:
            self.messagebox.showerror(
                "Could not save annotation",
                f"The annotation file could not be saved:\n{exc}",
            )
            return False

        for item_id, key in self.item_to_key.items():
            if key == pair.key:
                self.tree.item(
                    item_id,
                    values=(pair.display_identifier, annotation_text),
                )
                break
        self.annotation_display_var.set(
            f"Annotation: {annotation_text}" if annotation_text else "Annotation: none"
        )
        self._hide_annotation_editor()
        self.status_var.set(f"Annotation saved for {pair.display_identifier}.")
        return True

    def _display_plot(self) -> None:
        pair = self._selected_pair()
        if pair is None:
            return
        try:
            launch_viewer(
                pair,
                viewer=self.viewer,
                output_directory=self.output_directory,
                working_directory=self.directory,
            )
        except OSError as exc:
            self.messagebox.showerror(
                "Could not open viewer",
                f"The plotting program could not be launched:\n{exc}",
            )
            return
        self.status_var.set(f"Opened {pair.display_identifier} in the plotting program.")

    def refresh(self) -> None:
        previous_key = self.selected_key
        editor_was_visible = self.annotation_editor_visible
        editor_draft = (
            self.annotation_text.get("1.0", "end-1c")
            if editor_was_visible
            else ""
        )
        try:
            pairs, pending = discover_recording_pairs(
                self.directory,
                self.stability_states,
                stable_seconds=self.stable_seconds,
            )
        except OSError as exc:
            self.status_var.set(f"Could not scan folder: {exc}")
            return

        pairs.sort(
            key=lambda pair: (pair.display_identifier, pair.bioz_path.name),
            reverse=True,
        )
        self.pairs = pairs
        self.pairs_by_key = {pair.key: pair for pair in pairs}
        self.item_to_key.clear()
        self.tree.delete(*self.tree.get_children())

        selected_item: Optional[str] = None
        for index, pair in enumerate(pairs):
            item_id = f"pair_{index}"
            self.item_to_key[item_id] = pair.key
            self.tree.insert(
                "",
                "end",
                iid=item_id,
                values=(pair.display_identifier, self.annotations.get(pair.key, "")),
            )
            if pair.key == previous_key:
                selected_item = item_id

        if selected_item is not None:
            self.tree.selection_set(selected_item)
            self.tree.focus(selected_item)
            self._on_selection_changed()
            if editor_was_visible:
                self._show_annotation_editor()
                self._set_annotation_editor_text(editor_draft)
        elif not pairs:
            self.selected_key = None
            self._hide_selected_controls()
            if pending:
                self.details_var.set(pending[0])
            else:
                self.details_var.set("No complete recording pairs found yet.")
        elif previous_key not in self.pairs_by_key:
            self.selected_key = None
            self._hide_selected_controls()
            self.details_var.set("Select a recording to see its files.")

        status = f"{len(pairs)} complete recording pair(s)"
        if pending:
            status += f" | {len(pending)} pending/incomplete"
        self.status_var.set(status)

    def _on_close(self) -> None:
        self.root.destroy()


def run_console(
    *,
    directory: Path,
    viewer: Path,
    output_directory: Path,
    annotations_path: Path,
    stable_seconds: float,
) -> int:
    """Run a small text-menu fallback when a desktop display is unavailable."""

    try:
        annotations = load_annotations(annotations_path)
    except AnnotationFileError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        try:
            answer = input(
                "Proceed without the existing annotations and overwrite them on save? [y/N] "
            )
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if answer.strip().lower() not in {"y", "yes"}:
            print("Exiting so the annotation file can be fixed.")
            return 0
        annotations = {}
    stability_states: dict[Path, StabilityState] = {}

    while True:
        try:
            pairs, pending = discover_recording_pairs(
                directory,
                stability_states,
                stable_seconds=stable_seconds,
            )
        except OSError as exc:
            print(f"Could not scan folder: {exc}", file=sys.stderr)
            return 1

        pairs.sort(
            key=lambda pair: (pair.display_identifier, pair.bioz_path.name),
            reverse=True,
        )
        print("\nMAX30009 recordings")
        if pairs:
            for index, pair in enumerate(pairs, start=1):
                annotation = annotations.get(pair.key, "")
                suffix = f" — {annotation}" if annotation else ""
                print(f"  {index}. {pair.display_identifier}{suffix}")
        else:
            print("  No complete, stable recording pairs found yet.")
        if pending:
            print(f"  ({len(pending)} file pair(s) pending or incomplete.)")
        print("Commands: o NUMBER = open | a NUMBER = annotate | r = refresh | q = quit")

        try:
            command = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if command.lower() in {"q", "quit", "exit"}:
            return 0
        if command.lower() in {"r", "refresh", ""}:
            continue

        parts = command.split(maxsplit=1)
        if len(parts) != 2 or parts[0].lower() not in {"o", "a"}:
            print("Unrecognized command.")
            continue
        try:
            selected_index = int(parts[1]) - 1
            pair = pairs[selected_index]
        except (ValueError, IndexError):
            print("Please enter a valid recording number.")
            continue

        if parts[0].lower() == "a":
            try:
                annotation = input("Annotation (blank removes it): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if annotation:
                annotations[pair.key] = annotation
            else:
                annotations.pop(pair.key, None)
            try:
                save_annotations(annotations_path, annotations)
            except OSError as exc:
                print(f"Could not save annotation: {exc}", file=sys.stderr)
                continue
            print("Annotation saved.")
            continue

        try:
            launch_viewer(
                pair,
                viewer=viewer,
                output_directory=output_directory,
                working_directory=directory,
            )
        except OSError as exc:
            print(f"Could not launch viewer: {exc}", file=sys.stderr)
        else:
            print(f"Opened {pair.display_identifier}.")


def run_gui(
    *,
    directory: Path,
    viewer: Path,
    output_directory: Path,
    annotations_path: Path,
    stable_seconds: float,
) -> int:
    """Start the Tkinter manager, or fall back to the text menu."""

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        print(f"Tkinter is unavailable ({exc}); using the console menu.")
        return run_console(
            directory=directory,
            viewer=viewer,
            output_directory=output_directory,
            annotations_path=annotations_path,
            stable_seconds=stable_seconds,
        )

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"A desktop window could not be opened ({exc}); using the console menu.")
        return run_console(
            directory=directory,
            viewer=viewer,
            output_directory=output_directory,
            annotations_path=annotations_path,
            stable_seconds=stable_seconds,
        )

    try:
        annotations = load_annotations(annotations_path)
    except AnnotationFileError as exc:
        proceed = messagebox.askyesno(
            "Invalid annotation file",
            f"The annotation file could not be validated:\n\n{exc}\n\n"
            "Choose Yes to continue without its annotations. Any annotation "
            "saved during this session will overwrite the invalid file.\n\n"
            "Choose No to exit and fix the file.",
            icon="warning",
            default="no",
        )
        if not proceed:
            root.destroy()
            return 0
        annotations = {}

    RecordingManagerGUI(
        root,
        tk,
        ttk,
        messagebox,
        directory=directory,
        viewer=viewer,
        output_directory=output_directory,
        annotations_path=annotations_path,
        annotations=annotations,
        stable_seconds=stable_seconds,
    )
    root.mainloop()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    directory = args.directory.expanduser().resolve()
    viewer = args.viewer.expanduser().resolve()
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else directory
    )
    annotations_path = (
        args.annotations.expanduser().resolve()
        if args.annotations is not None
        else directory / ANNOTATIONS_FILENAME
    )

    if not directory.is_dir():
        print(f"ERROR: scan directory does not exist: {directory}", file=sys.stderr)
        return 2
    if not viewer.is_file():
        print(f"ERROR: viewer script does not exist: {viewer}", file=sys.stderr)
        return 2
    if args.stable_seconds < 0:
        print("ERROR: stable time cannot be negative", file=sys.stderr)
        return 2

    runner = run_console if args.console else run_gui
    return runner(
        directory=directory,
        viewer=viewer,
        output_directory=output_directory,
        annotations_path=annotations_path,
        stable_seconds=args.stable_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
