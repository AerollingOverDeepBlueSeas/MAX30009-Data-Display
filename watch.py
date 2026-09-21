#!/usr/bin/env python3
"""Manage and open MAX30009 recording CSV pairs.

The manager scans the archive root containing this script by default and
recursively finds calibrated ``.csv`` files that belong to each ``.bioz.csv``
export.  Each pair must remain in the same session folder.  It optionally waits
until both files have stopped changing, and lists complete recordings in a
scrollable desktop window.  The GUI supports a flat archive-wide list as well
as a navigable ``Folder View`` and can filter recordings by partial filename,
relative folder name, or annotation text.  The archive is scanned at startup
and again only when the user clicks ``Refresh Now``.  Selecting a recording
launches the existing ``plots.py`` program with both CSV files from that
recording; by default, generated output is written beside those source files.

Annotations are stored in ``max30009_recording_annotations.json`` beside the
script, so they remain available the next time the manager is opened.  The
program uses only Python's standard library; Tkinter is used for the desktop
window when it is available.

Run from the archive root containing this file and the plotting program.  The
root and all of its subfolders are scanned once at startup; click ``Refresh
Now`` when you want to look for newly exported files or folders:

    python watch_max30009.py

If Tkinter is unavailable, a text menu is used instead:

    python watch_max30009.py --console
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional


BIOZ_FILENAME_RE = re.compile(
    r"(?P<date>\d{8}|\d{4})[_-](?P<time>\d{6})\.bioz\.csv$",
    re.IGNORECASE,
)
RECORDING_TOKEN_RE = re.compile(
    r"(?P<date>\d{8}|\d{4})[_-](?P<time>\d{6})(?!\d)",
    re.IGNORECASE,
)
ANNOTATIONS_FILENAME = "max30009_recording_annotations.json"


class AnnotationFileError(ValueError):
    """The annotation sidecar exists but does not have a valid format."""


@dataclass(frozen=True)
class RecordingPair:
    """The two files needed to open one MAX30009 recording."""

    recording_identifier: str
    display_identifier: str
    normalized_identifier: str
    bioz_path: Path
    calibrated_path: Path
    relative_bioz_path: str
    relative_calibrated_path: str

    @property
    def key(self) -> str:
        """Return a stable archive-relative key for the annotation sidecar."""

        return "||".join(
            (
                self.recording_identifier,
                self.relative_bioz_path,
                self.relative_calibrated_path,
            )
        )

    @property
    def legacy_key(self) -> str:
        """Return the pre-recursive annotation key for compatibility."""

        return "||".join(
            (
                self.recording_identifier,
                self.bioz_path.name,
                self.calibrated_path.name,
            )
        )


@dataclass(frozen=True)
class CatalogueEntry:
    """One selectable file, folder, or parent entry in the catalogue."""

    kind: str
    entry_id: str
    label: str
    pair: Optional[RecordingPair] = None
    folder_path: Optional[PurePosixPath] = None


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


def filename_recording_identifier(path: Path) -> Optional[str]:
    """Return a normalized recording token from any CSV filename."""

    match = RECORDING_TOKEN_RE.search(path.name)
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
        for path in directory.rglob("*.bioz.csv")
        if is_bioz_csv(path)
    )


def recording_display_name(
    archive_root: Path,
    bioz_path: Path,
    recording_identifier_text: str,
) -> str:
    """Combine the relative folder path and recording identifier for display."""

    relative_folder = bioz_path.parent.relative_to(archive_root)
    if not relative_folder.parts:
        return recording_identifier_text
    return " > ".join((*relative_folder.parts, recording_identifier_text))


def relative_path_string(archive_root: Path, path: Path) -> str:
    """Return a stable, platform-independent archive-relative path."""

    return path.relative_to(archive_root).as_posix()


def annotation_for_pair(
    annotations: dict[str, str],
    pair: RecordingPair,
) -> str:
    """Read a current annotation, with compatibility for old sidecar keys."""

    return annotations.get(pair.key, annotations.get(pair.legacy_key, ""))


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
        and (
            identifier in path.name
            or filename_recording_identifier(path) == identifier
        )
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
    """Recursively return ready pairs and descriptions of pending files.

    Pairing is deliberately restricted to the folder containing each BioZ
    file, preventing similarly named recordings in separate sessions from
    being matched together.
    """

    ready: list[RecordingPair] = []
    pending: list[str] = []

    for bioz_path in list_bioz_files(directory):
        normalized_identifier = recording_identifier(bioz_path)
        display_identifier = display_recording_identifier(bioz_path)
        if normalized_identifier is None or display_identifier is None:
            continue

        companion_path, status = find_companion_csv(
            bioz_path.parent,
            bioz_path,
            normalized_identifier,
        )
        if companion_path is None:
            pending.append(
                f"{relative_path_string(directory, bioz_path)}: {status}"
            )
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
                f"{relative_path_string(directory, bioz_path)}: waiting for both "
                "files to finish writing"
            )
            continue

        ready.append(
            RecordingPair(
                recording_identifier=display_identifier,
                display_identifier=recording_display_name(
                    directory,
                    bioz_path,
                    display_identifier,
                ),
                normalized_identifier=normalized_identifier,
                bioz_path=bioz_path,
                calibrated_path=companion_path,
                relative_bioz_path=relative_path_string(directory, bioz_path),
                relative_calibrated_path=relative_path_string(
                    directory,
                    companion_path,
                ),
            )
        )

    return ready, pending


def load_annotations(path: Path) -> dict[str, str]:
    """Load and validate the annotation sidecar.

    The current format is::

        {"version": 2, "recordings": {"archive-relative-pair-key": {
            "annotation": "..."
        }}}

    Version 1 and a simple mapping from pair keys to strings are accepted for
    compatibility with earlier versions.  Other malformed content is reported
    to the caller so the GUI can ask whether to exit or continue.
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
        if payload.get("version") not in {1, 2}:
            raise AnnotationFileError("the annotation file must have version 1 or 2")
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
        "version": 2,
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
    output_directory: Optional[Path],
    working_directory: Path,
) -> None:
    """Launch the existing plotting program for a selected recording."""

    selected_output_directory = output_directory or pair.bioz_path.parent
    command = [
        sys.executable,
        str(viewer),
        str(pair.bioz_path),
        str(pair.calibrated_path),
        "--output-dir",
        str(selected_output_directory),
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
        help=(
            "archive root to scan recursively; defaults to the directory "
            "containing this script"
        ),
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
        help=(
            "optional common directory for viewer output; by default, generated "
            "files are stored beside the selected recording"
        ),
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
        output_directory: Optional[Path],
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
        self.entry_tag_to_entry: dict[str, CatalogueEntry] = {}
        self.entries: list[CatalogueEntry] = []
        self.folder_path = PurePosixPath()
        self.selected_entry: Optional[CatalogueEntry] = None
        self.selected_entry_id: Optional[str] = None
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
        main.rowconfigure(3, weight=1)

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
                f"Archive root: {self.directory}\n"
                "Only complete, stable .bioz.csv + calibrated .csv pairs are listed. "
                "Names include relative session folders and wrap at 100 characters. "
                "Use Search and Folder View to narrow the catalogue."
            ),
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 8))

        search_frame = self.ttk.Frame(main)
        search_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        search_frame.columnconfigure(1, weight=1)
        self.ttk.Label(search_frame, text="Search name/annotation:").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
        )
        self.search_var = self.tk.StringVar(value="")
        self.search_entry = self.ttk.Entry(
            search_frame,
            textvariable=self.search_var,
        )
        self.search_entry.grid(row=0, column=1, sticky="ew")
        self.search_entry.bind("<KeyRelease>", self._on_search_changed)
        self.ttk.Button(
            search_frame,
            text="Clear",
            command=self._clear_search,
        ).grid(row=0, column=2, sticky="w", padx=(6, 12))
        self.folder_view_var = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(
            search_frame,
            text="Folder View",
            variable=self.folder_view_var,
            command=self._toggle_folder_view,
        ).grid(row=0, column=3, sticky="w")
        self.folder_location_var = self.tk.StringVar(value="Folder: archive root")
        self.ttk.Label(
            search_frame,
            textvariable=self.folder_location_var,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        table_frame = self.ttk.Frame(main)
        table_frame.grid(row=3, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        # A Text widget is used instead of a Treeview because Tk's Treeview
        # cells do not reliably wrap long path names.  Each recording is a
        # tagged, read-only-in-practice block that still supports mouse and
        # keyboard selection.
        self.recording_text = self.tk.Text(
            table_frame,
            width=100,
            height=15,
            wrap="char",
            padx=6,
            pady=4,
            takefocus=True,
            cursor="arrow",
        )
        self.recording_text.tag_configure(
            "selected_recording",
            background="#d9eaf7",
        )
        self.recording_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = self.ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.recording_text.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.recording_text.configure(yscrollcommand=scrollbar.set)
        self.recording_text.bind("<Button-1>", self._recording_click)
        self.recording_text.bind("<KeyPress>", self._recording_key)
        self.recording_text.bind("<MouseWheel>", self._recording_mousewheel)
        self.recording_text.bind("<Button-4>", self._recording_mousewheel)
        self.recording_text.bind("<Button-5>", self._recording_mousewheel)

        self.action_frame = self.ttk.LabelFrame(
            main,
            text="Selected recording",
            padding=8,
        )
        self.action_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
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
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.status_var = self.tk.StringVar(value="Scanning...")
        self.ttk.Label(
            main,
            textvariable=self.status_var,
            justify="left",
        ).grid(row=6, column=0, sticky="w", pady=(4, 0))

    def _folder_for_pair(self, pair: RecordingPair) -> PurePosixPath:
        """Return the archive-relative folder containing a recording pair."""

        return PurePosixPath(pair.relative_bioz_path).parent

    @staticmethod
    def _is_within_folder(
        folder: PurePosixPath,
        ancestor: PurePosixPath,
    ) -> bool:
        """Return whether ``folder`` is ``ancestor`` or one of its children."""

        if not ancestor.parts:
            return True
        return folder.parts[: len(ancestor.parts)] == ancestor.parts

    def _folder_label(self, folder: PurePosixPath) -> str:
        if not folder.parts:
            return "archive root"
        return " > ".join(folder.parts)

    @staticmethod
    def _folder_entry_id(folder: PurePosixPath) -> str:
        return f"folder:{folder.as_posix()}"

    def _pair_matches_search(self, pair: RecordingPair) -> bool:
        query = self.search_var.get().strip().casefold()
        if not query:
            return True
        annotation = annotation_for_pair(self.annotations, pair)
        searchable_text = "\n".join(
            (
                pair.display_identifier,
                pair.recording_identifier,
                pair.relative_bioz_path,
                pair.relative_calibrated_path,
                annotation,
            )
        ).casefold()
        return query in searchable_text

    def _entry_text(self, entry: CatalogueEntry) -> str:
        if entry.kind == "parent":
            return "[Parent] .. (parent folder)"
        if entry.kind == "folder":
            return f"[Folder] {entry.label}"

        pair = entry.pair
        if pair is None:
            return entry.label
        label = pair.recording_identifier if self.folder_view_var.get() else pair.display_identifier
        annotation = annotation_for_pair(self.annotations, pair)
        if annotation:
            return f"{label}\n  Annotation: {annotation}"
        return label

    def _build_catalogue_entries(self) -> list[CatalogueEntry]:
        if not self.folder_view_var.get():
            self.folder_location_var.set("Folder View off: all folders")
            return [
                CatalogueEntry(
                    kind="file",
                    entry_id=pair.key,
                    label=pair.display_identifier,
                    pair=pair,
                )
                for pair in self.pairs
                if self._pair_matches_search(pair)
            ]

        current_folder = self.folder_path
        self.folder_location_var.set(
            f"Folder: {self._folder_label(current_folder)}"
        )
        matching_pairs = [
            pair
            for pair in self.pairs
            if self._is_within_folder(
                self._folder_for_pair(pair),
                current_folder,
            )
            and self._pair_matches_search(pair)
        ]

        entries: list[CatalogueEntry] = []
        if current_folder.parts:
            parent = current_folder.parent
            entries.append(
                CatalogueEntry(
                    kind="parent",
                    entry_id=self._folder_entry_id(parent),
                    label=".. (parent folder)",
                    folder_path=parent,
                )
            )

        child_folders: dict[str, PurePosixPath] = {}
        direct_pairs: list[RecordingPair] = []
        for pair in matching_pairs:
            pair_folder = self._folder_for_pair(pair)
            if pair_folder == current_folder:
                direct_pairs.append(pair)
                continue
            remaining_parts = pair_folder.parts[len(current_folder.parts) :]
            if not remaining_parts:
                continue
            child_folder = PurePosixPath(
                *(current_folder.parts + (remaining_parts[0],))
            )
            child_folders[child_folder.as_posix()] = child_folder

        for child_folder in sorted(child_folders.values(), key=lambda path: path.as_posix().casefold()):
            entries.append(
                CatalogueEntry(
                    kind="folder",
                    entry_id=self._folder_entry_id(child_folder),
                    label=child_folder.name,
                    folder_path=child_folder,
                )
            )
        entries.extend(
            CatalogueEntry(
                kind="file",
                entry_id=pair.key,
                label=pair.recording_identifier,
                pair=pair,
            )
            for pair in direct_pairs
        )
        return entries

    def _render_recording_list(self) -> None:
        self.entries = self._build_catalogue_entries()
        self.recording_text.delete("1.0", "end")
        for tag in tuple(self.entry_tag_to_entry):
            self.recording_text.tag_delete(tag)
        self.entry_tag_to_entry.clear()
        self.recording_text.tag_remove("selected_recording", "1.0", "end")

        if not self.entries:
            if not self.pairs:
                message = "No complete recording pairs found yet."
            elif self.search_var.get().strip():
                message = "No recordings match the current search."
            elif self.folder_view_var.get():
                message = "No complete recording pairs in this folder."
            else:
                message = "No recordings match the current search."
            self.recording_text.insert("1.0", message)
            return

        for index, entry in enumerate(self.entries):
            tag = f"entry_{index}"
            start = self.recording_text.index("end-1c")
            self.recording_text.insert("end-1c", self._entry_text(entry))
            end = self.recording_text.index("end-1c")
            self.recording_text.insert("end-1c", "\n")
            self.recording_text.tag_add(tag, start, end)
            self.recording_text.tag_configure(
                tag,
                foreground="#154b7a" if entry.kind != "file" else "#000000",
                spacing3=5,
            )
            self.entry_tag_to_entry[tag] = entry

    def _catalogue_entry_at_index(self, index: str) -> Optional[CatalogueEntry]:
        for tag, entry in self.entry_tag_to_entry.items():
            ranges = self.recording_text.tag_ranges(tag)
            if len(ranges) != 2:
                continue
            if self.recording_text.compare(ranges[0], "<=", index) and self.recording_text.compare(
                index,
                "<",
                ranges[1],
            ):
                return entry

        # A click can land on the newline immediately after a wrapped item.
        # Treat it as part of that item so selection does not feel fragile.
        if self.recording_text.compare(index, ">", "1.0"):
            previous = self.recording_text.index(f"{index} - 1 chars")
            for tag, entry in self.entry_tag_to_entry.items():
                ranges = self.recording_text.tag_ranges(tag)
                if len(ranges) == 2 and self.recording_text.compare(
                    ranges[0], "<=", previous
                ) and self.recording_text.compare(previous, "<", ranges[1]):
                    return entry
        return None

    def _clear_selection(self) -> None:
        self.selected_entry = None
        self.selected_entry_id = None
        self.selected_key = None
        self.recording_text.tag_remove("selected_recording", "1.0", "end")

    def _select_catalogue_entry(self, entry: Optional[CatalogueEntry]) -> None:
        entries_by_id = {candidate.entry_id: candidate for candidate in self.entries}
        if entry is None or entry.entry_id not in entries_by_id:
            self._clear_selection()
            self._on_selection_changed()
            return

        entry = entries_by_id[entry.entry_id]
        self.selected_entry = entry
        self.selected_entry_id = entry.entry_id
        self.selected_key = entry.pair.key if entry.kind == "file" and entry.pair else None
        self.recording_text.tag_remove("selected_recording", "1.0", "end")
        for tag, tag_entry in self.entry_tag_to_entry.items():
            if tag_entry.entry_id != entry.entry_id:
                continue
            ranges = self.recording_text.tag_ranges(tag)
            if len(ranges) == 2:
                self.recording_text.tag_add(
                    "selected_recording",
                    ranges[0],
                    ranges[1],
                )
                self.recording_text.see(ranges[0])
            break
        self._on_selection_changed()

    def _recording_click(self, event: Any) -> str:
        index = self.recording_text.index(f"@{event.x},{event.y}")
        self.recording_text.focus_set()
        entry = self._catalogue_entry_at_index(index)
        if entry is not None and entry.kind in {"folder", "parent"}:
            self._enter_folder(entry.folder_path or PurePosixPath())
        else:
            self._select_catalogue_entry(entry)
        return "break"

    def _recording_key(self, event: Any) -> str:
        if event.keysym in {"Return", "KP_Enter"}:
            if self.selected_entry and self.selected_entry.kind in {"folder", "parent"}:
                self._enter_folder(self.selected_entry.folder_path or PurePosixPath())
            else:
                self._display_plot()
            return "break"

        if event.keysym in {"Up", "Down", "Home", "End"}:
            if not self.entries:
                return "break"
            entry_ids = [entry.entry_id for entry in self.entries]
            if self.selected_entry_id not in entry_ids:
                index = 0 if event.keysym in {"Down", "Home"} else len(self.entries) - 1
            else:
                index = entry_ids.index(self.selected_entry_id)
                if event.keysym == "Up":
                    index = max(0, index - 1)
                elif event.keysym == "Down":
                    index = min(len(self.entries) - 1, index + 1)
                elif event.keysym == "Home":
                    index = 0
                elif event.keysym == "End":
                    index = len(self.entries) - 1
            self._select_catalogue_entry(self.entries[index])
            return "break"

        # The catalogue is intentionally read-only.  Navigation keys are
        # handled above; all other key presses must not edit its contents.
        return "break"

    def _recording_mousewheel(self, event: Any) -> str:
        button = getattr(event, "num", None)
        if button == 4:
            units = -3
        elif button == 5:
            units = 3
        else:
            delta = int(getattr(event, "delta", 0))
            if delta == 0:
                return "break"
            units = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
        self.recording_text.yview_scroll(units, "units")
        return "break"

    def _enter_folder(self, folder: PurePosixPath) -> None:
        self.folder_path = PurePosixPath(*folder.parts)
        self._clear_selection()
        self._render_recording_list()
        self._hide_selected_controls()
        self.details_var.set(
            f"Folder: {self._folder_label(self.folder_path)}. "
            "Select a recording or open a subfolder."
        )
        self.recording_text.focus_set()

    def _toggle_folder_view(self) -> None:
        if not self.folder_view_var.get():
            self.folder_path = PurePosixPath()
        self._clear_selection()
        self._render_recording_list()
        self._hide_selected_controls()
        if self.folder_view_var.get():
            self.details_var.set(
                "Folder View is active. Select a folder or recording."
            )
        else:
            self.details_var.set("Select a recording to see its files.")
        self.recording_text.focus_set()

    def _clear_search(self) -> None:
        self.search_var.set("")
        self._on_search_changed()
        self.search_entry.focus_set()

    def _on_search_changed(self, _event: Any = None) -> None:
        previous_entry_id = self.selected_entry_id
        self._render_recording_list()
        visible_entries = {
            entry.entry_id: entry
            for entry in self.entries
        }
        if previous_entry_id in visible_entries:
            self._select_catalogue_entry(visible_entries[previous_entry_id])
            return

        self._clear_selection()
        self._hide_selected_controls()
        if self.search_var.get().strip():
            self.details_var.set("No selected recording matches the current search.")
        elif self.folder_view_var.get():
            self.details_var.set(
                f"Folder: {self._folder_label(self.folder_path)}. "
                "Select a folder or recording."
            )
        else:
            self.details_var.set("Select a recording to see its files.")

    def _selected_pair(self) -> Optional[RecordingPair]:
        if self.selected_entry and self.selected_entry.kind == "file":
            return self.selected_entry.pair
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
        entry = self.selected_entry
        if entry is None:
            self._hide_selected_controls()
            self.details_var.set("No recording selected.")
            return

        if entry.kind != "file" or entry.pair is None:
            self.selected_key = None
            self._hide_selected_controls()
            folder = entry.folder_path or PurePosixPath()
            if entry.kind == "parent":
                self.details_var.set(
                    f"Parent folder: {self._folder_label(folder)}. "
                    "Press Enter to open it."
                )
            else:
                self.details_var.set(
                    f"Folder: {self._folder_label(folder)}. "
                    "Press Enter to open it."
                )
            return

        pair = entry.pair
        self.selected_key = pair.key
        self.action_frame.grid()
        self._hide_annotation_editor()
        annotation = annotation_for_pair(self.annotations, pair)
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
        self._set_annotation_editor_text(annotation_for_pair(self.annotations, pair))
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
            if pair.legacy_key != pair.key:
                self.annotations.pop(pair.legacy_key, None)
        else:
            self.annotations.pop(pair.key, None)
            self.annotations.pop(pair.legacy_key, None)
        try:
            save_annotations(self.annotations_path, self.annotations)
        except OSError as exc:
            self.messagebox.showerror(
                "Could not save annotation",
                f"The annotation file could not be saved:\n{exc}",
            )
            return False

        self._render_recording_list()
        visible_entry = next(
            (
                entry
                for entry in self.entries
                if entry.kind == "file" and entry.entry_id == pair.key
            ),
            None,
        )
        if visible_entry is not None:
            self._select_catalogue_entry(visible_entry)
        else:
            self._clear_selection()
            self._hide_selected_controls()
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
        previous_entry_id = self.selected_entry_id
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
            key=lambda pair: (pair.recording_identifier, pair.display_identifier),
            reverse=True,
        )
        self.pairs = pairs
        self.pairs_by_key = {pair.key: pair for pair in pairs}
        if self.folder_view_var.get() and self.folder_path.parts:
            current_folder_still_exists = any(
                self._is_within_folder(
                    self._folder_for_pair(pair),
                    self.folder_path,
                )
                for pair in pairs
            )
            if not current_folder_still_exists:
                self.folder_path = PurePosixPath()
        self._render_recording_list()

        visible_entries = {
            entry.entry_id: entry
            for entry in self.entries
        }
        if previous_entry_id in visible_entries:
            self._select_catalogue_entry(visible_entries[previous_entry_id])
            if editor_was_visible:
                self._show_annotation_editor()
                self._set_annotation_editor_text(editor_draft)
        elif not pairs:
            self._clear_selection()
            self._hide_selected_controls()
            if pending:
                self.details_var.set(pending[0])
            else:
                self.details_var.set("No complete recording pairs found yet.")
        else:
            self._clear_selection()
            self._hide_selected_controls()
            if self.search_var.get().strip() and not self.entries:
                self.details_var.set("No recordings match the current search.")
            elif self.folder_view_var.get():
                self.details_var.set(
                    f"Folder: {self._folder_label(self.folder_path)}. "
                    "Select a folder or recording."
                )
            else:
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
    output_directory: Optional[Path],
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
            key=lambda pair: (pair.recording_identifier, pair.display_identifier),
            reverse=True,
        )
        print("\nMAX30009 recordings")
        if pairs:
            for index, pair in enumerate(pairs, start=1):
                annotation = annotation_for_pair(annotations, pair)
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
                if pair.legacy_key != pair.key:
                    annotations.pop(pair.legacy_key, None)
            else:
                annotations.pop(pair.key, None)
                annotations.pop(pair.legacy_key, None)
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
    output_directory: Optional[Path],
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
        else None
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
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

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
