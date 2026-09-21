# MAX30009 BioZ viewer

`plots.py` reads either CSV from a MAX30009 recording. It finds
the companion CSV in the same directory using the shared recording time token
(`HHMMSS`). It supports both the long date form, such as
`20260909_114235`, and the shorter form used by the supplied calibrated file,
`0909_114235`.

## Install

The program uses Python 3, NumPy, and Matplotlib:

```text
python -m pip install numpy matplotlib
```

## Run

Pass only one of the two long CSV names:

```text
python plots.py MAX30009_20260909_114235.bioz.csv
```

You may also pass the directory containing the pair:

```text
python plots.py .
```

## Recording manager

`watch_max30009.py` manages the recordings below an archive root and provides a
selectable, scrollable catalogue of complete CSV pairs:

```text
python watch_max30009.py
```

The manager recursively searches for `.bioz.csv` files and matches each one to
a calibrated CSV in the same folder using the normalized `MMDD_HHMMSS` token.
The displayed name includes the relative folder path, for example:

```text
2026-09-15 Human - MAX30009 > Rod used > Protocol 1 > 114952
```

Long names wrap at 100 characters. The catalogue has a vertical scrollbar and
mouse-wheel support, and the arrow keys plus Enter remain available for
selection and opening. Existing complete pairs are scanned when the manager
starts, so no `--process-existing` flag is needed. Select a row to reveal
`Display Plot` and `Annotate`. The catalogue is refreshed only when `Refresh
Now` is clicked, so editing an annotation cannot be interrupted by a
background scan. Newly added session folders and files appear after the next
manual refresh.

The GUI also has a `Search name/annotation` field. It performs a
case-insensitive partial match against the displayed identifier, relative file
paths, folder names, and annotation text. The default flat view searches the
whole archive. Enable `Folder View` to navigate the archive structure: folders
are shown as entries, clicking a folder or selecting it with the arrow keys and
pressing Enter opens it, and `.. (parent folder)` moves back up. In Folder View
only the recording identifier is shown for files because the current folder
already supplies the path context. Search continues to apply within the
current folder and its descendants.

The annotation editor includes `Copy` and `Paste` buttons, standard keyboard
shortcuts such as Ctrl+C/Ctrl+V, and a right-click menu.

Annotations are stored in `max30009_recording_annotations.json` beside the
script. Version 2 of this sidecar uses archive-relative file paths in its keys,
so equal timestamps in different session folders remain separate. Older
annotation files are still accepted and are migrated when an annotation is
saved. This is a small sidecar file, not a measurement input, and it keeps
annotations available across sessions. If Tkinter is unavailable, use (or the
program will fall back to) a text menu:

```text
python watch_max30009.py --console
```

If the annotation file is malformed, the GUI shows a warning and asks whether
to exit and repair it or proceed without the old annotations. Saving an
annotation after proceeding replaces the malformed file with a valid one.

If a manual refresh might occur while an export is still being written,
`--stable-seconds 3` enables a safety delay; click `Refresh Now` again after
that delay to list the pair. Other useful options are `--directory` to select
a different archive root, `--annotations path/to/annotations.json` for a
different sidecar location, and `--output-dir path/to/output` to override the
default per-session output location. Without `--output-dir`, calculated CSVs,
stable-period CSVs, and decoded-settings JSON files are written beside the
selected recording's source CSVs.

The default initial smooth-period threshold is ±100 Ω. The interactive window
contains a working `Smooth ±Ω` slider that moves in 10 Ω steps; changing it
redraws the red horizontal stable-period markers immediately. The `Value (Ω)`
box beside the slider accepts a direct threshold and moves the slider to match
it. The slider and input are retained by the figure so they continue
responding after the window has opened.

By default, both plots show the complete recording. To inspect a moving time
window, select `Custom Time Window`, enter a positive length in seconds, and
press Enter or leave the input field. The horizontal `Window start (s)` slider
then moves continuously through the complete recording. For example, a length
of 5 seconds can show 1.0–6.0 s, 1.2–6.2 s, and so on. Unselecting the checkbox
returns to the full-recording view.

The magnitude graph also has independent checkboxes for `Limit Upper
(Impedance Magnitude)` and `Limit Lower (Impedance Magnitude)`. When selected,
each displays an ohm input. Values outside the selected limits are cropped
from the magnitude view. With both unchecked, the y-axis returns to the global
minimum and maximum of the measured data.

The phase graph uses a fixed −180° to 180° vertical range, with ticks at
−180°, −90°, 0°, 90°, and 180°.

The shared time axis uses 4-second major grid spacing. Clicking either plot
displays the nearest sample's time, magnitude, phase, and calibrated I/Q
values. If that sample lies within a detected stable period, the readout also
includes the period's average impedance.

The `Smoothing (s)` slider is a temporary, display-only control. It ranges from
0 to 0.5 seconds in 0.1-second steps; 0 means no smoothing. A selected window
applies a centred moving average to the real and imaginary impedance before
they are converted back to displayed magnitude and phase. The source data,
click readouts, stability detection, and exported CSV files are not changed,
and the smoothing selection is discarded when the program closes.

`Copy to Clipboard` copies a full-width PNG screenshot from the top of the
figure through the phase time-axis label, while leaving the controls below the
plots out of the image. The capture uses the actual rendered pixel buffer, so
it also works correctly on high-DPI Windows displays. On Windows this uses the
built-in image clipboard format. On Linux, an image clipboard helper such as
`wl-copy` or `xclip` is required; if no supported clipboard backend is
available, the program reports that in the readout area.

Useful options:

```text
python plots.py recording.csv \
    --smooth-threshold 100 \
    --stability-window 1.0 \
    --min-stable-duration 0.5 \
    --mag-ylim 0 600 \
    --save-plot
```

`--mag-ylim LOWER UPPER` remains available for starting the display with both
magnitude limit controls enabled. The on-screen controls can then be adjusted
without rerunning the program.

The program writes three files to `--output-dir`:

- `*_calculated.csv`: timestamp, time in seconds, calibrated I/Q, real and
  imaginary impedance, magnitude, and phase;
- `*_stable_periods.csv`: detected intervals and their statistics;
- `*_decoded_settings.json`: register values, decoded timing/current/gain,
  scale factor, and validation results.

## Calculation notes

The calibrated columns in the supplied second CSV are already the corrected
real and imaginary components in count units. The program therefore multiplies
them by the ohms/count scale; it does not apply the calibration coefficients a
second time. It also independently applies the MAX30009 calibration equations
to the raw columns and checks that the exported calibrated columns agree after
rounding.

The ohms/count scale is decoded for current-drive mode from registers 0x17,
0x18, 0x1A, 0x20, 0x22, and 0x24. The current-drive equation is the MAX30009
datasheet equation using the typical 1 V VREF. The current implementation
requires `--external-ref-hz` for an external PLL reference and `--rext-ohm`
for an external current-setting resistor. Voltage-drive and H-bridge absolute
impedance conversion are intentionally not implemented yet because they need
their respective divider/series-resistor parameters.

The default time axis is based on the logged epoch-millisecond timestamps, with
the first aligned sample set to 0 s. Use `--time-source sample-rate` if you
instead want time derived from the decoded BioZ sample rate.

The stability detector currently defines a stable window as one whose rolling
range is no greater than `2 × threshold`; thus ±100 Ω corresponds to a maximum
rolling range of 200 Ω. The detection window, minimum interval duration, merge
gap, and threshold are adjustable.

Datasheet:
<https://www.analog.com/media/en/technical-documentation/data-sheets/MAX30009.pdf>
