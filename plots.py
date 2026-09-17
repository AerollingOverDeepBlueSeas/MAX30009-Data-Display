#!/usr/bin/env python3
"""MAX30009 BioZ recording viewer and impedance calculator.

This program accepts either CSV from a MAX30009 recording.  It identifies the
companion CSV using the HHMMSS recording token in the filenames, decodes the
MAX30009 register snapshot, converts the supplied calibrated I/Q values to
impedance magnitude and phase, and opens an interactive Matplotlib view.

The supplied calibrated I/Q columns are the corrected real and imaginary
components in ADC-count units.  They are therefore converted as follows:

    Z_real_ohm = calibrated_I_counts * ohm_per_count
    Z_imag_ohm = calibrated_Q_counts * ohm_per_count
    |Z|         = hypot(Z_real_ohm, Z_imag_ohm)
    phase       = atan2(Z_imag_ohm, Z_real_ohm)

For current-drive mode, ohm_per_count is decoded from the MAX30009 register
settings using the datasheet equation:

    ohm_per_count = VREF /
        (2**19 * BIOZ_GAIN * (2/pi) * I_MAG_peak)

The MAX30009 datasheet is available at:
https://www.analog.com/media/en/technical-documentation/data-sheets/MAX30009.pdf

Examples:
    python plots.py MAX30009_20260909_114235.bioz.csv
    python plots.py MAX30009_20260909_114235.bioz.csv \
        MAX30009_BioZ-vs-time-cal-results_0909_114235.csv

The program writes calculated samples, detected stable periods, and decoded
settings to the output directory.  Use --no-show with a non-interactive
Matplotlib backend when only files are wanted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, Slider, TextBox


DATASHEET_URL = (
    "https://www.analog.com/media/en/technical-documentation/data-sheets/"
    "MAX30009.pdf"
)


class BioZViewerError(RuntimeError):
    """An expected, user-correctable input or configuration error."""


def normalise_text(value: str) -> str:
    """Normalise CSV labels for tolerant matching."""

    return " ".join(str(value).strip().lower().lstrip("\ufeff").split())


def parse_number(value: str) -> float:
    """Parse a decimal or hexadecimal CSV value."""

    text = str(value).strip().lstrip("\ufeff")
    if not text:
        raise ValueError("empty numeric field")
    if text.lower().startswith(("0x", "+0x", "-0x")):
        sign = -1 if text.startswith("-") else 1
        unsigned = text[1:] if text[:1] in "+-" else text
        return float(sign * int(unsigned, 16))
    return float(text)


def is_numeric(value: str) -> bool:
    try:
        parse_number(value)
    except (TypeError, ValueError):
        return False
    return True


def read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.reader(handle))
    except OSError as exc:
        raise BioZViewerError(f"Could not read {path}: {exc}") from exc


def filename_tokens(path: Path) -> dict[str, Optional[str]]:
    """Extract long/short date and HHMMSS tokens from a recording filename."""

    name = path.name
    full_match = re.findall(r"(?<!\d)(20\d{6})[_-](\d{6})(?!\d)", name)
    short_match = re.findall(r"(?<!\d)(\d{4})[_-](\d{6})(?!\d)", name)
    time_match = re.findall(r"(?<!\d)(\d{6})(?!\d)", name)

    date_key: Optional[str] = None
    time_key: Optional[str] = None
    if full_match:
        date_key, time_key = full_match[-1]
    elif short_match:
        date_key, time_key = short_match[-1]
    elif time_match:
        time_key = time_match[-1]

    # A long date and a short MMDD date can identify the same recording.
    if date_key and len(date_key) == 8:
        comparable_date = date_key[-4:]
    else:
        comparable_date = date_key

    return {
        "date": date_key,
        "comparable_date": comparable_date,
        "time": time_key,
    }


def same_recording(a: Path, b: Path) -> bool:
    ta = filename_tokens(a)
    tb = filename_tokens(b)
    if not ta["time"] or not tb["time"] or ta["time"] != tb["time"]:
        return False
    if ta["comparable_date"] and tb["comparable_date"]:
        return ta["comparable_date"] == tb["comparable_date"]
    return True


def looks_like_measurement_csv(path: Path) -> bool:
    try:
        rows = read_csv_rows(path)[:12]
    except BioZViewerError:
        return False
    text = " ".join(normalise_text(cell) for row in rows for cell in row)
    return "max30009reg0x17" in text and "timestamp" in text and "biozi" in text


def looks_like_calibrated_csv(path: Path) -> bool:
    try:
        rows = read_csv_rows(path)[:8]
    except BioZViewerError:
        return False
    text = " ".join(normalise_text(cell) for row in rows for cell in row)
    return (
        "i coef" in text
        and "ohm/count" in text
        and "bioz i adc calibrated" in text
    )


def find_recording_pair(
    input_path: Path,
    companion_input_path: Optional[Path] = None,
) -> tuple[Path, Path, str]:
    """Return (measurement_csv, calibrated_csv, identifier)."""

    if companion_input_path is not None:
        if input_path.is_dir() or companion_input_path.is_dir():
            raise BioZViewerError(
                "When two inputs are supplied, both inputs must be CSV files"
            )
        for path in (input_path, companion_input_path):
            if not path.exists():
                raise BioZViewerError(f"Input file does not exist: {path}")
            if path.suffix.lower() != ".csv":
                raise BioZViewerError(f"Input is not a CSV file: {path}")
        if not same_recording(input_path, companion_input_path):
            raise BioZViewerError(
                "The two supplied CSV files do not have the same recording identifier"
            )

        if looks_like_measurement_csv(input_path) and looks_like_calibrated_csv(
            companion_input_path
        ):
            measurement_path = input_path
            calibrated_path = companion_input_path
        elif looks_like_calibrated_csv(input_path) and looks_like_measurement_csv(
            companion_input_path
        ):
            measurement_path = companion_input_path
            calibrated_path = input_path
        else:
            raise BioZViewerError(
                "The two supplied CSV files must be one MAX30009 measurement "
                "CSV and one calibrated CSV"
            )
    elif input_path.is_dir():
        candidates = sorted(input_path.glob("*.csv"))
        if not candidates:
            raise BioZViewerError(f"No CSV files found in {input_path}")
        measurement_candidates = [p for p in candidates if looks_like_measurement_csv(p)]
        calibrated_candidates = [p for p in candidates if looks_like_calibrated_csv(p)]
        pairs = [
            (m, c)
            for m in measurement_candidates
            for c in calibrated_candidates
            if same_recording(m, c)
        ]
        if len(pairs) != 1:
            description = ", ".join(f"{m.name} + {c.name}" for m, c in pairs)
            raise BioZViewerError(
                "The directory does not contain exactly one identifiable CSV pair. "
                f"Candidate pairs: {description or 'none'}"
            )
        measurement_path, calibrated_path = pairs[0]
    else:
        if not input_path.exists():
            raise BioZViewerError(f"Input file does not exist: {input_path}")
        if input_path.suffix.lower() != ".csv":
            raise BioZViewerError("The input must be a CSV file or a directory of CSV files")

        if looks_like_measurement_csv(input_path):
            measurement_path = input_path
            kind = "measurement"
        elif looks_like_calibrated_csv(input_path):
            calibrated_path = input_path
            kind = "calibrated"
        else:
            raise BioZViewerError(
                f"Could not identify {input_path.name} as a MAX30009 measurement "
                "or calibrated CSV"
            )

        companions = []
        for candidate in sorted(input_path.parent.glob("*.csv")):
            if candidate == input_path or not same_recording(input_path, candidate):
                continue
            if kind == "measurement" and looks_like_calibrated_csv(candidate):
                companions.append(candidate)
            elif kind == "calibrated" and looks_like_measurement_csv(candidate):
                companions.append(candidate)

        if len(companions) != 1:
            names = ", ".join(p.name for p in companions) or "none"
            raise BioZViewerError(
                f"Could not identify exactly one companion CSV for {input_path.name}. "
                f"Matching candidates: {names}"
            )

        if kind == "measurement":
            calibrated_path = companions[0]
        else:
            measurement_path = companions[0]

    tokens = filename_tokens(measurement_path)
    identifier = "_".join(
        token for token in (tokens.get("date"), tokens.get("time")) if token
    )
    identifier = identifier or measurement_path.stem
    return measurement_path, calibrated_path, identifier


def parse_register_snapshot(path: Path) -> tuple[dict[int, int], list[str]]:
    """Parse the two-row register snapshot at the beginning of .bioz.csv."""

    rows = read_csv_rows(path)
    if len(rows) < 4:
        raise BioZViewerError(f"Register snapshot is incomplete in {path.name}")

    header_cells = rows[0] + rows[1]
    value_cells = rows[2] + rows[3]
    registers: dict[int, int] = {}
    warnings: list[str] = []

    for index, (header, value) in enumerate(zip(header_cells, value_cells)):
        match = re.search(r"max30009reg0x([0-9a-f]{2})$", header.strip(), re.I)
        if not match:
            continue
        address = int(match.group(1), 16)

        # The supplied exporter labels the 0x42 column as a second 0x41.
        # It occurs between 0x41 and 0x43, so repair that known export typo.
        if address == 0x41 and address in registers:
            if 0x42 not in registers:
                address = 0x42
                warnings.append(
                    "The register CSV labels the second 0x41 field as 0x41; "
                    "it was interpreted as 0x42 from the 0x41,0x42,0x43 sequence."
                )
            else:
                warnings.append(
                    f"Duplicate register 0x{address:02X} at snapshot column {index}; "
                    "the first value was retained."
                )
                continue

        try:
            parsed_value = int(parse_number(value)) & 0xFF
        except ValueError as exc:
            raise BioZViewerError(
                f"Invalid value {value!r} for register 0x{address:02X} in {path.name}"
            ) from exc
        registers[address] = parsed_value

    return registers, warnings


def require_registers(registers: dict[int, int], addresses: Iterable[int]) -> None:
    missing = [f"0x{address:02X}" for address in addresses if address not in registers]
    if missing:
        raise BioZViewerError(
            "The register snapshot is missing required register(s): " + ", ".join(missing)
        )


@dataclass
class DecodedSettings:
    registers: dict[str, str]
    ref_clk_sel: int
    clk_freq_sel: int
    clk_fine_tune_code: int
    clk_fine_tune_percent: float
    ref_clk_hz: float
    mdiv_code: int
    m: int
    ndiv_code: int
    ndiv: int
    kdiv_code: int
    kdiv: int
    dac_osr_code: int
    dac_osr: int
    adc_osr_code: int
    adc_osr: int
    pll_clk_hz: float
    stimulus_frequency_hz: float
    sample_rate_hz: float
    drive_mode_code: int
    drive_mode: str
    gain_code: int
    gain_vv: float
    external_resistor_selected: bool
    vdrv_mag_code: int
    idrv_range_code: int
    internal_range_resistor_ohm: Optional[float]
    vdrvr_peak_v: Optional[float]
    stimulus_current_peak_a: Optional[float]
    stimulus_current_rms_a: Optional[float]
    vref_v: float
    ohm_per_count: Optional[float]
    exported_ohm_per_count: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


KDIV_VALUES = {
    0x0: 1,
    0x1: 2,
    0x2: 4,
    0x3: 8,
    0x4: 16,
    0x5: 32,
    0x6: 64,
    0x7: 128,
    0x8: 256,
    0x9: 512,
    0xA: 1024,
    0xB: 2048,
    0xC: 4096,
    0xD: 8192,
    0xE: 8192,
    0xF: 8192,
}
DAC_OSR_VALUES = [32, 64, 128, 256]
ADC_OSR_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024]
GAIN_VALUES = [1.0, 2.0, 5.0, 10.0]
DRIVE_MODES = ["current", "voltage", "h_bridge", "standby"]
INTERNAL_RANGE_RESISTORS = {
    0: 552_500.0,
    1: 110_500.0,
    2: 5_525.0,
    3: 276.25,
}

# From MAX30009 datasheet Table 5. Values are the DRVR peak voltage in mV.
VDRVR_PEAK_MV = {
    0: [12.5, 25.0, 62.5, 125.0],
    1: [50.0, 100.0, 250.0, 500.0],
    2: [50.0, 100.0, 250.0, 500.0],
    3: [50.0, 100.0, 250.0, 500.0],
}


def signed_5_bit(value: int) -> int:
    return value - 32 if value & 0x10 else value


def decode_settings(
    registers: dict[int, int],
    *,
    vref_v: float = 1.0,
    external_ref_hz: Optional[float] = None,
    rext_ohm: Optional[float] = None,
    exported_ohm_per_count: Optional[float] = None,
    snapshot_warnings: Optional[list[str]] = None,
) -> DecodedSettings:
    """Decode timing, gain, drive, and scale settings from the datasheet map."""

    require_registers(registers, (0x17, 0x18, 0x1A, 0x20, 0x22, 0x24))
    r17, r18, r1a = registers[0x17], registers[0x18], registers[0x1A]
    r20, r22, r24 = registers[0x20], registers[0x22], registers[0x24]
    warnings = list(snapshot_warnings or [])

    ref_clk_sel = (r1a >> 6) & 0x01
    clk_freq_sel = (r1a >> 5) & 0x01
    fine_code = r1a & 0x1F
    fine_percent = signed_5_bit(fine_code) * 0.2

    if ref_clk_sel:
        if external_ref_hz is None:
            raise BioZViewerError(
                "REF_CLK_SEL=1 indicates an external reference clock, but its frequency "
                "was not supplied. Use --external-ref-hz."
            )
        ref_clk_hz = float(external_ref_hz)
        if fine_code:
            warnings.append("CLK_FINE_TUNE is ignored when an external reference is selected.")
    else:
        ref_clk_hz = 32_768.0 if clk_freq_sel else 32_000.0
        ref_clk_hz *= 1.0 + fine_percent / 100.0

    mdiv_code = ((r17 >> 6) & 0x03) << 8 | r18
    m = mdiv_code + 1
    ndiv_code = (r17 >> 5) & 0x01
    ndiv = 1024 if ndiv_code else 512
    kdiv_code = (r17 >> 1) & 0x0F
    kdiv = KDIV_VALUES[kdiv_code]

    dac_osr_code = (r20 >> 6) & 0x03
    adc_osr_code = (r20 >> 3) & 0x07
    dac_osr = DAC_OSR_VALUES[dac_osr_code]
    adc_osr = ADC_OSR_VALUES[adc_osr_code]

    pll_clk_hz = m * ref_clk_hz
    stimulus_frequency_hz = pll_clk_hz / (kdiv * dac_osr)
    sample_rate_hz = pll_clk_hz / (ndiv * adc_osr)

    drive_mode_code = r22 & 0x03
    drive_mode = DRIVE_MODES[drive_mode_code]
    gain_code = r24 & 0x03
    gain_vv = GAIN_VALUES[gain_code]
    ext_res = bool((r22 >> 7) & 0x01)
    vdrv_mag_code = (r22 >> 4) & 0x03
    idrv_range_code = (r22 >> 2) & 0x03

    internal_resistor = INTERNAL_RANGE_RESISTORS.get(idrv_range_code)
    vdrvr_peak_v: Optional[float] = None
    current_peak_a: Optional[float] = None
    current_rms_a: Optional[float] = None
    ohm_per_count: Optional[float] = None

    if drive_mode == "current":
        vdrvr_peak_v = VDRVR_PEAK_MV[idrv_range_code][vdrv_mag_code] * 1e-3
        if ext_res:
            if rext_ohm is None:
                raise BioZViewerError(
                    "BIOZ_EXT_RES=1 selects an external resistor. Use --rext-ohm so "
                    "the current and impedance scale can be calculated."
                )
            if rext_ohm <= 0:
                raise BioZViewerError("--rext-ohm must be positive")
            current_peak_a = vdrvr_peak_v / rext_ohm
        else:
            assert internal_resistor is not None
            current_peak_a = vdrvr_peak_v / internal_resistor
        current_rms_a = current_peak_a / math.sqrt(2.0)
        ohm_per_count = vref_v / (
            (2**19) * gain_vv * (2.0 / math.pi) * current_peak_a
        )
    else:
        warnings.append(
            f"Drive mode is {drive_mode!r}; this viewer currently calculates absolute "
            "impedance only for current-drive mode."
        )

    return DecodedSettings(
        registers={f"0x{key:02X}": f"0x{value:02X}" for key, value in sorted(registers.items())},
        ref_clk_sel=ref_clk_sel,
        clk_freq_sel=clk_freq_sel,
        clk_fine_tune_code=fine_code,
        clk_fine_tune_percent=fine_percent,
        ref_clk_hz=ref_clk_hz,
        mdiv_code=mdiv_code,
        m=m,
        ndiv_code=ndiv_code,
        ndiv=ndiv,
        kdiv_code=kdiv_code,
        kdiv=kdiv,
        dac_osr_code=dac_osr_code,
        dac_osr=dac_osr,
        adc_osr_code=adc_osr_code,
        adc_osr=adc_osr,
        pll_clk_hz=pll_clk_hz,
        stimulus_frequency_hz=stimulus_frequency_hz,
        sample_rate_hz=sample_rate_hz,
        drive_mode_code=drive_mode_code,
        drive_mode=drive_mode,
        gain_code=gain_code,
        gain_vv=gain_vv,
        external_resistor_selected=ext_res,
        vdrv_mag_code=vdrv_mag_code,
        idrv_range_code=idrv_range_code,
        internal_range_resistor_ohm=internal_resistor,
        vdrvr_peak_v=vdrvr_peak_v,
        stimulus_current_peak_a=current_peak_a,
        stimulus_current_rms_a=current_rms_a,
        vref_v=vref_v,
        ohm_per_count=ohm_per_count,
        exported_ohm_per_count=exported_ohm_per_count,
        warnings=warnings,
    )


@dataclass
class MeasurementLog:
    timestamp_ms: np.ndarray
    sample_number: np.ndarray
    raw_i_counts: np.ndarray
    raw_q_counts: np.ndarray


def read_measurement_log(path: Path) -> MeasurementLog:
    rows = read_csv_rows(path)
    header_index = None
    for index, row in enumerate(rows):
        labels = [normalise_text(cell) for cell in row]
        if "timestamp" in labels and ("samplenum" in labels or "sample num" in labels):
            header_index = index
            break
    if header_index is None:
        raise BioZViewerError(f"Could not find the timestamp data header in {path.name}")

    headers = [normalise_text(cell) for cell in rows[header_index]]
    try:
        timestamp_index = headers.index("timestamp")
        sample_index = headers.index("samplenum")
        i_index = headers.index("biozi")
        q_index = headers.index("biozq")
    except ValueError as exc:
        raise BioZViewerError(
            f"The measurement header in {path.name} does not contain timestamp, "
            "sampleNum, BIOZI, and BIOZQ"
        ) from exc

    timestamp: list[float] = []
    sample_number: list[float] = []
    raw_i: list[float] = []
    raw_q: list[float] = []
    for row in rows[header_index + 1 :]:
        if len(row) <= max(timestamp_index, sample_index, i_index, q_index):
            continue
        if not is_numeric(row[timestamp_index]) or not is_numeric(row[sample_index]):
            continue
        try:
            timestamp.append(parse_number(row[timestamp_index]))
            sample_number.append(parse_number(row[sample_index]))
            raw_i.append(parse_number(row[i_index]))
            raw_q.append(parse_number(row[q_index]))
        except ValueError:
            continue

    if not timestamp:
        raise BioZViewerError(f"No measurement samples found in {path.name}")
    return MeasurementLog(
        timestamp_ms=np.asarray(timestamp, dtype=float),
        sample_number=np.asarray(sample_number, dtype=float),
        raw_i_counts=np.asarray(raw_i, dtype=float),
        raw_q_counts=np.asarray(raw_q, dtype=float),
    )


@dataclass
class CalibratedLog:
    metadata: dict[str, float]
    raw_i_counts: Optional[np.ndarray]
    raw_q_counts: Optional[np.ndarray]
    calibrated_i_counts: np.ndarray
    calibrated_q_counts: np.ndarray


def read_calibrated_log(path: Path) -> CalibratedLog:
    rows = read_csv_rows(path)
    metadata: dict[str, float] = {}
    if rows:
        first_row = rows[0]
        for index in range(0, len(first_row) - 1, 2):
            key = first_row[index].strip().lstrip("\ufeff")
            value = first_row[index + 1].strip()
            if key and is_numeric(value):
                metadata[key] = parse_number(value)

    header_index = None
    for index, row in enumerate(rows):
        labels = [normalise_text(cell) for cell in row]
        if any("bioz i adc calibrated" in label for label in labels):
            header_index = index
            break
    if header_index is None:
        raise BioZViewerError(f"Could not find calibrated data header in {path.name}")

    headers = [normalise_text(cell) for cell in rows[header_index]]

    def locate(fragment: str) -> Optional[int]:
        for index, label in enumerate(headers):
            if fragment in label:
                return index
        return None

    i_cal_index = locate("bioz i adc calibrated")
    q_cal_index = locate("bioz q adc calibrated")
    i_raw_index = locate("bioz i adc (counts)")
    q_raw_index = locate("bioz q adc (counts)")
    if i_cal_index is None or q_cal_index is None:
        raise BioZViewerError(
            f"The calibrated data header in {path.name} is missing calibrated I/Q columns"
        )

    raw_i: list[float] = []
    raw_q: list[float] = []
    cal_i: list[float] = []
    cal_q: list[float] = []
    for row in rows[header_index + 1 :]:
        if len(row) <= max(i_cal_index, q_cal_index):
            continue
        if not is_numeric(row[i_cal_index]) or not is_numeric(row[q_cal_index]):
            continue
        cal_i.append(parse_number(row[i_cal_index]))
        cal_q.append(parse_number(row[q_cal_index]))
        if i_raw_index is not None and q_raw_index is not None and len(row) > max(i_raw_index, q_raw_index):
            if is_numeric(row[i_raw_index]) and is_numeric(row[q_raw_index]):
                raw_i.append(parse_number(row[i_raw_index]))
                raw_q.append(parse_number(row[q_raw_index]))

    if not cal_i:
        raise BioZViewerError(f"No calibrated samples found in {path.name}")

    raw_i_array: Optional[np.ndarray] = None
    raw_q_array: Optional[np.ndarray] = None
    if len(raw_i) == len(cal_i) and len(raw_q) == len(cal_q):
        raw_i_array = np.asarray(raw_i, dtype=float)
        raw_q_array = np.asarray(raw_q, dtype=float)

    return CalibratedLog(
        metadata=metadata,
        raw_i_counts=raw_i_array,
        raw_q_counts=raw_q_array,
        calibrated_i_counts=np.asarray(cal_i, dtype=float),
        calibrated_q_counts=np.asarray(cal_q, dtype=float),
    )


def reproduce_calibrated_columns(
    raw_i: np.ndarray,
    raw_q: np.ndarray,
    calibrated_i: np.ndarray,
    calibrated_q: np.ndarray,
    metadata: dict[str, float],
) -> dict[str, Any]:
    """Check the supplied calibrated columns against the datasheet correction equations."""

    required = [
        "I Coef",
        "Q Coef",
        "I Phase Coef",
        "Q Phase Coef",
        "I Offset",
        "Q Offset",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        return {"available": False, "reason": f"missing metadata: {', '.join(missing)}"}

    i_coef = metadata["I Coef"]
    q_coef = metadata["Q Coef"]
    i_phase = math.radians(metadata["I Phase Coef"])
    q_phase = math.radians(metadata["Q Phase Coef"])
    i_offset = metadata["I Offset"]
    q_offset = metadata["Q Offset"]

    i_load_offset = raw_i - i_offset
    q_load_offset = raw_q - q_offset
    i_real = (i_load_offset / i_coef) * math.cos(i_phase)
    i_imag = (i_load_offset / i_coef) * math.sin(i_phase)
    q_real = (q_load_offset / q_coef) * math.sin(q_phase)
    q_imag = (q_load_offset / q_coef) * math.cos(q_phase)
    predicted_i = i_real - q_real
    predicted_q = i_imag + q_imag

    residual_i = predicted_i - calibrated_i
    residual_q = predicted_q - calibrated_q
    predicted_rounded_i = np.rint(predicted_i)
    predicted_rounded_q = np.rint(predicted_q)
    exact_i = np.count_nonzero(predicted_rounded_i == calibrated_i)
    exact_q = np.count_nonzero(predicted_rounded_q == calibrated_q)

    return {
        "available": True,
        "sample_count": int(len(calibrated_i)),
        "max_absolute_residual_i_counts": float(np.max(np.abs(residual_i))),
        "max_absolute_residual_q_counts": float(np.max(np.abs(residual_q))),
        "rounded_matches_i": int(exact_i),
        "rounded_matches_q": int(exact_q),
        "all_rounded_values_match": bool(
            exact_i == len(calibrated_i) and exact_q == len(calibrated_q)
        ),
    }


@dataclass
class CalculatedData:
    time_s: np.ndarray
    timestamp_ms: np.ndarray
    sample_number: np.ndarray
    raw_i_counts: np.ndarray
    raw_q_counts: np.ndarray
    calibrated_i_counts: np.ndarray
    calibrated_q_counts: np.ndarray
    real_ohm: np.ndarray
    imag_ohm: np.ndarray
    magnitude_ohm: np.ndarray
    phase_deg: np.ndarray


def calculate_impedance(
    measurement: MeasurementLog,
    calibrated: CalibratedLog,
    settings: DecodedSettings,
    *,
    time_source: str = "timestamp",
) -> tuple[CalculatedData, list[str]]:
    if settings.ohm_per_count is None:
        raise BioZViewerError(
            "No ohm/count conversion is available. The current implementation supports "
            "only current-drive MAX30009 recordings."
        )

    warnings: list[str] = []
    n = min(len(measurement.timestamp_ms), len(calibrated.calibrated_i_counts))
    if len(measurement.timestamp_ms) != len(calibrated.calibrated_i_counts):
        warnings.append(
            f"The measurement CSV has {len(measurement.timestamp_ms)} samples while the "
            f"calibrated CSV has {len(calibrated.calibrated_i_counts)}; the first {n} "
            "samples were aligned by order."
        )
    if n < 2:
        raise BioZViewerError("At least two aligned samples are required")

    timestamp_ms = measurement.timestamp_ms[:n]
    if time_source == "timestamp":
        # The logger timestamps are epoch milliseconds; use their differences
        # so the first logged sample is exactly t=0.
        time_s = (timestamp_ms - timestamp_ms[0]) / 1000.0
    elif time_source == "sample-rate":
        time_s = (
            measurement.sample_number[:n] - measurement.sample_number[0]
        ) / settings.sample_rate_hz
    else:
        raise BioZViewerError(f"Unknown time source: {time_source}")
    if np.any(np.diff(time_s) <= 0):
        warnings.append("Some timestamps are not strictly increasing.")

    raw_i = measurement.raw_i_counts[:n]
    raw_q = measurement.raw_q_counts[:n]
    cal_i = calibrated.calibrated_i_counts[:n]
    cal_q = calibrated.calibrated_q_counts[:n]
    real_ohm = cal_i * settings.ohm_per_count
    imag_ohm = cal_q * settings.ohm_per_count
    magnitude_ohm = np.hypot(real_ohm, imag_ohm)
    phase_deg = np.degrees(np.arctan2(imag_ohm, real_ohm))

    if calibrated.raw_i_counts is not None and calibrated.raw_q_counts is not None:
        compare_n = min(n, len(calibrated.raw_i_counts), len(calibrated.raw_q_counts))
        raw_difference = np.maximum(
            np.abs(raw_i[:compare_n] - calibrated.raw_i_counts[:compare_n]),
            np.abs(raw_q[:compare_n] - calibrated.raw_q_counts[:compare_n]),
        )
        if np.any(raw_difference != 0):
            warnings.append(
                "The raw I/Q columns in the calibrated CSV do not exactly match the "
                "measurement CSV at every aligned row."
            )

    return (
        CalculatedData(
            time_s=time_s,
            timestamp_ms=timestamp_ms,
            sample_number=measurement.sample_number[:n],
            raw_i_counts=raw_i,
            raw_q_counts=raw_q,
            calibrated_i_counts=cal_i,
            calibrated_q_counts=cal_q,
            real_ohm=real_ohm,
            imag_ohm=imag_ohm,
            magnitude_ohm=magnitude_ohm,
            phase_deg=phase_deg,
        ),
        warnings,
    )


@dataclass
class StablePeriod:
    start_s: float
    end_s: float
    duration_s: float
    mean_ohm: float
    median_ohm: float
    min_ohm: float
    max_ohm: float
    range_ohm: float
    threshold_ohm: float
    window_s: float
    sample_count: int


def detect_stable_periods(
    time_s: np.ndarray,
    magnitude_ohm: np.ndarray,
    *,
    threshold_ohm: float = 100.0,
    window_s: float = 1.0,
    min_duration_s: float = 0.5,
    merge_gap_s: float = 0.2,
) -> list[StablePeriod]:
    """Detect windows whose magnitude range is no more than ±threshold.

    A centered rolling window is considered stable when max(window)-min(window)
    <= 2*threshold.  This is deliberately a transparent first definition of
    “smooth”; the threshold, window, and minimum duration are all adjustable.
    """

    if threshold_ohm <= 0 or window_s <= 0 or min_duration_s < 0 or merge_gap_s < 0:
        raise BioZViewerError(
            "Stability threshold and window must be positive; durations cannot be negative"
        )
    if len(time_s) != len(magnitude_ohm) or len(time_s) < 3:
        return []

    dt = np.diff(time_s)
    positive_dt = dt[dt > 0]
    if len(positive_dt) == 0:
        return []
    samples_per_window = max(3, int(round(window_s / float(np.median(positive_dt)))))
    if samples_per_window > len(magnitude_ohm):
        return []

    # Use NaN-aware rolling extrema without requiring pandas.
    series = np.asarray(magnitude_ohm, dtype=float)
    half = samples_per_window // 2
    rolling_min = np.full(len(series), np.nan)
    rolling_max = np.full(len(series), np.nan)
    for index in range(half, len(series) - (samples_per_window - half - 1)):
        segment = series[index - half : index - half + samples_per_window]
        if np.all(np.isfinite(segment)):
            rolling_min[index] = np.min(segment)
            rolling_max[index] = np.max(segment)

    stable_mask = np.isfinite(rolling_min) & np.isfinite(rolling_max)
    stable_mask &= (rolling_max - rolling_min) <= 2.0 * threshold_ohm
    if not np.any(stable_mask):
        return []

    # Convert the Boolean mask into intervals, allowing short gaps to be merged.
    true_indices = np.flatnonzero(stable_mask)
    groups: list[list[int]] = []
    current_group: list[int] = [int(true_indices[0])]
    current_min = float(series[current_group[0]])
    current_max = current_min
    previous = current_group[0]
    median_dt = float(np.median(positive_dt))
    for index_value in true_indices[1:]:
        index = int(index_value)
        segment = series[previous : index + 1]
        candidate_min = min(current_min, float(np.min(segment)))
        candidate_max = max(current_max, float(np.max(segment)))
        short_gap = time_s[index] - time_s[previous] <= merge_gap_s + median_dt
        # Do not merge sections if the resulting reported interval would no
        # longer satisfy the requested ±threshold criterion.
        if short_gap and candidate_max - candidate_min <= 2.0 * threshold_ohm:
            current_group.append(index)
            current_min, current_max = candidate_min, candidate_max
        else:
            groups.append(current_group)
            current_group = [index]
            current_min = float(series[index])
            current_max = current_min
        previous = index
    groups.append(current_group)

    periods: list[StablePeriod] = []
    for group in groups:
        start_index, end_index = group[0], group[-1]
        duration = float(time_s[end_index] - time_s[start_index])
        if duration < min_duration_s:
            continue
        values = series[start_index : end_index + 1]
        periods.append(
            StablePeriod(
                start_s=float(time_s[start_index]),
                end_s=float(time_s[end_index]),
                duration_s=duration,
                mean_ohm=float(np.mean(values)),
                median_ohm=float(np.median(values)),
                min_ohm=float(np.min(values)),
                max_ohm=float(np.max(values)),
                range_ohm=float(np.max(values) - np.min(values)),
                threshold_ohm=float(threshold_ohm),
                window_s=float(window_s),
                sample_count=int(len(values)),
            )
        )
    return periods


def write_calculated_csv(path: Path, data: CalculatedData) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "timestamp_ms",
                "sampleNum",
                "BioZ_I_calibrated_counts",
                "BioZ_Q_calibrated_counts",
                "Z_real_ohm",
                "Z_imag_ohm",
                "impedance_magnitude_ohm",
                "impedance_phase_deg",
            ]
        )
        for values in zip(
            data.time_s,
            data.timestamp_ms,
            data.sample_number,
            data.calibrated_i_counts,
            data.calibrated_q_counts,
            data.real_ohm,
            data.imag_ohm,
            data.magnitude_ohm,
            data.phase_deg,
        ):
            writer.writerow([f"{float(value):.12g}" for value in values])


def write_stable_csv(path: Path, periods: list[StablePeriod]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "start_s",
                "end_s",
                "duration_s",
                "mean_impedance_ohm",
                "median_impedance_ohm",
                "min_impedance_ohm",
                "max_impedance_ohm",
                "range_ohm",
                "threshold_ohm",
                "window_s",
                "sample_count",
            ]
        )
        for period in periods:
            writer.writerow(
                [
                    f"{period.start_s:.12g}",
                    f"{period.end_s:.12g}",
                    f"{period.duration_s:.12g}",
                    f"{period.mean_ohm:.12g}",
                    f"{period.median_ohm:.12g}",
                    f"{period.min_ohm:.12g}",
                    f"{period.max_ohm:.12g}",
                    f"{period.range_ohm:.12g}",
                    f"{period.threshold_ohm:.12g}",
                    f"{period.window_s:.12g}",
                    period.sample_count,
                ]
            )


def write_settings_json(
    path: Path,
    *,
    measurement_path: Path,
    calibrated_path: Path,
    identifier: str,
    settings: DecodedSettings,
    calibration_check: dict[str, Any],
    warnings: list[str],
    time_source: str,
) -> None:
    payload: dict[str, Any] = {
        "recording_identifier": identifier,
        "measurement_csv": measurement_path.name,
        "calibrated_csv": calibrated_path.name,
        "datasheet_url": DATASHEET_URL,
        "time_axis_source": time_source,
        "decoded_settings": asdict(settings),
        "calibrated_column_check": calibration_check,
        "warnings": warnings,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def set_global_axis_limits(ax: Any, values: np.ndarray, requested: Optional[tuple[float, float]]) -> None:
    if requested is not None:
        ax.set_ylim(*requested)
        return
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if lower == upper:
        padding = max(1.0, abs(lower) * 0.05)
        lower -= padding
        upper += padding
    ax.set_ylim(lower, upper)


def create_interactive_figure(
    data: CalculatedData,
    periods: list[StablePeriod],
    *,
    identifier: str,
    threshold_ohm: float,
    window_s: float,
    min_duration_s: float,
    merge_gap_s: float = 0.2,
    mag_ylim: Optional[tuple[float, float]] = None,
    phase_ylim: Optional[tuple[float, float]] = None,
) -> tuple[Any, Any]:
    """Build the interactive magnitude/phase plot and its controls."""

    fig, (ax_mag, ax_phase) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.36, hspace=0.12)

    magnitude_line, = ax_mag.plot(
        data.time_s,
        data.magnitude_ohm,
        color="#225ea8",
        linewidth=0.9,
        label="Impedance magnitude",
        picker=5,
    )
    phase_line, = ax_phase.plot(
        data.time_s,
        data.phase_deg,
        color="#238b45",
        linewidth=0.9,
        label="Impedance phase",
        picker=5,
    )
    del magnitude_line, phase_line

    ax_mag.set_ylabel("Impedance magnitude (Ω)")
    ax_phase.set_ylabel("Phase (°)")
    ax_phase.set_xlabel("Time from first logged sample (s)")
    ax_mag.set_title(f"MAX30009 BioZ: {identifier}")
    ax_mag.grid(True, alpha=0.3)
    ax_phase.grid(True, alpha=0.3)
    set_global_axis_limits(ax_mag, data.magnitude_ohm, mag_ylim)
    phase_display_ylim = phase_ylim if phase_ylim is not None else (-180.0, 180.0)
    set_global_axis_limits(ax_phase, data.phase_deg, phase_display_ylim)
    if phase_ylim is None:
        ax_phase.set_yticks([-180.0, -90.0, 0.0, 90.0, 180.0])
    ax_mag.set_xlim(float(data.time_s[0]), float(data.time_s[-1]))

    stable_artists: list[Any] = []
    status_text = fig.text(0.09, 0.305, "", fontsize=9, va="center")
    click_text = fig.text(
        0.09,
        0.005,
        "Click a plot near a sample to display its numerical values.",
        fontsize=9,
        va="bottom",
    )
    clicked_lines = [
        ax_mag.axvline(data.time_s[0], color="#636363", linestyle="--", visible=False),
        ax_phase.axvline(data.time_s[0], color="#636363", linestyle="--", visible=False),
    ]

    def redraw_stable(new_threshold: float) -> None:
        nonlocal stable_artists, periods
        for artist in stable_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        stable_artists = []
        periods = detect_stable_periods(
            data.time_s,
            data.magnitude_ohm,
            threshold_ohm=float(new_threshold),
            window_s=window_s,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
        )
        for index, period in enumerate(periods):
            (artist,) = ax_mag.plot(
                [period.start_s, period.end_s],
                [period.mean_ohm, period.mean_ohm],
                color="#d7301f",
                linewidth=2.2,
                solid_capstyle="round",
                label="Detected stable period" if index == 0 else "_nolegend_",
                picker=5,
            )
            stable_artists.append(artist)
        status_text.set_text(
            f"Stable periods: {len(periods)}  |  criterion: rolling {window_s:g}s "
            f"range ≤ {2.0 * float(new_threshold):g}Ω (±{float(new_threshold):g}Ω)  |  "
            f"minimum duration: {min_duration_s:g}s"
        )
        ax_mag.legend(loc="upper right")
        fig.canvas.draw_idle()

    time_start = float(data.time_s[0])
    time_end = float(data.time_s[-1])
    data_span = max(time_end - time_start, 1e-9)

    # Keep the initial time-window length useful for a typical recording while
    # allowing the user to replace it with any positive value in the display.
    initial_window_s = min(5.0, data_span)

    slider_max = max(
        1000.0,
        threshold_ohm * 4.0,
        float(np.nanpercentile(data.magnitude_ohm, 95)),
    )
    threshold_axis = fig.add_axes([0.17, 0.255, 0.58, 0.035])
    threshold_step_values = np.arange(10.0, slider_max + 0.01, 10.0)
    threshold_slider = Slider(
        threshold_axis,
        "Smooth ±Ω",
        0.1,
        slider_max,
        valinit=threshold_ohm,
        valstep=threshold_step_values,
        valfmt="%0.0f",
    )
    threshold_slider.valtext.set_visible(False)
    threshold_value_label = fig.text(0.765, 0.272, "Value (Ω)", fontsize=8, va="center")
    threshold_value_axis = fig.add_axes([0.825, 0.245, 0.08, 0.045])
    threshold_value_text = TextBox(
        threshold_value_axis,
        "",
        initial=f"{threshold_ohm:g}",
    )
    threshold_value_text.text_disp.set_fontsize(8)

    threshold_update_guard = False

    def extend_threshold_slider_if_needed(value: float) -> None:
        if value <= threshold_slider.valmax:
            return
        new_max = math.ceil(value / 10.0) * 10.0
        threshold_slider.valmax = new_max
        threshold_slider.valstep = np.arange(10.0, new_max + 0.01, 10.0)
        threshold_slider.ax.set_xlim(threshold_slider.valmin, new_max)

    def update_threshold_from_slider(new_threshold: float) -> None:
        nonlocal threshold_update_guard
        if threshold_update_guard:
            return
        threshold_update_guard = True
        try:
            threshold_value_text.set_val(f"{float(new_threshold):g}")
        finally:
            threshold_update_guard = False
        redraw_stable(float(new_threshold))

    def apply_threshold_input(value_text: str) -> None:
        nonlocal threshold_update_guard
        try:
            requested = float(value_text)
        except (TypeError, ValueError):
            threshold_value_axis.set_facecolor("#ffe5e5")
            return
        if not np.isfinite(requested) or requested < threshold_slider.valmin:
            threshold_value_axis.set_facecolor("#ffe5e5")
            return

        threshold_value_axis.set_facecolor("#f0f0f0")
        extend_threshold_slider_if_needed(requested)
        threshold_update_guard = True
        try:
            threshold_slider.set_val(requested)
        finally:
            threshold_update_guard = False
        redraw_stable(requested)

    threshold_slider.on_changed(update_threshold_from_slider)
    threshold_value_text.on_submit(apply_threshold_input)
    redraw_stable(threshold_ohm)

    # The old RangeSlider was replaced by a checkbox plus a window-length
    # input and a continuous start-position slider.  When the checkbox is off
    # the full recording is shown, preserving the original default behavior.
    custom_time_axis = fig.add_axes([0.02, 0.185, 0.16, 0.055])
    custom_time_check = CheckButtons(custom_time_axis, ["Custom Time Window"], [False])
    custom_time_check.labels[0].set_fontsize(8)

    window_label = fig.text(0.19, 0.211, "Length (s)", fontsize=8, va="center")
    window_text_axis = fig.add_axes([0.245, 0.185, 0.105, 0.045])
    window_text = TextBox(
        window_text_axis,
        "",
        initial=f"{initial_window_s:g}",
    )
    window_text.text_disp.set_fontsize(8)
    window_label.set_visible(False)
    window_text_axis.set_visible(False)

    scroll_axis = fig.add_axes([0.50, 0.185, 0.43, 0.035])
    scroll_slider = Slider(
        scroll_axis,
        "Window start (s)",
        time_start,
        time_start + data_span,
        valinit=time_start,
        valfmt="%0.2f",
        valstep=None,
    )
    scroll_axis.set_visible(False)

    time_status = fig.text(
        0.50,
        0.228,
        "Custom time window off: showing the full recording",
        fontsize=8,
        va="center",
    )

    time_update_guard = False
    custom_window_s = initial_window_s

    def set_time_status(message: str) -> None:
        time_status.set_text(message)

    def apply_custom_time_window(window_value: str | float) -> bool:
        """Apply a user-entered custom window and return whether it is valid."""

        nonlocal custom_window_s, time_update_guard
        try:
            requested = float(window_value)
        except (TypeError, ValueError):
            window_text_axis.set_facecolor("#ffe5e5")
            set_time_status("Custom time window must be a positive number of seconds")
            return False
        if not np.isfinite(requested) or requested <= 0:
            window_text_axis.set_facecolor("#ffe5e5")
            set_time_status("Custom time window must be a positive number of seconds")
            return False

        window_text_axis.set_facecolor("#f0f0f0")
        custom_window_s = min(requested, data_span)
        max_start = max(time_start, time_end - custom_window_s)

        # Slider.valmax and the slider axis limits are intentionally updated
        # together.  valstep=None keeps the start position continuous.
        scroll_slider.valmax = max_start if max_start > time_start else time_start + 1e-9
        scroll_slider.ax.set_xlim(time_start, scroll_slider.valmax)
        current_start = min(max(float(scroll_slider.val), time_start), max_start)
        time_update_guard = True
        try:
            scroll_slider.set_val(current_start)
        finally:
            time_update_guard = False
        ax_mag.set_xlim(current_start, current_start + custom_window_s)
        set_time_status(
            f"Custom time window: {custom_window_s:g}s | "
            f"showing {current_start:.2f}–{current_start + custom_window_s:.2f}s"
        )
        fig.canvas.draw_idle()
        return True

    def update_scroll(start_value: float) -> None:
        if time_update_guard or not custom_time_check.get_status()[0]:
            return
        max_start = max(time_start, time_end - custom_window_s)
        current_start = min(max(float(start_value), time_start), max_start)
        ax_mag.set_xlim(current_start, current_start + custom_window_s)
        set_time_status(
            f"Custom time window: {custom_window_s:g}s | "
            f"showing {current_start:.2f}–{current_start + custom_window_s:.2f}s"
        )
        fig.canvas.draw_idle()

    def toggle_custom_time(_label: str) -> None:
        active = custom_time_check.get_status()[0]
        window_label.set_visible(active)
        window_text_axis.set_visible(active)
        scroll_axis.set_visible(active)
        if active:
            apply_custom_time_window(window_text.text)
        else:
            ax_mag.set_xlim(time_start, time_end)
            set_time_status("Custom time window off: showing the full recording")
            fig.canvas.draw_idle()

    window_text.on_submit(apply_custom_time_window)
    scroll_slider.on_changed(update_scroll)
    custom_time_check.on_clicked(toggle_custom_time)

    # Magnitude limits are display-only controls.  They never modify the
    # calculated samples or the decoded register settings.
    finite_magnitude = data.magnitude_ohm[np.isfinite(data.magnitude_ohm)]
    if len(finite_magnitude):
        global_mag_lower = float(np.min(finite_magnitude))
        global_mag_upper = float(np.max(finite_magnitude))
    else:
        global_mag_lower, global_mag_upper = 0.0, 1.0
    initial_upper_active = mag_ylim is not None
    initial_lower_active = mag_ylim is not None
    initial_lower = mag_ylim[0] if mag_ylim is not None else global_mag_lower
    initial_upper = mag_ylim[1] if mag_ylim is not None else global_mag_upper

    limit_axis = fig.add_axes([0.02, 0.035, 0.235, 0.125])
    limit_check = CheckButtons(
        limit_axis,
        [
            "Limit Upper (Impedance Magnitude)",
            "Limit Lower (Impedance Magnitude)",
        ],
        [initial_upper_active, initial_lower_active],
    )
    for label in limit_check.labels:
        label.set_fontsize(7.5)

    upper_limit_label = fig.text(0.27, 0.125, "Upper (Ω)", fontsize=8, va="center")
    upper_limit_axis = fig.add_axes([0.32, 0.10, 0.10, 0.045])
    upper_limit_text = TextBox(upper_limit_axis, "", initial=f"{initial_upper:g}")
    upper_limit_text.text_disp.set_fontsize(8)
    lower_limit_label = fig.text(0.27, 0.065, "Lower (Ω)", fontsize=8, va="center")
    lower_limit_axis = fig.add_axes([0.32, 0.04, 0.10, 0.045])
    lower_limit_text = TextBox(lower_limit_axis, "", initial=f"{initial_lower:g}")
    lower_limit_text.text_disp.set_fontsize(8)
    upper_limit_label.set_visible(initial_upper_active)
    upper_limit_axis.set_visible(initial_upper_active)
    lower_limit_label.set_visible(initial_lower_active)
    lower_limit_axis.set_visible(initial_lower_active)

    limit_update_guard = False

    def apply_magnitude_limits(_text: str | None = None) -> None:
        nonlocal limit_update_guard
        if limit_update_guard:
            return
        upper_active, lower_active = limit_check.get_status()
        lower_value = global_mag_lower
        upper_value = global_mag_upper
        valid = True
        if lower_active:
            try:
                lower_value = float(lower_limit_text.text)
                valid &= bool(np.isfinite(lower_value))
            except (TypeError, ValueError):
                valid = False
        if upper_active:
            try:
                upper_value = float(upper_limit_text.text)
                valid &= bool(np.isfinite(upper_value))
            except (TypeError, ValueError):
                valid = False
        valid &= upper_value > lower_value
        upper_limit_axis.set_facecolor("#f0f0f0" if valid or not upper_active else "#ffe5e5")
        lower_limit_axis.set_facecolor("#f0f0f0" if valid or not lower_active else "#ffe5e5")
        if valid:
            ax_mag.set_ylim(lower_value, upper_value)
            fig.canvas.draw_idle()

    def toggle_magnitude_limit(_label: str) -> None:
        upper_active, lower_active = limit_check.get_status()
        upper_limit_label.set_visible(upper_active)
        upper_limit_axis.set_visible(upper_active)
        lower_limit_label.set_visible(lower_active)
        lower_limit_axis.set_visible(lower_active)
        apply_magnitude_limits()
        fig.canvas.draw_idle()

    upper_limit_text.on_submit(apply_magnitude_limits)
    lower_limit_text.on_submit(apply_magnitude_limits)
    limit_check.on_clicked(toggle_magnitude_limit)

    def nearest_sample(x_value: float) -> int:
        index = int(np.searchsorted(data.time_s, x_value))
        if index <= 0:
            return 0
        if index >= len(data.time_s):
            return len(data.time_s) - 1
        before = index - 1
        return before if abs(data.time_s[before] - x_value) <= abs(data.time_s[index] - x_value) else index

    def on_click(event: Any) -> None:
        if event.inaxes not in (ax_mag, ax_phase) or event.xdata is None:
            return
        index = nearest_sample(float(event.xdata))
        x_value = float(data.time_s[index])
        for line in clicked_lines:
            line.set_xdata([x_value, x_value])
            line.set_visible(True)
        click_text.set_text(
            f"t={x_value:.6f}s   |   |Z|={data.magnitude_ohm[index]:.6f}Ω   |   "
            f"phase={data.phase_deg[index]:.6f}°   |   "
            f"I={data.calibrated_i_counts[index]:.0f} counts   |   "
            f"Q={data.calibrated_q_counts[index]:.0f} counts"
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Matplotlib widget callbacks do not necessarily keep widget instances
    # alive.  Retaining them on the figure fixes the previously unresponsive
    # threshold slider and also keeps the new controls interactive.
    fig._max30009_widgets = {
        "threshold_slider": threshold_slider,
        "threshold_value_text": threshold_value_text,
        "custom_time_check": custom_time_check,
        "window_text": window_text,
        "scroll_slider": scroll_slider,
        "limit_check": limit_check,
        "upper_limit_text": upper_limit_text,
        "lower_limit_text": lower_limit_text,
    }
    return fig, periods


def format_hz(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:,.9g}"


def print_summary(
    *,
    measurement_path: Path,
    calibrated_path: Path,
    settings: DecodedSettings,
    data: CalculatedData,
    periods: list[StablePeriod],
    calibration_check: dict[str, Any],
    warnings: list[str],
    time_source: str,
) -> None:
    print(f"Measurement CSV : {measurement_path.name}")
    print(f"Calibrated CSV  : {calibrated_path.name}")
    print(f"Aligned samples : {len(data.time_s)}")
    print(f"Time span       : {data.time_s[0]:.6f} to {data.time_s[-1]:.6f} s")
    timestamp_diffs = np.diff(data.timestamp_ms)
    positive_diffs = timestamp_diffs[timestamp_diffs > 0]
    if len(positive_diffs):
        print(
            f"Timestamp rate : {1000.0 / float(np.median(positive_diffs)):.6f} samples/s "
            f"(median logged interval {float(np.median(positive_diffs)):.9g} ms)"
        )
    print(f"Time-axis source: {time_source}")
    print(f"Decoded stimulus frequency : {format_hz(settings.stimulus_frequency_hz)} Hz")
    print(f"Decoded BioZ sample rate   : {format_hz(settings.sample_rate_hz)} samples/s")
    print(f"Drive mode / gain          : {settings.drive_mode} / {settings.gain_vv:g} V/V")
    print(f"Stimulus current           : {format_hz(None if settings.stimulus_current_peak_a is None else settings.stimulus_current_peak_a * 1e6)} µA peak")
    print(f"Decoded ohm/count          : {format_hz(settings.ohm_per_count)} Ω/count")
    if settings.exported_ohm_per_count is not None and settings.ohm_per_count is not None:
        difference = settings.ohm_per_count - settings.exported_ohm_per_count
        print(
            f"CSV ohm/count              : {settings.exported_ohm_per_count:.15g} Ω/count "
            f"(decoder difference {difference:+.3g})"
        )
    if calibration_check.get("available"):
        print(
            "Calibration-column check   : "
            f"rounded match I={calibration_check['rounded_matches_i']}/"
            f"{calibration_check['sample_count']}, Q={calibration_check['rounded_matches_q']}/"
            f"{calibration_check['sample_count']}"
        )
    else:
        print(f"Calibration-column check   : unavailable ({calibration_check.get('reason')})")
    print(f"Stable periods (initial threshold): {len(periods)}")
    for index, period in enumerate(periods, start=1):
        print(
            f"  {index}: {period.start_s:.3f}–{period.end_s:.3f} s, "
            f"mean {period.mean_ohm:.3f} Ω, range {period.range_ohm:.3f} Ω"
        )
    for warning in warnings + settings.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def pair_limits(values: list[str]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("provide two values: LOWER UPPER")
    lower, upper = map(float, values)
    if upper <= lower:
        raise argparse.ArgumentTypeError("upper limit must be greater than lower limit")
    return lower, upper


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode and visualise a MAX30009 BioZ CSV recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="one CSV from the recording, or a directory containing the pair",
    )
    parser.add_argument(
        "companion",
        type=Path,
        nargs="?",
        default=None,
        help="optional corresponding second CSV; bypasses automatic pair discovery",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory for calculated CSV, stable-period CSV, and settings JSON",
    )
    parser.add_argument(
        "--smooth-threshold",
        type=positive_float,
        default=100.0,
        help="initial ±ohm threshold for a stable magnitude period",
    )
    parser.add_argument(
        "--stability-window",
        type=positive_float,
        default=1.0,
        help="rolling window length used by stable-period detection, in seconds",
    )
    parser.add_argument(
        "--min-stable-duration",
        type=float,
        default=0.5,
        help="minimum duration for a reported stable period, in seconds",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.2,
        help="merge stable sections separated by no more than this gap, in seconds",
    )
    parser.add_argument(
        "--mag-ylim",
        nargs=2,
        metavar=("LOWER", "UPPER"),
        type=float,
        default=None,
        help="fixed magnitude-axis limits; omitted means global data limits",
    )
    parser.add_argument(
        "--phase-ylim",
        nargs=2,
        metavar=("LOWER", "UPPER"),
        type=float,
        default=None,
        help="fixed phase-axis limits; omitted means global data limits",
    )
    parser.add_argument(
        "--vref",
        type=positive_float,
        default=1.0,
        help="VREF used in the datasheet ADC-to-impedance equation, in volts",
    )
    parser.add_argument(
        "--external-ref-hz",
        type=positive_float,
        default=None,
        help="external PLL reference frequency, required when REF_CLK_SEL=1",
    )
    parser.add_argument(
        "--rext-ohm",
        type=positive_float,
        default=None,
        help="external drive resistor value, required when BIOZ_EXT_RES=1",
    )
    parser.add_argument(
        "--time-source",
        choices=("timestamp", "sample-rate"),
        default="timestamp",
        help="time axis from logged epoch-millisecond timestamps or decoded sample rate",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="save the initial plot view as a PNG in --output-dir",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open the interactive Matplotlib window",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        for name, limits in (("--mag-ylim", args.mag_ylim), ("--phase-ylim", args.phase_ylim)):
            if limits is not None and limits[1] <= limits[0]:
                raise BioZViewerError(f"{name} requires UPPER to be greater than LOWER")
        measurement_path, calibrated_path, identifier = find_recording_pair(
            args.input,
            args.companion,
        )
        registers, register_warnings = parse_register_snapshot(measurement_path)
        calibrated = read_calibrated_log(calibrated_path)
        exported_scale = calibrated.metadata.get("Ohm/Count")
        settings = decode_settings(
            registers,
            vref_v=args.vref,
            external_ref_hz=args.external_ref_hz,
            rext_ohm=args.rext_ohm,
            exported_ohm_per_count=exported_scale,
            snapshot_warnings=register_warnings,
        )
        measurement = read_measurement_log(measurement_path)

        calibration_check: dict[str, Any]
        if calibrated.raw_i_counts is not None and calibrated.raw_q_counts is not None:
            n_check = min(
                len(calibrated.raw_i_counts),
                len(calibrated.raw_q_counts),
                len(calibrated.calibrated_i_counts),
                len(calibrated.calibrated_q_counts),
            )
            calibration_check = reproduce_calibrated_columns(
                calibrated.raw_i_counts[:n_check],
                calibrated.raw_q_counts[:n_check],
                calibrated.calibrated_i_counts[:n_check],
                calibrated.calibrated_q_counts[:n_check],
                calibrated.metadata,
            )
        else:
            calibration_check = {
                "available": False,
                "reason": "raw I/Q columns were not available in the calibrated CSV",
            }

        data, calculation_warnings = calculate_impedance(
            measurement,
            calibrated,
            settings,
            time_source=args.time_source,
        )
        periods = detect_stable_periods(
            data.time_s,
            data.magnitude_ohm,
            threshold_ohm=args.smooth_threshold,
            window_s=args.stability_window,
            min_duration_s=args.min_stable_duration,
            merge_gap_s=args.merge_gap,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_base = args.output_dir / f"MAX30009_{identifier}"
        calculated_path = output_base.with_name(output_base.name + "_calculated.csv")
        stable_path = output_base.with_name(output_base.name + "_stable_periods.csv")
        settings_path = output_base.with_name(output_base.name + "_decoded_settings.json")
        write_calculated_csv(calculated_path, data)
        write_stable_csv(stable_path, periods)
        all_warnings = calculation_warnings
        write_settings_json(
            settings_path,
            measurement_path=measurement_path,
            calibrated_path=calibrated_path,
            identifier=identifier,
            settings=settings,
            calibration_check=calibration_check,
            warnings=all_warnings,
            time_source=args.time_source,
        )

        print_summary(
            measurement_path=measurement_path,
            calibrated_path=calibrated_path,
            settings=settings,
            data=data,
            periods=periods,
            calibration_check=calibration_check,
            warnings=all_warnings,
            time_source=args.time_source,
        )
        print(f"Wrote calculated samples: {calculated_path}")
        print(f"Wrote stable periods    : {stable_path}")
        print(f"Wrote decoded settings  : {settings_path}")

        fig, _ = create_interactive_figure(
            data,
            periods,
            identifier=identifier,
            threshold_ohm=args.smooth_threshold,
            window_s=args.stability_window,
            min_duration_s=args.min_stable_duration,
            merge_gap_s=args.merge_gap,
            mag_ylim=tuple(args.mag_ylim) if args.mag_ylim else None,
            phase_ylim=tuple(args.phase_ylim) if args.phase_ylim else None,
        )
        if args.save_plot:
            plot_path = output_base.with_name(output_base.name + "_plot.png")
            fig.savefig(plot_path, dpi=160)
            print(f"Wrote initial plot       : {plot_path}")
        if not args.no_show:
            plt.show()
        else:
            plt.close(fig)
        return 0
    except BioZViewerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - protects users from opaque GUI failures
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
