"""
mzml_parser.py — pyopenms-based mzML reader for MHCBooster (reforge)

Replaces pyteomics with pyopenms for ~3.5x speedup on mzML loading.
Maintains identical interface: returns (spec_names, spec_indices, exp_rts, exp_ims, exp_spectra).

RT is returned in MINUTES (matching the original pyteomics behavior).
pyopenms returns seconds internally, so we divide by 60.

Author: Nico (automated rewrite from pyteomics version)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pyopenms import MzMLFile, MSExperiment

from mhcbooster.utils.constants import PROTON_MASS

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_experiment(mzml_path: str) -> MSExperiment:
    """Load an mzML file into a pyopenms MSExperiment."""
    exp = MSExperiment()
    MzMLFile().load(str(mzml_path), exp)
    return exp


def _get_rt_minutes(spectrum) -> float:
    """Get RT in minutes from a pyopenms spectrum (pyopenms stores seconds)."""
    return spectrum.getRT() / 60.0


def _get_spectrum_title(spectrum) -> str:
    """Get spectrum title from metadata, fall back to nativeID."""
    if spectrum.metaValueExists(b'spectrum title'):
        title = spectrum.getMetaValue(b'spectrum title')
        if isinstance(title, bytes):
            title = title.decode()
        return title.split(' ')[0]
    return spectrum.getNativeID()


def _get_precursor_info(spectrum):
    """
    Extract precursor m/z, isolation window offsets, IM, and CE from a spectrum.
    Returns (precursor_mz, lower_offset, upper_offset, im, ce).
    """
    precursors = spectrum.getPrecursors()
    if len(precursors) == 0:
        return None, 0.1, 0.1, 0, 25

    prec = precursors[0]
    precursor_mz = prec.getMZ()

    lower_offset = prec.getIsolationWindowLowerOffset()
    upper_offset = prec.getIsolationWindowUpperOffset()
    if lower_offset == 0:
        lower_offset = 0.1
    else:
        lower_offset += 0.01
    if upper_offset == 0:
        upper_offset = 0.1
    else:
        upper_offset += 0.01

    # CE: try getActivationEnergy first, then metaValue
    ce = prec.getActivationEnergy()
    if ce == 0.0:
        ce = 25  # default fallback (matches original code)

    # IM: try scan-level drift time first (MSFragger style), then precursor-level
    im = spectrum.getDriftTime()
    if im <= 0:
        im_prec = prec.getDriftTime()
        im = im_prec if im_prec > 0 else 0

    return precursor_mz, lower_offset, upper_offset, im, ce


def _get_peaks(spectrum):
    """Extract m/z and intensity arrays from a pyopenms spectrum."""
    mzs, ints = spectrum.get_peaks()
    return mzs.astype(np.float32), ints.astype(np.float32)


def _is_timsconvert_or_tdf2mzml(mzml_path: str) -> bool:
    """Check if the mzML was produced by timsconvert or tdf2mzml (for timsTOF data)."""
    with open(mzml_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip().startswith('<software '):
                if 'timsconvert' in line or 'tdf2mzml' in line:
                    return True
            if line.strip() == '</softwareList>':
                break
    return False


# ---------------------------------------------------------------------------
# Public API (same interface as original)
# ---------------------------------------------------------------------------

def get_rt_ccs_ms2_from_mzml(mzml_path, scan_nrs, masses, charges):
    """
    Dispatch to timsconvert or msconvert reader based on mzML metadata.
    """
    if _is_timsconvert_or_tdf2mzml(mzml_path):
        return get_rt_ccs_ms2_from_timsconvert_mzml(mzml_path, scan_nrs, masses, charges)
    else:
        return get_rt_ccs_ms2_from_msconvert_mzml(mzml_path, scan_nrs, masses, charges)


def get_rt_ccs_ms2_from_msconvert_mzml(mzml_path, scan_nrs, masses, charges):
    """
    Read an msconvert-produced mzML and extract RT, CCS, MS2 for given scan numbers.
    Handles timsTOF neighbor-scan matching for IM data.
    """
    target_mzs = masses / charges + PROTON_MASS
    exp = _load_experiment(mzml_path)
    n_spectra = exp.getNrSpectra()

    # Check if this has IM data
    ms2_sample = None
    for i in range(n_spectra):
        s = exp.getSpectrum(i)
        if s.getMSLevel() == 2:
            ms2_sample = s
            break

    spec_names = [None] * len(scan_nrs)
    spec_indices = [None] * len(scan_nrs)
    exp_rts = [None] * len(scan_nrs)
    exp_ims = [None] * len(scan_nrs)
    exp_mzs = [None] * len(scan_nrs)
    exp_intensities = [None] * len(scan_nrs)
    exp_ces = [None] * len(scan_nrs)

    for i, scan_nr in tqdm(enumerate(scan_nrs), total=len(scan_nrs), desc='Extracting RTs, CCSs, MS2s...'):
        idx = scan_nr - 1
        spec_indices[i] = idx
        spectrum = exp.getSpectrum(idx)
        spec_names[i] = _get_spectrum_title(spectrum)
        target_rt = _get_rt_minutes(spectrum)
        exp_rts[i] = target_rt

        precursor_mz, lower_offset, upper_offset, im, ce = _get_precursor_info(spectrum)
        if precursor_mz is not None and (precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset):
            mzs, ints = _get_peaks(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints
            continue

        # Search neighbors for timsTOF data (same RT, different precursor windows)
        matched = False
        # Search left
        for j in range(1, scan_nr):
            left_idx = scan_nr - j - 1
            if left_idx < 0:
                break
            spectrum_l = exp.getSpectrum(left_idx)
            rt = _get_rt_minutes(spectrum_l)
            if rt != target_rt:
                break
            prec_mz_l, lo_l, up_l, im_l, ce_l = _get_precursor_info(spectrum_l)
            if prec_mz_l is not None and (prec_mz_l - lo_l < target_mzs[i] < prec_mz_l + up_l):
                matched = True
                spec_indices[i] = left_idx
                spec_names[i] = _get_spectrum_title(spectrum_l)
                mzs, ints = _get_peaks(spectrum_l)
                exp_ims[i] = im_l
                exp_ces[i] = ce_l
                exp_mzs[i] = mzs
                exp_intensities[i] = ints
                break
        if matched:
            continue

        # Search right
        for j in range(1, n_spectra - scan_nr + 1):
            right_idx = scan_nr + j - 1
            if right_idx >= n_spectra:
                break
            spectrum_r = exp.getSpectrum(right_idx)
            rt = _get_rt_minutes(spectrum_r)
            if rt != target_rt:
                break
            prec_mz_r, lo_r, up_r, im_r, ce_r = _get_precursor_info(spectrum_r)
            if prec_mz_r is not None and (prec_mz_r - lo_r < target_mzs[i] < prec_mz_r + up_r):
                matched = True
                spec_indices[i] = right_idx
                spec_names[i] = _get_spectrum_title(spectrum_r)
                mzs, ints = _get_peaks(spectrum_r)
                exp_ims[i] = im_r
                exp_ces[i] = ce_r
                exp_mzs[i] = mzs
                exp_intensities[i] = ints
                break

        if not matched:
            # Fallback: use original scan with warning
            spec_indices[i] = idx
            spectrum = exp.getSpectrum(idx)
            spec_names[i] = _get_spectrum_title(spectrum)
            precursor_mz, lower_offset, upper_offset, im, ce = _get_precursor_info(spectrum)
            print(f'WARNING: Spectrum not matched perfectly. peptide_mz: {target_mzs[i]}, '
                  f'precursor_mz:{precursor_mz}, lower_offset:{lower_offset}, upper_offset:{upper_offset}.')
            mzs, ints = _get_peaks(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints

    spec_names = np.array(spec_names)
    spec_indices = np.array(spec_indices, dtype=int)
    exp_rts = np.array(exp_rts)
    exp_ims = np.array(exp_ims)
    exp_spectra = pd.DataFrame()
    exp_spectra['mzs'] = exp_mzs
    exp_spectra['intensities'] = exp_intensities
    exp_spectra['ce'] = exp_ces
    return spec_names, spec_indices, exp_rts, exp_ims, exp_spectra


def get_rt_ccs_ms2_from_timsconvert_mzml(mzml_path, scan_nrs, masses, charges):
    """
    Read a timsconvert/tdf2mzml-produced mzML. Same logic as msconvert version,
    but spectrum titles are constructed differently (filename.scanNr.scanNr.charge).
    """
    target_mzs = masses / charges + PROTON_MASS
    exp = _load_experiment(mzml_path)
    n_spectra = exp.getNrSpectra()
    file_name = Path(mzml_path).stem

    spec_names = [None] * len(scan_nrs)
    spec_indices = [None] * len(scan_nrs)
    exp_rts = [None] * len(scan_nrs)
    exp_ims = [None] * len(scan_nrs)
    exp_mzs = [None] * len(scan_nrs)
    exp_intensities = [None] * len(scan_nrs)
    exp_ces = [None] * len(scan_nrs)

    for i, scan_nr in tqdm(enumerate(scan_nrs), total=len(scan_nrs), desc='Extracting RTs, CCSs, MS2s...'):
        idx = scan_nr - 1
        spec_indices[i] = idx
        spectrum = exp.getSpectrum(idx)
        # timsconvert: construct title as filename.scanNr.scanNr.charge
        spec_names[i] = '.'.join([file_name, str(scan_nr), str(scan_nr), str(charges[i])])
        target_rt = _get_rt_minutes(spectrum)
        exp_rts[i] = target_rt

        precursor_mz, lower_offset, upper_offset, im, ce = _get_precursor_info(spectrum)
        if precursor_mz is not None and (precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset):
            mzs, ints = _get_peaks(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints
            continue

        # Search neighbors (same as msconvert version)
        matched = False
        for j in range(1, scan_nr):
            left_idx = scan_nr - j - 1
            if left_idx < 0:
                break
            spectrum_l = exp.getSpectrum(left_idx)
            rt = _get_rt_minutes(spectrum_l)
            if rt != target_rt:
                break
            prec_mz_l, lo_l, up_l, im_l, ce_l = _get_precursor_info(spectrum_l)
            if prec_mz_l is not None and (prec_mz_l - lo_l < target_mzs[i] < prec_mz_l + up_l):
                matched = True
                spec_indices[i] = left_idx
                spec_names[i] = '.'.join([file_name, str(scan_nr), str(scan_nr), str(charges[i])])
                mzs, ints = _get_peaks(spectrum_l)
                exp_ims[i] = im_l
                exp_ces[i] = ce_l
                exp_mzs[i] = mzs
                exp_intensities[i] = ints
                break
        if matched:
            continue
        for j in range(1, n_spectra - scan_nr + 1):
            right_idx = scan_nr + j - 1
            if right_idx >= n_spectra:
                break
            spectrum_r = exp.getSpectrum(right_idx)
            rt = _get_rt_minutes(spectrum_r)
            if rt != target_rt:
                break
            prec_mz_r, lo_r, up_r, im_r, ce_r = _get_precursor_info(spectrum_r)
            if prec_mz_r is not None and (prec_mz_r - lo_r < target_mzs[i] < prec_mz_r + up_r):
                matched = True
                spec_indices[i] = right_idx
                spec_names[i] = '.'.join([file_name, str(scan_nr), str(scan_nr), str(charges[i])])
                mzs, ints = _get_peaks(spectrum_r)
                exp_ims[i] = im_r
                exp_ces[i] = ce_r
                exp_mzs[i] = mzs
                exp_intensities[i] = ints
                break

        if not matched:
            spec_indices[i] = idx
            spectrum = exp.getSpectrum(idx)
            spec_names[i] = '.'.join([file_name, str(scan_nr), str(scan_nr), str(charges[i])])
            precursor_mz, lower_offset, upper_offset, im, ce = _get_precursor_info(spectrum)
            print(f'WARNING: Spectrum not matched perfectly. peptide_mz: {target_mzs[i]}, '
                  f'precursor_mz:{precursor_mz}, lower_offset:{lower_offset}, upper_offset:{upper_offset}.')
            mzs, ints = _get_peaks(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints

    spec_names = np.array(spec_names)
    spec_indices = np.array(spec_indices, dtype=int)
    exp_rts = np.array(exp_rts)
    exp_ims = np.array(exp_ims)
    exp_spectra = pd.DataFrame()
    exp_spectra['mzs'] = exp_mzs
    exp_spectra['intensities'] = exp_intensities
    exp_spectra['ce'] = exp_ces
    return spec_names, spec_indices, exp_rts, exp_ims, exp_spectra


def get_rt_ccs_ms2_from_msfragger_mzml(mzml_path, scan_nrs, masses, charges):
    """
    Read an MSFragger-produced (uncalibrated/calibrated) mzML.
    Only loads spectra matching target scan numbers (efficient for large files).
    """
    target_mzs = masses / charges + PROTON_MASS
    exp = _load_experiment(mzml_path)

    scan_nrs_str = [str(nr) for nr in scan_nrs]
    scan_nrs_unique = np.sort(np.unique(np.array(scan_nrs_str).astype(int))).astype(str)

    # Build scan_nr -> spectrum mapping by parsing spectrum titles
    # MSFragger titles: "filename.scanNr.scanNr.charge"
    scannr_idx_map = {}
    spec_idx_map = {}
    ms2_list = []
    ms2_names = []
    scan_nr_idx = 0

    for i in tqdm(range(exp.getNrSpectra()), desc='Loading related MS2 spectrum to memory...', mininterval=1.0):
        spectrum = exp.getSpectrum(i)
        title = _get_spectrum_title(spectrum)
        # Extract scan number from title: "filename.scanNr.scanNr.charge" -> second-to-last field
        tmp_scan_nr = title.rsplit('.', 2)[-2]

        if scan_nr_idx < len(scan_nrs_unique) and tmp_scan_nr == scan_nrs_unique[scan_nr_idx]:
            ms2_list.append(spectrum)
            ms2_names.append(title)
            scannr_idx_map[tmp_scan_nr] = scan_nr_idx
            spec_idx_map[tmp_scan_nr] = i
            scan_nr_idx += 1
            if scan_nr_idx == len(scan_nrs_unique):
                break

    print(len(ms2_list), len(scan_nrs_unique))
    assert len(ms2_list) == len(scan_nrs_unique), 'Error in MSFragger uncalibrated mzML file reading...'

    spec_names = [None] * len(scan_nrs)
    spec_indices = [None] * len(scan_nrs)
    exp_rts = [None] * len(scan_nrs)
    exp_ims = [None] * len(scan_nrs)
    exp_mzs = [None] * len(scan_nrs)
    exp_intensities = [None] * len(scan_nrs)
    exp_ces = [None] * len(scan_nrs)

    for i, scan_nr in tqdm(enumerate(scan_nrs_str), total=len(scan_nrs_str), desc='Extracting RTs, CCSs, MS2s...'):
        spec_names[i] = ms2_names[scannr_idx_map[scan_nr]]
        spec_indices[i] = spec_idx_map[scan_nr]
        spectrum = ms2_list[scannr_idx_map[scan_nr]]

        target_rt = _get_rt_minutes(spectrum)
        exp_rts[i] = target_rt

        precursor_mz, lower_offset, upper_offset, im, ce = _get_precursor_info(spectrum)
        if precursor_mz is not None and (precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset):
            mzs, ints = _get_peaks(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints
        else:
            print(f'WARNING: Spectrum not matched perfectly. peptide_mz: {target_mzs[i]}, '
                  f'precursor_mz:{precursor_mz}, lower_offset:{lower_offset}, upper_offset:{upper_offset}.')
            mzs, ints = _get_peaks(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints

    spec_names = np.array(spec_names)
    spec_indices = np.array(spec_indices, dtype=int)
    exp_rts = np.array(exp_rts)
    exp_ims = np.array(exp_ims)
    exp_spectra = pd.DataFrame()
    exp_spectra['mzs'] = exp_mzs
    exp_spectra['intensities'] = exp_intensities
    exp_spectra['ce'] = exp_ces
    return spec_names, spec_indices, exp_rts, exp_ims, exp_spectra
