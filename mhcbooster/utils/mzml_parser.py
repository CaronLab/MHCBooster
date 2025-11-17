
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm
from collections import deque
from pyteomics import mzml

from mhcbooster.utils.constants import PROTON_MASS

def _extract_rt(spectrum):
    rt = spectrum['scanList']['scan'][0]['scan start time']
    return rt

def _extract_mz(spectrum):
    precursor = spectrum['precursorList']['precursor'][0]
    precursor_mz = precursor['selectedIonList']['selectedIon'][0]['selected ion m/z']
    if 'isolationWindow' in precursor.keys():
        precursor_mz = precursor['isolationWindow']['isolation window target m/z'] #TODO
        lower_offset = precursor['isolationWindow']['isolation window lower offset'] + 0.01
        upper_offset = precursor['isolationWindow']['isolation window upper offset'] + 0.01
    else:
        lower_offset = 0.1
        upper_offset = 0.1
    return precursor_mz, lower_offset, upper_offset

def _extract_im_ms2(spectrum):
    mzs = spectrum['m/z array']
    ints = spectrum['intensity array']
    precursor = spectrum['precursorList']['precursor'][0]
    ce = 25
    if 'collision energy' in precursor['activation'].keys():
        ce = precursor['activation']['collision energy']
    im = 0
    if 'inverse reduced ion mobility' in precursor['selectedIonList']['selectedIon'][0].keys():
        im = precursor['selectedIonList']['selectedIon'][0]['inverse reduced ion mobility']
    return im, ce, mzs, ints

def _extract_im_ms2_msfragger(spectrum):
    mzs = spectrum['m/z array']
    ints = spectrum['intensity array']
    if 'collision energy' in spectrum['precursorList']['precursor'][0]['activation'].keys():
        ce = spectrum['precursorList']['precursor'][0]['activation']['collision energy']
    else:
        ce = 25
    im = 0
    if 'inverse reduced ion mobility' in spectrum['scanList']['scan'][0].keys():
        im = spectrum['scanList']['scan'][0]['inverse reduced ion mobility']
    return im, ce, mzs, ints


def get_rt_ccs_ms2_from_mzml(mzml_path, scan_nrs, masses, charges):
    timsconvert_mzml = False
    for line in open(mzml_path):
        if line.strip().startswith('<software '):
            if 'timsconvert' in line or 'tdf2mzml' in line:
                timsconvert_mzml = True
        if line.strip() == '</softwareList>':
            break
    if timsconvert_mzml:
        return get_rt_ccs_ms2_from_timsconvert_mzml(mzml_path, scan_nrs, masses, charges)
    else:
        return get_rt_ccs_ms2_from_msconvert_mzml(mzml_path, scan_nrs, masses, charges)


def get_rt_ccs_ms2_from_msconvert_mzml(mzml_path, scan_nrs, masses, charges):

    target_mzs = masses / charges + PROTON_MASS
    mzml_file = mzml.read(mzml_path)
    ms1_list = [data for data in tqdm(mzml_file, desc='Loading mzML spectra to memory...')]
    ms2_list = [data for data in ms1_list if data['ms level'] == 2]

    if len(ms2_list) > 0 and 'inverse reduced ion mobility' in ms2_list[0]['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0].keys():
        ms_list = ms2_list
    else:
        ms_list = ms1_list
    spec_names = [None] * len(scan_nrs)
    spec_indices = [None] * len(scan_nrs)
    exp_rts = [None] * len(scan_nrs)
    exp_ims = [None] * len(scan_nrs)
    exp_mzs = [None] * len(scan_nrs)
    exp_intensities = [None] * len(scan_nrs)
    exp_ces = [None] * len(scan_nrs)
    for i, scan_nr in tqdm(enumerate(scan_nrs), total=len(scan_nrs), desc='Extracting RTs, CCSs, MS2s...'):
        spec_indices[i] = scan_nr - 1
        spectrum = ms_list[scan_nr - 1]
        spec_names[i] = spectrum['spectrum title'].split(' ')[0]
        target_rt = _extract_rt(spectrum)
        exp_rts[i] = target_rt
        precursor_mz, lower_offset, upper_offset = _extract_mz(spectrum)
        if precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset:
            im, ce, mzs, ints = _extract_im_ms2(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints
            continue

        # Search neighbor for TimsTOF data
        matched = False
        for j in range(1, scan_nr):  # to left
            spec_indices[i] = scan_nr - j - 1
            spectrum = ms_list[scan_nr - j - 1]
            spec_names[i] = spectrum['spectrum title'].split(' ')[0]
            rt = _extract_rt(spectrum)
            if rt != target_rt:
                break
            precursor_mz, lower_offset, upper_offset = _extract_mz(spectrum)
            if precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset:
                matched = True
                im, ce, mzs, ints = _extract_im_ms2(spectrum)
                exp_ims[i] = im
                exp_ces[i] = ce
                exp_mzs[i] = mzs
                exp_intensities[i] = ints
                break
        if matched:
            continue
        for j in range(1, len(ms_list) - scan_nr + 1): # to right
            spec_indices[i] = scan_nr + j - 1
            spectrum = ms_list[scan_nr + j - 1]
            spec_names[i] = spectrum['spectrum title'].split(' ')[0]
            rt = spectrum['scanList']['scan'][0]['scan start time']
            if rt != target_rt:
                break
            precursor_mz, lower_offset, upper_offset = _extract_mz(spectrum)
            if precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset:
                matched = True
                im, ce, mzs, ints = _extract_im_ms2(spectrum)
                exp_ims[i] = im
                exp_ces[i] = ce
                exp_mzs[i] = mzs
                exp_intensities[i] = ints
                break
        if not matched:
            spec_indices[i] = scan_nr - 1
            spectrum = ms_list[scan_nr - 1]
            spec_names[i] = spectrum['spectrum title'].split(' ')[0]
            precursor_mz, lower_offset, upper_offset = _extract_mz(spectrum)
            print(f'WARNING: Spectrum not matched perfectly. peptide_mz: {target_mzs[i]}, precursor_mz:{precursor_mz}, lower_offset:{lower_offset}, upper_offset:{upper_offset}.')
            im, ce, mzs, ints = _extract_im_ms2(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints

    spec_names = np.array(spec_names)
    spec_indices = np.array(spec_indices).astype(int)
    exp_rts = np.array(exp_rts)
    exp_ims = np.array(exp_ims)
    exp_spectra = pd.DataFrame()
    exp_spectra['mzs'] = exp_mzs
    exp_spectra['intensities'] = exp_intensities
    exp_spectra['ce'] = exp_ces
    return spec_names, spec_indices, exp_rts, exp_ims, exp_spectra


def get_rt_ccs_ms2_from_timsconvert_mzml(mzml_path, scan_nrs, masses, charges):

    target_mzs = masses / charges + PROTON_MASS
    mzml_file = mzml.read(mzml_path)
    file_name = Path(mzml_path).stem
    ms1_list = [data for data in tqdm(mzml_file, desc='Loading mzML spectra to memory...')]
    # ms2_list = [data for data in ms1_list if data['ms level'] == 2]

    # if len(ms2_list) > 0 and 'inverse reduced ion mobility' in ms2_list[0]['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0].keys():
    #     ms_list = ms2_list
    # else:
    #     ms_list = ms1_list
    ms_list = ms1_list
    spec_names = [None] * len(scan_nrs)
    spec_indices = [None] * len(scan_nrs)
    exp_rts = [None] * len(scan_nrs)
    exp_ims = [None] * len(scan_nrs)
    exp_mzs = [None] * len(scan_nrs)
    exp_intensities = [None] * len(scan_nrs)
    exp_ces = [None] * len(scan_nrs)
    for i, scan_nr in tqdm(enumerate(scan_nrs), total=len(scan_nrs), desc='Extracting RTs, CCSs, MS2s...'):
        spec_indices[i] = scan_nr - 1
        spectrum = ms_list[scan_nr - 1]
        spec_names[i] = '.'.join([file_name, scan_nr, scan_nr, charges[i]])
        target_rt = _extract_rt(spectrum)
        exp_rts[i] = target_rt
        precursor_mz, lower_offset, upper_offset = _extract_mz(spectrum)
        if precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset:
            im, ce, mzs, ints = _extract_im_ms2(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints
            continue

        # Search neighbor for TimsTOF data
        matched = False
        for j in range(1, scan_nr):  # to left
            spec_indices[i] = scan_nr - j - 1
            spectrum = ms_list[scan_nr - j - 1]
            spec_names[i] = '.'.join([file_name, scan_nr, scan_nr, charges[i]])
            rt = _extract_rt(spectrum)
            if rt != target_rt:
                break
            precursor_mz, lower_offset, upper_offset = _extract_mz(spectrum)
            if precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset:
                matched = True
                im, ce, mzs, ints = _extract_im_ms2(spectrum)
                exp_ims[i] = im
                exp_ces[i] = ce
                exp_mzs[i] = mzs
                exp_intensities[i] = ints
                break
        if matched:
            continue
        for j in range(1, len(ms_list) - scan_nr + 1): # to right
            spec_indices[i] = scan_nr + j - 1
            spectrum = ms_list[scan_nr + j - 1]
            spec_names[i] = '.'.join([file_name, scan_nr, scan_nr, charges[i]])
            rt = spectrum['scanList']['scan'][0]['scan start time']
            if rt != target_rt:
                break
            precursor_mz, lower_offset, upper_offset = _extract_mz(spectrum)
            if precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset:
                matched = True
                im, ce, mzs, ints = _extract_im_ms2(spectrum)
                exp_ims[i] = im
                exp_ces[i] = ce
                exp_mzs[i] = mzs
                exp_intensities[i] = ints
                break
        if not matched:
            spec_indices[i] = scan_nr - 1
            spectrum = ms_list[scan_nr - 1]
            spec_names[i] = '.'.join([file_name, scan_nr, scan_nr, charges[i]])
            precursor_mz, lower_offset, upper_offset = _extract_mz(spectrum)
            print(f'WARNING: Spectrum not matched perfectly. peptide_mz: {target_mzs[i]}, precursor_mz:{precursor_mz}, lower_offset:{lower_offset}, upper_offset:{upper_offset}.')
            im, ce, mzs, ints = _extract_im_ms2(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints

    spec_names = np.array(spec_names)
    spec_indices = np.array(spec_indices).astype(int)
    exp_rts = np.array(exp_rts)
    exp_ims = np.array(exp_ims)
    exp_spectra = pd.DataFrame()
    exp_spectra['mzs'] = exp_mzs
    exp_spectra['intensities'] = exp_intensities
    exp_spectra['ce'] = exp_ces
    return spec_names, spec_indices, exp_rts, exp_ims, exp_spectra


def get_rt_ccs_ms2_from_msfragger_mzml(mzml_path, scan_nrs, masses, charges):

    target_mzs = masses / charges + PROTON_MASS
    mzml_file = mzml.read(mzml_path)
    scan_nrs = [str(nr) for nr in scan_nrs]
    scan_nrs_unique = np.sort(np.unique(np.array(scan_nrs).astype(int))).astype(str)
    scannr_idx_map = {}
    spec_idx_map = {}
    ms2_list, ms2_names = deque(), deque()
    scan_nr_idx = 0
    for i, data in enumerate(tqdm(mzml_file, desc='Loading related MS2 spectrum to memory...')):
        tmp_scan_nr = data['spectrum title'].rsplit('.', 2)[-2]
        if tmp_scan_nr == scan_nrs_unique[scan_nr_idx]:
            ms2_list.append(data)
            ms2_names.append(data['spectrum title'])
            scannr_idx_map[tmp_scan_nr] = scan_nr_idx
            spec_idx_map[tmp_scan_nr] = i
            scan_nr_idx += 1
            if scan_nr_idx == len(scan_nrs_unique):
                break
    ms2_list = list(ms2_list)
    print(len(ms2_list), len(scan_nrs_unique))
    assert len(ms2_list) == len(scan_nrs_unique), 'Error in MSFragger uncalibrated mzML file reading...'
    spec_names = [None] * len(scan_nrs)
    spec_indices = [None] * len(scan_nrs)
    exp_rts = [None] * len(scan_nrs)
    exp_ims = [None] * len(scan_nrs)
    exp_mzs = [None] * len(scan_nrs)
    exp_intensities = [None] * len(scan_nrs)
    exp_ces = [None] * len(scan_nrs)
    for i, scan_nr in tqdm(enumerate(scan_nrs), total=len(scan_nrs), desc='Extracting RTs, CCSs, MS2s...'):
        spec_names[i] = ms2_names[scannr_idx_map[scan_nr]]
        spec_indices[i] = spec_idx_map[scan_nr]
        spectrum = ms2_list[scannr_idx_map[scan_nr]]
        target_rt = _extract_rt(spectrum)
        exp_rts[i] = target_rt
        precursor_mz, lower_offset, upper_offset = _extract_mz(spectrum)
        if precursor_mz - lower_offset < target_mzs[i] < precursor_mz + upper_offset:
            im, ce, mzs, ints = _extract_im_ms2_msfragger(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints
        else:
            print(f'WARNING: Spectrum not matched perfectly. peptide_mz: {target_mzs[i]}, precursor_mz:{precursor_mz}, lower_offset:{lower_offset}, upper_offset:{upper_offset}.')
            im, ce, mzs, ints = _extract_im_ms2(spectrum)
            exp_ims[i] = im
            exp_ces[i] = ce
            exp_mzs[i] = mzs
            exp_intensities[i] = ints

    spec_names = np.array(spec_names)
    spec_indices = np.array(spec_indices).astype(int)
    exp_rts = np.array(exp_rts)
    exp_ims = np.array(exp_ims)
    exp_spectra = pd.DataFrame()
    exp_spectra['mzs'] = exp_mzs
    exp_spectra['intensities'] = exp_intensities
    exp_spectra['ce'] = exp_ces
    return spec_names, spec_indices, exp_rts, exp_ims, exp_spectra