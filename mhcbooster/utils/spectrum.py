import numba
import numpy as np
from mhcbooster.utils.constants import EPSILON

# Remove zero and low intensity signals
@numba.njit
def remove_low_intensity_signal(mzs: np.array, ints: np.array, rel_thresh: float = 0.01) -> np.array:
    intensity_threshold = np.max(ints) * rel_thresh
    mask = ints > intensity_threshold
    mzs = mzs[mask]
    ints = ints[mask]
    return mzs, ints

@numba.njit
def match_spectra(exp_mzs, exp_ints, pred_mzs, pred_ints, mz_tolerance, use_ppm):
    exp_mzs, exp_ints = remove_low_intensity_signal(exp_mzs, exp_ints)
    exp_indices = np.argsort(exp_mzs)
    exp_mzs = exp_mzs[exp_indices]
    exp_ints = exp_ints[exp_indices]

    pred_mzs, pred_ints = remove_low_intensity_signal(pred_mzs, pred_ints)
    pred_indices = np.argsort(pred_mzs)
    pred_mzs = pred_mzs[pred_indices]
    pred_ints = pred_ints[pred_indices]

    exp_idx = 0
    matched_exp_ints = []
    matched_pred_ints = []
    for pred_idx in range(len(pred_mzs)):
        tmp_mz_tolerance = mz_tolerance * pred_mzs[pred_idx] * 1E-6 if use_ppm else mz_tolerance
        tmp_mz_tolerance = max(tmp_mz_tolerance, 0.01)
        mz_start = pred_mzs[pred_idx] - tmp_mz_tolerance
        mz_end = pred_mzs[pred_idx] + tmp_mz_tolerance
        while exp_idx < len(exp_mzs) and exp_mzs[exp_idx] < mz_start:
            exp_idx += 1
        if exp_idx == len(exp_mzs):
            break
        if exp_mzs[exp_idx] > mz_end:
            continue
        max_intensity = -1
        for tmp_exp_idx in range(exp_idx, len(exp_mzs)):
            if exp_mzs[tmp_exp_idx] > mz_end:
                break
            if exp_ints[tmp_exp_idx] > max_intensity:
                max_intensity = exp_ints[tmp_exp_idx]
        matched_exp_ints.append(max_intensity)
        matched_pred_ints.append(pred_ints[pred_idx])
    return np.array(matched_exp_ints, dtype=np.float32), np.array(matched_pred_ints, dtype=np.float32)

@numba.njit
def match_spectra_to_pred(exp_mzs, exp_ints, pred_mzs, mz_tolerance, use_ppm):
    exp_idx = 0
    matched_exp_ints = np.zeros(len(pred_mzs), dtype=np.float32)
    for pred_idx in range(len(pred_mzs)):
        tmp_mz_tolerance = mz_tolerance * pred_mzs[pred_idx] * 1E-6 if use_ppm else mz_tolerance
        tmp_mz_tolerance = max(tmp_mz_tolerance, 0.01)
        mz_start = pred_mzs[pred_idx] - tmp_mz_tolerance
        mz_end = pred_mzs[pred_idx] + tmp_mz_tolerance
        while exp_idx < len(exp_mzs) and exp_mzs[exp_idx] < mz_start:
            exp_idx += 1
        if exp_idx == len(exp_mzs):
            break
        if exp_mzs[exp_idx] > mz_end:
            continue
        max_intensity = 0
        tmp_exp_idx = exp_idx
        while tmp_exp_idx < len(exp_mzs) and exp_mzs[tmp_exp_idx] <= mz_end:
            if exp_ints[tmp_exp_idx] > max_intensity:
                max_intensity = exp_ints[tmp_exp_idx]
            tmp_exp_idx += 1
        matched_exp_ints[pred_idx] = max_intensity
    return matched_exp_ints

@numba.njit
def calc_spectral_entropy(matched_exp_ints, matched_pred_ints):
    if len(matched_exp_ints) == 0 or np.sum(matched_exp_ints) == 0:
        return -1
    matched_exp_ints = matched_exp_ints / np.sum(matched_exp_ints)
    matched_pred_ints = matched_pred_ints / np.sum(matched_pred_ints)
    total_ints = (matched_exp_ints + matched_pred_ints) / 2

    exp_entropy = - np.sum(matched_exp_ints[matched_exp_ints > 0] * np.log(matched_exp_ints[matched_exp_ints > 0]))
    pred_entropy = - np.sum(matched_pred_ints * np.log(matched_pred_ints))
    total_entropy = - np.sum(total_ints * np.log(total_ints))
    return 1 - (2 * total_entropy - exp_entropy - pred_entropy) / np.log(4)

@numba.njit
def calc_cosine_similarity(matched_exp_ints, matched_pred_ints):
    if len(matched_exp_ints) == 0:
        return -1
    dot_product = 0
    for i in range(len(matched_exp_ints)):
        dot_product += matched_exp_ints[i] * matched_pred_ints[i]

    exp_len, pred_len = 0, 0
    for i in range(len(matched_exp_ints)):
        exp_len += matched_exp_ints[i] * matched_exp_ints[i]
        pred_len += matched_pred_ints[i] * matched_pred_ints[i]
    exp_len, pred_len = np.sqrt(exp_len), np.sqrt(pred_len)

    if exp_len == 0 or pred_len == 0:
        return -1
    cos_similarity = dot_product / (exp_len * pred_len)
    return cos_similarity

@numba.njit
def calc_forward_reverse(exp_len, pred_len, matched_len):
    forward_score = float(matched_len) / pred_len
    reverse_score = float(matched_len) / exp_len
    return forward_score, reverse_score

@numba.njit(parallel=True)
def calc_all_ms2_scores(exp_mzs_arr, exp_ints_arr, pred_mzs_arr, pred_ints_arr, pred_anno_arr, mz_tolerance, use_ppm):
    assert len(exp_mzs_arr) == len(pred_mzs_arr)
    n = len(exp_mzs_arr)
    entropy_scores = np.zeros(n, dtype=np.float32)
    cosine_scores = np.zeros(n, dtype=np.float32)
    forward_scores = np.zeros(n, dtype=np.float32)
    reverse_scores = np.zeros(n, dtype=np.float32)
    entropy_b_scores = np.zeros(n, dtype=np.float32)
    entropy_y_scores = np.zeros(n, dtype=np.float32)

    for i in numba.prange(n):  # Parallel loop
        exp_mzs = exp_mzs_arr[i]
        exp_ints = exp_ints_arr[i]
        pred_mzs = pred_mzs_arr[i]
        pred_ints = pred_ints_arr[i]
        pred_annos = pred_anno_arr[i]

        # Sort experimental data
        exp_indices = np.argsort(exp_mzs)
        exp_mzs = exp_mzs[exp_indices]
        exp_ints = exp_ints[exp_indices]

        # Process predicted data
        # pred_mzs, pred_ints = remove_low_intensity_signal(pred_mzs, pred_ints, rel_thresh=0.01)
        pred_indices = np.argsort(pred_mzs)
        pred_mzs = pred_mzs[pred_indices]
        pred_ints = pred_ints[pred_indices]
        pred_annos = pred_annos[pred_indices]

        # Match spectra
        matched_exp_ints = match_spectra_to_pred(exp_mzs, exp_ints, pred_mzs, mz_tolerance, use_ppm)

        # Calculate scores
        entropy_scores[i] = calc_spectral_entropy(matched_exp_ints, pred_ints)
        cosine_scores[i] = calc_cosine_similarity(matched_exp_ints, pred_ints)
        n_matched = np.sum(matched_exp_ints > 0)
        forward_scores[i], reverse_scores[i] = calc_forward_reverse(len(exp_mzs), len(pred_mzs), n_matched)

        matched_b_exp_ints = matched_exp_ints[pred_annos == -1]
        matched_y_exp_ints = matched_exp_ints[pred_annos == 1]
        pred_b_ints = pred_ints[pred_annos == -1]
        pred_y_ints = pred_ints[pred_annos == 1]
        entropy_b_scores[i] = calc_spectral_entropy(matched_b_exp_ints, pred_b_ints)
        entropy_y_scores[i] = calc_spectral_entropy(matched_y_exp_ints, pred_y_ints)

    entropy_scores = np.clip(entropy_scores, a_min=EPSILON, a_max=None)
    cosine_scores = np.clip(cosine_scores, a_min=EPSILON, a_max=None)
    forward_scores = np.clip(forward_scores, a_min=EPSILON, a_max=None)
    reverse_scores = np.clip(reverse_scores, a_min=EPSILON, a_max=None)
    entropy_b_scores = np.clip(entropy_b_scores, a_min=EPSILON, a_max=None)
    entropy_y_scores = np.clip(entropy_y_scores, a_min=EPSILON, a_max=None)

    return entropy_scores, cosine_scores, forward_scores, reverse_scores, entropy_b_scores, entropy_y_scores


if __name__ == '__main__':
    mzs_top = np.array(
        [214.11862, 285.15573, 356.19284, 427.22995, 498.26706, 569.3042, 640.3413, 711.3784, 782.4155, 853.45264,
         303.1663, 374.2034, 445.2405, 516.27765, 587.31476, 658.35187, 729.389])
    ints_top = np.array(
        [0.058000322, 0.11392251, 0.13280249, 0.14267457, 0.16630515, 0.13712373, 0.13238974, 0.055706777, 0.02375361,
         0.004642043, 0.0011714724, 0.0053157704, 0.008672275, 0.0057611926, 0.007857157, 0.0030173366, 0.00088385516])

    mzs_bottom = np.array(
        [72.04439, 90.054955, 143.0815, 161.09207, 214.11862, 232.12918, 285.15573, 303.1663, 356.19284, 374.2034,
         427.22995, 445.2405, 498.26706, 516.27765, 569.3042, 587.31476, 640.3413, 658.35187, 711.3784, 729.389,
         782.4155, 800.4261, 853.45264, 871.4632])
    ints_bottom = np.array(
        [0.000585687, 0.011478545, 0.08412633, 0.074349344, 0.098416984, 0.0050035985, 0.07764679, 0.0024212322,
         0.093568265, 0.002461908, 0.121157214, 0.002367523, 0.12113316, 0.0017787149, 0.14215595, 0.0011818329,
         0.10583924, 0.0013559341, 0.03691124, 0.0017238454, 0.011039282, 0.0001858432, 0.0026443356, 0.00046720088])

    matched_ints_top, matched_ints_bottom = match_spectra(mzs_top, ints_top, mzs_bottom, ints_bottom, 20, True)

    entropy_score = calc_spectral_entropy(matched_ints_top, matched_ints_bottom)
    cosine_similarity = calc_cosine_similarity(matched_ints_top, matched_ints_bottom)
    metapro_similarity = cosine_similarity * len(matched_ints_top) / len(ints_top)
    print(entropy_score, cosine_similarity)