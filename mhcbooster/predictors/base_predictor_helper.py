import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mhcbooster.utils.constants import EPSILON
from mhcbooster.utils.spectrum import calc_all_ms2_scores, calc_spectral_entropy, calc_forward_reverse, calc_cosine_similarity, \
    match_spectra_to_pred, remove_low_intensity_signal
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import pickle, zlib, time

MEMORY_DB = {}

class BasePredictorHelper:
    def __init__(self, predictor_name, report_directory):
        self.predictor_name = predictor_name
        self.report_directory = report_directory

    def predict_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def score_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def format_pred_result_for_saving(self) -> pd.DataFrame:
        raise NotImplementedError

    def try_load_from_db(self, keys, predictor_name=None):
        start_time = time.time()
        db_name = predictor_name if predictor_name is not None else self.predictor_name

        # db_path = str((Path(self.report_directory).parent / f'{db_name}.db').resolve())
        matched_mask = np.zeros(len(keys), dtype=bool)
        data = []
        if len(keys) == 0 or not db_name in MEMORY_DB:
            return data, matched_mask
        db = MEMORY_DB[db_name]
        for i, key in enumerate(keys):
            if key in db:
                matched_mask[i] = True
                data.append(pickle.loads(zlib.decompress(db[key])))
        print(f'Loading from db {db_name} took {time.time() - start_time:.2f} seconds')
        return data, matched_mask

    def save_to_db(self, keys, values, predictor_name=None):
        start_time = time.time()
        if len(keys) == 0:
            return True
        db_name = predictor_name if predictor_name is not None else self.predictor_name

        if db_name not in MEMORY_DB:
            MEMORY_DB[db_name] = {}

        for i in range(len(keys)):
            MEMORY_DB[db_name][keys[i]] = zlib.compress(pickle.dumps(values[i], protocol=4))

        print(f'Saving {len(keys)} KVs to db {db_name} took {time.time() - start_time:.2f} seconds')
        return True

    def calc_rt_scores(self, exp_rts: np.ndarray, pred_rts: np.ndarray, predictor_name: str = None) -> pd.DataFrame:
        predictions = pd.DataFrame()
        if predictor_name is None:
            predictions[f'{self.predictor_name}_rt_error'] = np.clip(np.abs(exp_rts - pred_rts), a_min=EPSILON, a_max=None)
            predictions[f'{self.predictor_name}_log_rt_error'] = np.log(np.abs(exp_rts - pred_rts) + 0.01)
            predictions[f'{self.predictor_name}_rt_rel_error'] = np.clip(predictions[f'{self.predictor_name}_rt_error'] / exp_rts, a_min=EPSILON, a_max=1)
            predictions[f'{self.predictor_name}_log_rt_rel_error'] = np.log(predictions[f'{self.predictor_name}_rt_error'] / exp_rts + 0.01)
        else:
            predictions[f'{predictor_name}_rt_error'] = np.clip(np.abs(exp_rts - pred_rts), a_min=EPSILON, a_max=None)
            predictions[f'{predictor_name}_log_rt_error'] = np.log(np.abs(exp_rts - pred_rts) + 0.01)
            predictions[f'{predictor_name}_rt_rel_error'] = np.clip(predictions[f'{predictor_name}_rt_error'] / exp_rts, a_min=EPSILON, a_max=1)
            predictions[f'{predictor_name}_log_rt_rel_error'] = np.log(predictions[f'{predictor_name}_rt_error'] / exp_rts + 0.01)
        return predictions

    def calc_im_scores(self, exp_ims: np.ndarray, pred_ims: np.ndarray, predictor_name: str = None) -> pd.DataFrame:
        predictions = pd.DataFrame()
        if predictor_name is None:
            predictions[f'{self.predictor_name}_im_error'] = np.clip(np.abs(exp_ims - pred_ims), a_min=EPSILON, a_max=None)
        else:
            predictions[f'{predictor_name}_im_error'] = np.clip(np.abs(exp_ims - pred_ims), a_min=EPSILON, a_max=None)
        return predictions

    def calc_ms2_scores(self, exp_spectra_df: pd.DataFrame, pred_spectra_df: pd.DataFrame,
                        mz_tolerance: float, use_ppm: bool, predictor_name: str = None) -> pd.DataFrame:
        from numba.typed import List
        assert len(exp_spectra_df) == len(pred_spectra_df)
        exp_mzs_arr, exp_ints_arr, pred_mzs_arr, pred_ints_arr, pred_anno_arr = List(), List(), List(), List(), List()
        exp_mzs_arr.extend([np.array(mzs, dtype=np.float32) for mzs in exp_spectra_df['mzs']])
        exp_ints_arr.extend([np.array(ints, dtype=np.float32) for ints in exp_spectra_df['intensities']])
        pred_mzs_arr.extend([np.array(mzs, dtype=np.float32) for mzs in pred_spectra_df['mzs']])
        pred_ints_arr.extend([np.array(ints, dtype=np.float32) for ints in pred_spectra_df['intensities']])
        pred_anno_arr.extend([np.array(anno, dtype=np.int32) for anno in pred_spectra_df['annotations']])

        entropy_scores, cosine_scores, forward_scores, reverse_scores, entropy_b_scores, entropy_y_scores = (
            calc_all_ms2_scores(exp_mzs_arr, exp_ints_arr, pred_mzs_arr, pred_ints_arr, pred_anno_arr, mz_tolerance, use_ppm))

        predictions = pd.DataFrame()
        if predictor_name is None:
            predictions[f'{self.predictor_name}_entropy_score'] = entropy_scores
            predictions[f'{self.predictor_name}_cosine_score'] = cosine_scores
            predictions[f'{self.predictor_name}_forward_score'] = forward_scores
            predictions[f'{self.predictor_name}_reverse_score'] = reverse_scores
            predictions[f'{self.predictor_name}_entropy_b_score'] = entropy_b_scores
            predictions[f'{self.predictor_name}_entropy_y_score'] = entropy_y_scores
        else:
            predictions[f'{predictor_name}_entropy_score'] = entropy_scores
            predictions[f'{predictor_name}_cosine_score'] = cosine_scores
            predictions[f'{predictor_name}_forward_score'] = forward_scores
            predictions[f'{predictor_name}_reverse_score'] = reverse_scores
            predictions[f'{predictor_name}_entropy_b_score'] = entropy_b_scores
            predictions[f'{predictor_name}_entropy_y_score'] = entropy_y_scores
        return predictions

    def calc_ms2_scores_combine(self, exp_spectra_df: pd.DataFrame, pred_spectra_df: pd.DataFrame,
                        mz_tolerance: float, use_ppm: bool, predictor_name: str = None) -> pd.DataFrame:
        assert len(exp_spectra_df) == len(pred_spectra_df)
        if predictor_name is None:
            predictions = pd.DataFrame(columns=[f'{self.predictor_name}_entropy_score',
                                                f'{self.predictor_name}_cosine_score',
                                                f'{self.predictor_name}_entropy_metapro',
                                                f'{self.predictor_name}_cosine_metapro',
                                                f'{self.predictor_name}_forward_score',
                                                f'{self.predictor_name}_reverse_score'])
        else:
            predictions = pd.DataFrame(columns=[f'{predictor_name}_entropy_score',
                                                f'{predictor_name}_cosine_score',
                                                f'{predictor_name}_entropy_metapro',
                                                f'{predictor_name}_cosine_metapro',
                                                f'{predictor_name}_forward_score',
                                                f'{predictor_name}_reverse_score'])
        for i in range(len(exp_spectra_df)):
            exp_spectrum = exp_spectra_df.iloc[i]
            pred_spectrum = pred_spectra_df.iloc[i]
            exp_mzs = np.array(exp_spectrum.mzs, dtype=np.float32)
            exp_ints = np.array(exp_spectrum.intensities, dtype=np.float32)
            pred_mzs = np.array(pred_spectrum.mzs, dtype=np.float32)
            pred_ints = np.array(pred_spectrum.intensities, dtype=np.float32)

            exp_indices = np.argsort(exp_mzs)
            exp_mzs = exp_mzs[exp_indices]
            exp_ints = exp_ints[exp_indices]

            pred_mzs, pred_ints = remove_low_intensity_signal(pred_mzs, pred_ints, rel_thresh=0.01)
            pred_indices = np.argsort(pred_mzs)
            pred_mzs = pred_mzs[pred_indices]
            pred_ints = pred_ints[pred_indices]

            matched_exp_ints = match_spectra_to_pred(exp_mzs, exp_ints, pred_mzs, mz_tolerance, use_ppm)
            entropy_score = calc_spectral_entropy(matched_exp_ints, pred_ints)
            cosine_score = calc_cosine_similarity(matched_exp_ints, pred_ints)

            n_matched = np.sum(matched_exp_ints > 0)
            forward_score, reverse_score = calc_forward_reverse(len(exp_mzs), len(pred_mzs), n_matched)
            entropy_metapro = calc_spectral_entropy(matched_exp_ints[matched_exp_ints > 0], pred_ints[matched_exp_ints > 0]) * forward_score
            cosine_metapro = calc_cosine_similarity(matched_exp_ints[matched_exp_ints > 0], pred_ints[matched_exp_ints > 0]) * forward_score

            predictions.loc[i] = np.clip([entropy_score, cosine_score, entropy_metapro, cosine_metapro, forward_score, reverse_score],
                                         a_min=EPSILON, a_max=None)
        return predictions

    def align_pred_to_exp(self, pred_array: np.ndarray, exp_array: np.ndarray, full_pred_array: np.ndarray = None, figure_name: str = None):
        # Remove far noises
        min_exp, max_exp = np.min(exp_array), np.max(exp_array)
        min_pred, max_pred = np.min(pred_array), np.max(pred_array)
        if min_exp == max_exp or min_pred == max_pred:
            return pred_array if full_pred_array is None else full_pred_array
        k = (max_exp - min_exp) / (max_pred - min_pred)
        b = (max_exp + min_exp) / 2 - (max_pred + min_pred) / 2 * k
        m = (max_exp - min_exp) / 2
        below_1 = exp_array <= (pred_array * k + b + m)
        above_2 = exp_array >= (pred_array * k + b - m)
        mask = below_1 * above_2
        pred_array = pred_array[mask]
        exp_array = exp_array[mask]

        if len(pred_array) < 200:
            poly_func = np.poly1d(np.polyfit(pred_array, exp_array, 2))
            aligned_pred_array = poly_func(pred_array)
            sorted_pred_array = aligned_pred_array[np.argsort(pred_array)]
            monotonic = np.all(sorted_pred_array[:-1] <= sorted_pred_array[1:])
            if not monotonic:
                return self.align_pred_to_exp_coarse(pred_array, exp_array, full_pred_array, figure_name)
        else:
            frac = max(0.2, 0.7/np.log10(len(pred_array)))
            monotonic = False
            iter = 0
            while not monotonic and iter < 5 and frac <= 1:
                aligned_pred_array = lowess(exp_array, pred_array, frac=frac, it=5, return_sorted=False)
                sorted_pred_array = aligned_pred_array[np.argsort(pred_array)]
                monotonic = np.all(sorted_pred_array[:-1] <= sorted_pred_array[1:])
                if not monotonic:
                    frac = frac * 1.5
                iter += 1
            if not monotonic or frac > 1:
                poly_func = np.poly1d(np.polyfit(pred_array, exp_array, 2))
                aligned_pred_array = poly_func(pred_array)
                sorted_pred_array = aligned_pred_array[np.argsort(pred_array)]
                monotonic = np.all(sorted_pred_array[:-1] <= sorted_pred_array[1:])
                if not monotonic:
                    return self.align_pred_to_exp_coarse(pred_array, exp_array, full_pred_array, figure_name)

        alignment_repository = os.path.join(self.report_directory, 'alignment')
        os.makedirs(alignment_repository, exist_ok=True)
        if figure_name is not None:
            plt.scatter(pred_array, exp_array, alpha=0.5, s=5, color='black')
            plt.scatter(pred_array, aligned_pred_array, alpha=0.5, s=5, color='red')
            plt.savefig(os.path.join(alignment_repository, f'{figure_name}.png'))
            plt.close()

        if full_pred_array is None:
            return aligned_pred_array
        _, indices = np.unique(pred_array, return_index=True)
        function = interp1d(pred_array[indices], aligned_pred_array[indices], kind='linear', fill_value="extrapolate")
        full_aligned_pred_array = function(full_pred_array)

        return full_aligned_pred_array

    def align_pred_to_exp_coarse(self, pred_array: np.ndarray, exp_array: np.ndarray,
                                 full_pred_array: np.ndarray = None, figure_name: str = None):
        min_exp, max_exp = np.min(exp_array), np.max(exp_array)
        min_pred, max_pred = np.min(pred_array), np.max(pred_array)
        if min_exp == max_exp or min_pred == max_pred:
            return pred_array if full_pred_array is None else full_pred_array
        aligned_pred_array = (pred_array - min_pred) / (max_pred - min_pred) * (max_exp - min_exp) + min_exp

        alignment_repository = os.path.join(self.report_directory, 'alignment')
        os.makedirs(alignment_repository, exist_ok=True)
        if figure_name is not None:
            plt.scatter(pred_array, exp_array, alpha=0.5, s=5, color='black')
            plt.scatter(pred_array, aligned_pred_array, alpha=0.5, s=5, color='red')
            plt.savefig(os.path.join(alignment_repository, f'{figure_name}.png'))
            plt.close()

        if full_pred_array is None:
            return aligned_pred_array
        _, indices = np.unique(pred_array, return_index=True)
        function = interp1d(pred_array[indices], aligned_pred_array[indices], kind='linear', fill_value="extrapolate")
        full_aligned_pred_array = function(full_pred_array)

        return full_aligned_pred_array


    def draw_prediction_distributions(self, predictions, label):
        fig_repository = os.path.join(self.report_directory, 'score_distribution')
        os.makedirs(fig_repository, exist_ok=True)

        for column in predictions.columns:
            values = predictions[column]
            plt.hist(values[label == 1], bins=100, alpha=0.5, color='green', label='Target')
            plt.hist(values[label == 0], bins=100, alpha=0.5, color='red', label='Decoy')
            plt.legend()
            plt.title(f'{self.predictor_name}-{column}')
            plt.savefig(os.path.join(fig_repository, f'{self.predictor_name}-{column}.png'))
            plt.close()