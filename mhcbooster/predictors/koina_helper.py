import re
import time
import numpy as np
import pandas as pd
from typing import List, Union
from koinapy import Koina
from mhcbooster.utils.constants import MASS_UNIMOD_MAP
from mhcbooster.predictors.base_predictor_helper import BasePredictorHelper
from mhcbooster.utils.peptide import convert_mass_diff_to_unimod, remove_modifications

# Updated in Nov 5, 2024
koina_predictor_map = {
    'Deeplc_hela_hf': 'RT',
    'AlphaPeptDeep_rt_generic': 'RT',
    'Prosit_2019_irt': 'RT',
    'Prosit_2024_irt_cit': 'RT',
    'Prosit_2020_irt_TMT': 'RT', # only TMT labeled
    'Chronologer_RT': 'RT',

    'IM2Deep': 'CCS',
    'AlphaPeptDeep_ccs_generic': 'CCS',

    # 'UniSpec': 'MS2', # instrument_types, ion format, not available
    'AlphaPeptDeep_ms2_generic': 'MS2', # instrument_types
    'ms2pip_HCD2021': 'MS2',
    'ms2pip_timsTOF2023': 'MS2',
    'ms2pip_iTRAQphospho': 'MS2',
    'ms2pip_Immuno_HCD': 'MS2',
    'ms2pip_TTOF5600': 'MS2',
    'ms2pip_timsTOF2024': 'MS2',
    'ms2pip_CID_TMT': 'MS2',
    'Prosit_2019_intensity': 'MS2',
    'Prosit_2024_intensity_cit': 'MS2', # fragmentation_types
    'Prosit_2023_intensity_timsTOF': 'MS2',
    # 'Prosit_2024_intensity_PTMs_gl': 'MS2', # fragmentation_types, not available
    'Prosit_2020_intensity_CID': 'MS2',
    'Prosit_2020_intensity_HCD': 'MS2',
    # 'Prosit_2024_intensity_XL_NMS2': 'MS2', # cross-linked only
    # 'Prosit_2023_intensity_XL_CMS2': 'MS2', # cross-linked only
    'Prosit_2020_intensity_TMT': 'MS2', # fragmentation_types, TMT only
    # 'Prosit_2023_intensity_XL_CMS3': 'MS2' # MS3
}

all_koina_predictors = list(koina_predictor_map.keys())
general_koina_predictors = [p for p in koina_predictor_map.keys() if 'TMT' not in p]
general_koina_predictors.remove('AlphaPeptDeep_ms2_generic') # Bug in return length 02082025

supported_mod_map = {
    'Prosit': ['C4', 'M35', 'n737', 'K737'],
    'UniSpec': ['n1', 'C4', 'S21', 'T21', 'Y21', 'C26', 'E27', 'Q28', 'M35'],
    'diann': ['n1', 'C4', 'S21', 'T21', 'Y21', 'M35', 'n121', 'K121', 'T121', 'C121', 'S121', 'n737', 'K737', 'S737'],
    'ms2pip': ['C4', 'M35'],
    'Deeplc': ['n1', 'C4', 'S21', 'T21', 'Y21', 'M35']
}
# TODO
supported_instrument_map = {
    'peptdeep': ['QE', 'LUMOS', 'TIMSTOF', 'SCIEXTOF'],
    'UniSpec': ['QE', 'QEHFX', 'LUMOS', 'ELITE', 'VELOS', 'NONE']
}

class KoinaHelper(BasePredictorHelper):

    def __init__(self,
                 peptides: List[str],
                 peptides_with_mods: List[str],
                 charges: List[str],
                 predictor_names: Union[str, List[str]],
                 report_directory: str,
                 exp_rts: np.array = None,
                 exp_ims: np.array = None,
                 exp_ms2s: pd.DataFrame = None,
                 high_prob_indices: np.array = None,
                 koina_server_url: str = 'koina.wilhelmlab.org:443',
                 mz_tolerance: float = 20,
                 use_ppm: bool = True,
                 instrument_type: str = 'QE',
                 fragmentation_type: str = 'HCD',
                 ):

        if isinstance(predictor_names, str):
            predictor_names = [predictor_names]

        super().__init__(f'Koina', report_directory)

        self.predictor_map = {}
        for predictor_name in predictor_names:
            match_idx = -1
            for i, predictor in enumerate(all_koina_predictors):
                if predictor_name.lower() == predictor.lower():
                    match_idx = i
                    break
            if match_idx == -1:
                print(f'Skipping.... {predictor_name} Koina predictor is not supported')
                continue
            predictor_name = all_koina_predictors[match_idx]
            predictor_type = koina_predictor_map[predictor_name]
            if predictor_type not in self.predictor_map.keys():
                self.predictor_map[predictor_type] = [predictor_name]
            else:
                self.predictor_map[predictor_type].append(predictor_name)

        self.exp_rts = exp_rts
        self.exp_ims = exp_ims
        self.exp_ms2s = exp_ms2s
        self.mz_tolerance = mz_tolerance
        self.use_ppm = use_ppm
        self.instrument_type = instrument_type
        self.fragmentation_type = fragmentation_type
        self.charges = charges

        self.peptides_no_mod = peptides
        self.peptides_with_mods = peptides_with_mods
        self.koina_server_url = koina_server_url

        self.high_prob_indices = high_prob_indices
        if np.sum(high_prob_indices) == 0:
            self.high_prob_indices = None


    def _predict_rt(self, peptides, unsolved_mods, charges, ces):
        peptide_df = pd.DataFrame(peptides, columns=['peptide_sequences'])
        pred_df = self.model.predict(peptide_df)
        return pred_df

    def _predict_ccs(self, peptides, unsolved_mods, charges, ces):
        peptide_df = pd.DataFrame(peptides, columns=['peptide_sequences'])
        peptide_df['precursor_charges'] = charges
        pred_df = self.model.predict(peptide_df)
        return pred_df

    def _predict_ms2(self, peptides, unsolved_mods, charges, ces):
        peptide_df = pd.DataFrame(peptides, columns=['peptide_sequences'])
        peptide_df['precursor_charges'] = charges
        peptide_df['collision_energies'] = ces
        peptide_df['instrument_types'] = np.array([self.instrument_type] * len(peptide_df))
        peptide_df['fragmentation_types'] = np.array([self.fragmentation_type] * len(peptide_df))
        peptide_df = peptide_df.reset_index(drop=False)
        pred_df = pd.DataFrame(self.model.predict(peptide_df))

        pred_df = (pred_df.groupby('index').agg({'intensities': list, 'mz': list, 'annotation': list})
                   .sort_values(by=['index'], ascending=True))
        pred_df = pred_df.rename(columns={'mz': 'mzs'})

        #deal with unsupported modifications by prediction models
        sequences = np.array(remove_modifications(peptides))
        for i in range(len(pred_df)):
            for position, mass_diff in unsolved_mods[i].items():
                b_position = position
                y_position = len(sequences[i]) - position
                for j, anno in enumerate(pred_df.loc[i, 'annotation']):
                    ion = anno.decode('utf8')
                    ion_type = ion[0]
                    ion_pos = int(ion[1:-2]) # TODO
                    ion_charge = int(ion[-1])
                    if (ion_type == 'b' and ion_pos >= b_position) or (ion_type == 'y' and ion_pos > y_position):
                        pred_df.loc[i, 'mzs'][j] += mass_diff / ion_charge

        return pred_df

    def predict_df(self) -> pd.DataFrame:
        pred_func_map = {'RT': self._predict_rt,
                         'CCS': self._predict_ccs,
                         'MS2': self._predict_ms2}
        self.ces = self.exp_ms2s['ce'].to_numpy(dtype=np.float32) if self.exp_ms2s is not None else np.array([25] * len(self.peptides_with_mods))
        combined_pred_df = pd.DataFrame()
        for predictor_type in self.predictor_map.keys():
            for predictor_name in self.predictor_map[predictor_type]:
                keys = np.array([','.join([self.peptides_with_mods[i], str(self.charges[i]), str(self.ces[i])]) for i in range(len(self.peptides_with_mods))])
                db_data, matched_mask = self.try_load_from_db(keys, predictor_name)
                print(f'Matched {len(db_data)} peptides from DB. Predicting on {len(keys) - len(db_data)} remaining peptides.')

                pred_df = pd.DataFrame(index=range(len(keys)))
                if len(db_data) > 0:
                    pred_df = pd.DataFrame(np.nan, index=range(len(keys)), columns=list(db_data[0].keys())).astype(object)
                    pred_df.loc[matched_mask] = pd.DataFrame(db_data)[pred_df.columns].values

                if len(db_data) < len(keys):
                    peptides_with_mods = np.array(self.peptides_with_mods)[~matched_mask]
                    charges = np.array(self.charges)[~matched_mask]
                    ces = np.array(self.ces)[~matched_mask]

                    supported_mods = None
                    for tool_name in supported_mod_map.keys():
                        if tool_name in predictor_name:
                            supported_mods = supported_mod_map[tool_name]
                            break
                    # TODO: Add more supported mods
                    peptides, unsolved_mods = convert_mass_diff_to_unimod(peptides_with_mods, MASS_UNIMOD_MAP, supported_mods)

                    attempt = 0
                    max_retries = 1  # TODO: Maximum number of retries
                    while attempt < max_retries:
                        try:
                            self.model = Koina(predictor_name, self.koina_server_url)
                            result_df = pred_func_map[predictor_type](peptides, unsolved_mods, charges, ces)
                            self.save_to_db(keys=keys[~matched_mask], values=result_df.to_dict(orient='records'), predictor_name=predictor_name)
                            pred_df.loc[~matched_mask, result_df.columns] = result_df.values
                            break
                        except Exception as e:
                            print(f"Warning: {e}")
                            attempt += 1
                            if attempt < max_retries:
                                print(f"Retrying in 1 minute... (Attempt {attempt}/{max_retries})")
                                time.sleep(60)
                            else:
                                print("Max retries reached. Operation failed.")
                                raise

                if predictor_type == 'RT':
                    if 'irt' in pred_df.keys():
                        combined_pred_df[f'{predictor_name}_rt'] = pred_df['irt']
                    else:
                        combined_pred_df[f'{predictor_name}_rt'] = pred_df['rt']
                elif predictor_type == 'CCS':
                    combined_pred_df[f'{predictor_name}_ccs'] = pred_df['ccs']
                elif predictor_type == 'MS2':
                    combined_pred_df[f'{predictor_name}_mzs'] = pred_df['mzs']
                    combined_pred_df[f'{predictor_name}_intensities'] = pred_df['intensities']
                    combined_pred_df[f'{predictor_name}_annotations'] = pred_df['annotation']

        self.pred_df = combined_pred_df

        return self.pred_df

    def score_df(self) -> pd.DataFrame:

        def convert_anno(anno):
            if isinstance(anno, bytes):
                anno = anno.decode('utf8')
            match = re.match(r'([by])(\d+)', anno)
            if match:
                prefix = match.group(1)
                num = int(match.group(2))
                if prefix == 'b' and num >= 3:
                    return -1
                elif prefix == 'y' and num >= 3:
                    return 1
            return 0

        all_predictions = pd.DataFrame()
        for predictor_type in self.predictor_map.keys():
            for predictor_name in self.predictor_map[predictor_type]:
                if predictor_type == 'RT':
                    pred_rts = self.pred_df[f'{predictor_name}_rt'].to_numpy(dtype=np.float32)
                    if self.high_prob_indices is None:
                        aligned_pred_rts =  self.align_pred_to_exp_coarse(pred_rts, self.exp_rts, figure_name=f'alignment_{predictor_name}')
                    else:
                        aligned_pred_rts = self.align_pred_to_exp(pred_rts[self.high_prob_indices], self.exp_rts[self.high_prob_indices], pred_rts, figure_name=f'alignment_{predictor_name}')
                    predictions = self.calc_rt_scores(self.exp_rts, aligned_pred_rts, predictor_name)
                    all_predictions = all_predictions.join(predictions, how='outer')
                elif predictor_type == 'CCS':
                    pred_ims = self.pred_df[f'{predictor_name}_ccs'].to_numpy(dtype=np.float32) / self.charges
                    if self.high_prob_indices is None:
                        aligned_pred_ims = self.align_pred_to_exp_coarse(pred_ims, self.exp_ims, figure_name=f'alignment_{predictor_name}')
                    else:
                        aligned_pred_ims = self.align_pred_to_exp(pred_ims[self.high_prob_indices], self.exp_ims[self.high_prob_indices], pred_ims, figure_name=f'alignment_{predictor_name}')
                    predictions = self.calc_im_scores(self.exp_ims, aligned_pred_ims, predictor_name)
                    all_predictions = all_predictions.join(predictions, how='outer')
                elif predictor_type == 'MS2':
                    pred_ms2 = pd.DataFrame()
                    pred_ms2['mzs'] = self.pred_df[f'{predictor_name}_mzs']
                    pred_ms2['intensities'] = self.pred_df[f'{predictor_name}_intensities']
                    pred_ms2['annotations'] = self.pred_df[f'{predictor_name}_annotations']
                    pred_ms2['annotations'] = pred_ms2['annotations'].apply(lambda x: [convert_anno(a) for a in x])

                    predictions = self.calc_ms2_scores(self.exp_ms2s, pred_ms2, self.mz_tolerance, self.use_ppm, predictor_name)
                    all_predictions = all_predictions.join(predictions, how='outer')

        return all_predictions


