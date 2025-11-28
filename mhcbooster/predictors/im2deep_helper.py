import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from mhcbooster.utils.constants import MASS_UNIMOD_MAP, UNIMOD_NAME_MAP
from mhcbooster.utils.peptide import get_pos_unimod_map
from mhcbooster.predictors.base_predictor_helper import BasePredictorHelper


class IM2DeepHelper(BasePredictorHelper):
    def __init__(self,
                 peptides: list[str],
                 peptides_with_mods: list[str],
                 charges: list[int],
                 exp_ims: np.array,
                 high_prob_indices: np.array,
                 report_directory: str,
                 fine_tune: bool = False,
                 verbose: bool = False):
        super().__init__('IM2Deep', report_directory)

        # Prepare dataframe for PeptDeep prediction
        self.peptide_df = pd.DataFrame(peptides, columns=['seq'])
        self.peptide_df['charge'] = charges
        self.peptide_df['modifications'] = ''
        for i, peptide in enumerate(peptides_with_mods):
            pos_unimod_map, _ = get_pos_unimod_map(peptide, MASS_UNIMOD_MAP)
            if len(pos_unimod_map) == 0:
                continue
            mod = []
            for pos, unimod_num in pos_unimod_map.items():
                mod.append(str(pos))
                mod.append(UNIMOD_NAME_MAP[unimod_num])
            mod = '|'.join(mod)
            self.peptide_df.loc[i, 'modifications'] = mod
        self.peptide_df['CCS'] = exp_ims * charges * 200

        self.exp_ims = exp_ims
        self.high_prob_indices = high_prob_indices
        self.fine_tune = fine_tune
        self.verbose = verbose

        if self.high_prob_indices is None or len(self.high_prob_indices) < 100:
            print('Not enough high quality peptides for fine-tuning. Skipping...')
            self.fine_tune = False

    def predict_df(self):
        print('Running IM2Deep predictor...')

        # Prepare file for calibration
        if self.fine_tune:
            with tempfile.NamedTemporaryFile('w', delete=False) as train_file:
                cal_df = self.peptide_df[self.high_prob_indices]
                if len(cal_df) < 100:
                    print('Not enough high quality PSMs for model calibration. Skipping...')
                    self.fine_tune = False
                else:
                    cal_df.to_csv(train_file, index=False, header=True)
                    print(f'{len(cal_df)} high quality PSMs are used for calibration.')

        keys = self.peptide_df.apply(lambda x: f"{x['seq']},{x['charge']},{x['modifications']}", axis=1)
        db_data, matched_mask = self.try_load_from_db(keys=keys)
        print(f'Matched {len(db_data)} peptides from DB. Predicting on {len(keys) - len(db_data)} remaining peptides.')

        self.pred_df = pd.DataFrame({'predicted CCS': [None] * len(self.peptide_df),
                                     'charge': [None] * len(self.peptide_df)})
        if len(db_data) > 0:
            db_df = pd.DataFrame(db_data)
            self.pred_df.loc[matched_mask, 'predicted CCS'] = db_df['predicted CCS'].values
            self.pred_df.loc[matched_mask, 'charge'] = db_df['charge'].values

        if len(db_data) < len(keys):
            # Prepare file for prediction
            with tempfile.NamedTemporaryFile('w', delete=False) as input_file:
                self.peptide_df.loc[~matched_mask].to_csv(input_file, index=False, header=True)
            # Perform prediction
            with tempfile.NamedTemporaryFile('w', delete=False) as result_file:
                if self.fine_tune:
                    command = f'im2deep {input_file.name} -c {train_file.name} -o {result_file.name}'
                else:
                    command = f'im2deep {input_file.name} -o {result_file.name}'

                print('Predicting CCSs using IM2Deep...')
                if self.verbose:
                    subprocess.run(command, shell=True)
                else:
                    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                pred_df = pd.read_csv(result_file.name)
                keys = keys[~matched_mask]
                values = pred_df.to_dict(orient='records')
                self.save_to_db(keys=keys, values=values)
                self.pred_df.loc[~matched_mask, 'predicted CCS'] = pred_df['predicted CCS'].values
                self.pred_df.loc[~matched_mask, 'charge'] = pred_df['charge'].values

        if self.fine_tune:
            os.remove(train_file.name)
        os.remove(input_file.name)
        os.remove(result_file.name)

        return self.pred_df

    def score_df(self) -> pd.DataFrame:

        pred_ims = self.pred_df['predicted CCS'].to_numpy(dtype=np.float32) / self.pred_df['charge'].to_numpy(
            dtype=int) / 200
        if self.fine_tune:
            predictions = self.calc_im_scores(self.exp_ims, pred_ims)
            self.align_pred_to_exp(pred_ims[self.high_prob_indices], self.exp_ims[self.high_prob_indices], pred_ims, figure_name='alignment_im2deep')
        else:
            if self.high_prob_indices is None:
                aligned_pred_ims = self.align_pred_to_exp_coarse(pred_ims, self.exp_ims, figure_name='alignment_im2deep')
            else:
                aligned_pred_ims = self.align_pred_to_exp(pred_ims[self.high_prob_indices], self.exp_ims[self.high_prob_indices], pred_ims, figure_name='alignment_im2deep')
            predictions = self.calc_im_scores(self.exp_ims, aligned_pred_ims)

        return predictions
