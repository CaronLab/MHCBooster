import os
import re
import requests
import zipfile
import tempfile
import subprocess
import numpy as np
import pandas as pd

from pathlib import Path

from tqdm import tqdm

from mhcbooster.predictors.base_predictor_helper import BasePredictorHelper


class AutortHelper(BasePredictorHelper):
    def __init__(self,
                 peptides_with_mod: list[str],
                 exp_rts: np.array,
                 high_prob_indices: np.array,
                 report_directory: str,
                 fine_tune: bool = False,
                 verbose: bool = False):
        super().__init__('AutoRT', report_directory)

        self.autort_url = 'https://github.com/bzhanglab/AutoRT/archive/refs/heads/master.zip'
        self.autort_root = Path(__file__).parent.parent / 'third_party' / 'AutoRT-master'
        if not self.autort_root.exists():
            self._fetch_autort_by_url()

        # Prepare dataframe for PeptDeep prediction
        self.peptide_df = pd.DataFrame()
        self.peptide_df['sequence'] = [re.sub(r'\[.*?]', '', peptide.replace('M[15.9949]', '1')) for peptide in peptides_with_mod]
        self.peptide_df['retention_time'] = exp_rts

        self.high_prob_indices = high_prob_indices
        self.fine_tune = fine_tune
        self.verbose = verbose

        if self.high_prob_indices is None or len(self.high_prob_indices) < 100:
            print('Not enough high quality peptides for fine-tuning. Skipping...')
            self.fine_tune = False

    def _fetch_autort_by_url(self):
        # Fetch AutoRT from GitHub
        third_party_root = self.autort_root.parent
        if not third_party_root.exists():
            third_party_root.mkdir()

        print('Start to download AutoRT from GitHub. It will take several minutes...')
        response = requests.get(self.autort_url, stream=True)
        if response.status_code == 200:
            destination_path = third_party_root / 'AutoRT-master.zip'
            with open(destination_path, 'wb') as file:
                # Initialize tqdm progress bar with total size and chunk size
                with tqdm(unit='B', unit_scale=True) as pbar:
                    # Download the file in chunks
                    for chunk in response.iter_content(chunk_size=1024):
                        # Write the chunk to the file
                        file.write(chunk)
                        # Update the progress bar with the size of the chunk
                        pbar.update(len(chunk))

                print("\nDownload completed!")
            zip_file = zipfile.ZipFile(str(destination_path))
            zip_file.extractall(third_party_root)
            print(f'Successfully downloaded AutoRT from GitHub and extracted to {third_party_root}')
        else:
            print(f'Failed to download AutoRT from GitHub. Status code: {response.status_code}')

    def predict_df(self):
        print('Running AutoRT predictor...')

        self.model_path = self.autort_root / 'models' / 'general_base_model' / 'model.json'
        autort_exe_path = self.autort_root / 'autort.py'

        # Perform fine_tuning
        if self.fine_tune:
            fine_tuned_model_folder = Path(self.report_directory).parent / 'fine_tuned_models' / 'AutoRT'
            if fine_tuned_model_folder.exists():
                self.model_path = fine_tuned_model_folder / 'model.json'
            else:
                with tempfile.NamedTemporaryFile('w', delete=False) as train_file:
                    # Prepare file for fine-tuning
                    autort_cal_df = self.peptide_df[self.high_prob_indices]

                    autort_input = pd.DataFrame()
                    autort_input['x'] = autort_cal_df['sequence']
                    autort_input['y'] = autort_cal_df['retention_time']
                    autort_input['mod'] = '' # TODO add mods
                    autort_input.to_csv(train_file, sep='\t', index=False, header=True)

                    command = f'python {autort_exe_path} train -i {train_file.name} -m {self.model_path} -o {fine_tuned_model_folder} -rlr'
                    command = 'source ~/.bashrc; conda run -n autort --no-capture-output ' + command
                    print('Fine-tuning AutoRT. It may take several minutes...')
                    print(command)
                    subprocess.run(command, shell=True)
                    self.model_path = fine_tuned_model_folder / 'model.json'

        # try to load from db
        db_data, matched_mask = self.try_load_from_db(keys=self.peptide_df['sequence'])
        print(f'Matched {np.sum(matched_mask)} peptides from DB. Predicting on {len(self.peptide_df) - np.sum(matched_mask)} remaining peptides.')
        self.pred_df = pd.DataFrame({'y_pred': [None] * len(self.peptide_df)})
        self.pred_df.loc[matched_mask, 'y_pred'] = db_data

        if np.sum(matched_mask) < len(self.peptide_df):
            # Prepare file for prediction
            with tempfile.NamedTemporaryFile('w', delete=False) as input_file:
                autort_input = pd.DataFrame()
                autort_input['x'] = self.peptide_df['sequence'][~matched_mask]
                autort_input.to_csv(input_file, sep='\t', index=False)

            # Perform prediction
            with tempfile.NamedTemporaryFile('w', delete=False) as result_file:
                result_file_path = Path(result_file.name)
                command = f'python {autort_exe_path} predict -t {input_file.name} -s {self.model_path} -o {result_file_path.parent} -p {result_file_path.stem}'
                command = 'source ~/.bashrc; conda run -n autort --no-capture-output ' + command
                print('Predicting RTs using AutoRT...')
                print(command)
                subprocess.run(command, shell=True)
                pred_df = pd.read_csv(str(result_file_path) + '.tsv', sep='\t')
                self.save_to_db(keys=self.peptide_df['sequence'][~matched_mask], values=pred_df['y_pred'].tolist())
                self.pred_df.loc[~matched_mask, 'y_pred'] = pred_df['y_pred'].values
                os.remove(input_file.name)
                subprocess.run(f'rm {result_file.name}*', shell=True)

        return self.pred_df

    def score_df(self) -> pd.DataFrame:

        exp_rts = self.peptide_df['retention_time']
        pred_rts = self.pred_df['y_pred'].to_numpy(dtype=np.float32)

        if self.fine_tune:
            predictions = self.calc_rt_scores(exp_rts, pred_rts)
            self.align_pred_to_exp(pred_rts[self.high_prob_indices], exp_rts[self.high_prob_indices], figure_name='alignment_autort')
        else:
            if self.high_prob_indices is None:
                aligned_pred_rts = self.align_pred_to_exp_coarse(pred_rts, exp_rts, figure_name='alignment_autort')
            else:
                aligned_pred_rts = self.align_pred_to_exp(pred_rts[self.high_prob_indices], exp_rts[self.high_prob_indices], pred_rts, figure_name='alignment_autort')
            predictions = self.calc_rt_scores(exp_rts, aligned_pred_rts)

        return predictions
