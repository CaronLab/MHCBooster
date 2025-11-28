import re
import tempfile
import subprocess
import time
import requests
import zipfile

import mhcgnomes
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm
from mhcnames import normalize_allele_name
from mhcbooster.utils.constants import EPSILON
from mhcbooster.predictors.base_predictor_helper import BasePredictorHelper


class BigMhcHelper(BasePredictorHelper):
    def __init__(self,
                 peptides: list[str],
                 alleles: list[str],
                 report_directory: str = None):
        super().__init__('BigMHC', report_directory)

        self.bigmhc_url = 'https://github.com/KarchinLab/bigmhc/archive/refs/heads/master.zip'
        self.bigmhc_root = Path(__file__).parent.parent / 'third_party' / 'bigmhc-master'
        if not self.bigmhc_root.exists():
            self._fetch_bigmhc_by_url()

        if alleles is None or len(alleles) == 0:
            raise RuntimeError('Alleles are needed for BigMHC predictions.')
        if np.min(np.vectorize(len)(peptides)) < 8:
            raise RuntimeError('BigMHC cannot make predictions on peptides shorter than 8 mer.')

        self.peptides = peptides
        self.alleles = self._format_class_I_alleles(alleles)
        self.bigmhc_exe_path = Path(__file__).parent.parent / 'third_party' / 'bigmhc-master' / 'src' / 'predict.py'

    def _fetch_bigmhc_by_url(self):
        # Fetch BigMHC from GitHub
        third_party_root = self.bigmhc_root.parent
        if not third_party_root.exists():
            third_party_root.mkdir()

        print('Start to download BigMHC from GitHub. It will take several minutes...')
        response = requests.get(self.bigmhc_url, stream=True)
        if response.status_code == 200:
            destination_path = third_party_root / 'bigmhc-master.zip'
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
            print(f'Successfully downloaded BigMHC from GitHub and extracted to {third_party_root}')
        else:
            print(f'Failed to download BigMHC from GitHub. Status code: {response.status_code}')

    def _format_class_I_alleles(self, alleles: List[str]):
        std_alleles = []
        for allele in set(alleles):
            try:
                std_alleles.append(normalize_allele_name(allele))
            except ValueError:
                try:
                    std_alleles.append(mhcgnomes.parse(allele).to_string())
                except ValueError:
                    print(f'ERROR: Allele {allele} not supported.')
        return std_alleles

    def predict_df(self, im=False, gpu=True):
        print('Running BigMHC...')

        matched_pmhcs = []
        pep_file_lines = []
        for allele in self.alleles:
            keys = [allele + ',' + self.peptides[i] for i in range(len(self.peptides))]
            db_data, matched_mask = self.try_load_from_db(keys=keys)
            matched_pmhcs += db_data
            pep_file_lines += np.array(keys)[~matched_mask].tolist()
        print(f'Matched {len(matched_pmhcs)} pMHCs from DB. Predicting on {len(pep_file_lines)} remaining pMHCs.')

        if len(pep_file_lines) == 0:
            self.pred_df = pd.DataFrame(matched_pmhcs)
        else:
            with tempfile.NamedTemporaryFile('w', delete=False) as pep_file:
                pep_file.write('mhc,pep\n')
                pep_file.write('\n'.join(pep_file_lines))
                pep_file.write('\n')
                pep_file_path = pep_file.name
            with tempfile.NamedTemporaryFile('w') as results:
                results_file_path = results.name

            device = 'all' if gpu else 'cpu'
            if not im:
                command = f'python {self.bigmhc_exe_path} -i {pep_file_path} -o {results_file_path} -m el -d {device}'
            else:
                command = f'python {self.bigmhc_exe_path} -i {pep_file_path} -o {results_file_path} -m im -d {device}'
            subprocess.run(command, shell=True)

            pred_df = pd.read_csv(results_file_path, index_col=False)
            pred_df['mhc'] = pred_df['mhc'].apply(lambda x: normalize_allele_name(x))
            keys = [pred_df.loc[i, 'mhc'] + ',' + pred_df.loc[i, 'pep'] for i in range(len(pred_df))]
            values = pred_df.to_dict(orient='records')
            self.save_to_db(keys=keys, values=values)
            self.pred_df = pd.DataFrame(matched_pmhcs + values)

        return self.pred_df

    def score_df(self) -> pd.DataFrame:
        """
        Add features from mhcflurry predictions to feature_matrix. Affinity predictions added as log values.
        All non-log value clipped to a minimum of 1e-7.
        """

        predictions = pd.DataFrame()

        alleles = list(self.pred_df.loc[:, 'mhc'].unique())
        for allele in alleles:
            df = self.pred_df.loc[self.pred_df['mhc'] == allele, :]
            df = df.iloc[pd.Series(self.peptides).map({v: i for i, v in enumerate(df['pep'])})]
            assert list(df['pep']) == list(self.peptides)
            formatted_allele = re.sub(r'[^a-zA-Z0-9-_]', '', allele)
            predictions[f'{formatted_allele}_BigMHC_ELScore'] = df['BigMHC_EL'].clip(lower=EPSILON).to_numpy()
            predictions[f'{formatted_allele}_logBigMHC_ELScore'] = np.log(df['BigMHC_EL'].clip(lower=EPSILON)).to_numpy()
        return predictions
