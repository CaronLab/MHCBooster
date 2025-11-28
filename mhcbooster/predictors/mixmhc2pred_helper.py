
import tempfile
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from mhcnames import normalize_allele_name

from mhcbooster.utils.allele import prepare_class_II_alleles
from mhcbooster.utils.constants import EPSILON
from mhcbooster.predictors.base_predictor_helper import BasePredictorHelper


class MixMhc2PredHelper(BasePredictorHelper):
    def __init__(self,
                 peptides: list[str],
                 alleles: list[str],
                 report_directory: str = None):
        super().__init__('MixMHC2pred', report_directory)

        if alleles is None or len(alleles) == 0:
            raise RuntimeError('Alleles are needed for MixMHC2pred predictions.')
        pep_lens = np.vectorize(len)(peptides)
        if np.min(pep_lens) < 8 and np.max(pep_lens) > 25:
            raise RuntimeError('MixMHC2pred cannot make predictions on peptides over length 16.')

        self.peptides = peptides
        self.alleles = self._format_class_II_alleles(alleles)
        self.mixmhc2pred_exe_path = Path(__file__).parent.parent/'third_party'/'MixMHC2pred-2.0'/'MixMHC2pred_unix'

    def _format_class_II_alleles(self, alleles: List[str]):
        avail_allele_path = Path(__file__).parent.parent/'third_party'/'MixMHC2pred-2.0'/'PWMdef'/'Alleles_list_Human.txt'
        avail_alleles = [line.split()[0].replace('__', '-').replace('_', '') \
                         for line in open(avail_allele_path).readlines() if line.startswith('D')]
        paired_alleles = prepare_class_II_alleles(alleles, avail_alleles)
        for i in range(len(paired_alleles)):
            allele = paired_alleles[i]
            allele = normalize_allele_name(allele)
            if allele.startswith('HLA-DRA1*01:01'):
                allele = allele.split('-')[-1].replace(':', '_').replace('*', '_')
            else:
                allele = '__'.join(allele.split('-')[-2:]).replace(':', '_').replace('*', '_')
            paired_alleles[i] = allele
        return paired_alleles

    def predict_df(self):
        print('Running MixMHC2pred')
        alleles_key = ','.join(self.alleles)
        keys = [alleles_key + ',' + self.peptides[i] for i in range(len(self.peptides))]
        db_data, matched_mask = self.try_load_from_db(keys=keys)
        pep_file_lines = np.array(self.peptides)[~matched_mask].tolist()
        print(f'Matched {len(db_data)} peptides from DB. Predicting on {len(pep_file_lines)} remaining peptides.')

        if len(pep_file_lines) == 0:
            self.pred_df = pd.DataFrame(db_data)
        else:
            with tempfile.NamedTemporaryFile('w', delete=False) as pep_file:
                pep_file.write('\n'.join(pep_file_lines))
                pep_file.write('\n')
                pep_file_path = pep_file.name
            with tempfile.NamedTemporaryFile('w') as results:
                results_file = results.name

            alleles = ' '.join(self.alleles)
            command = f'{self.mixmhc2pred_exe_path} -i {pep_file_path} -o {results_file} -a {alleles} --no_context'
            subprocess.run(command, shell=True)

            pred = []
            for line in open(results_file, 'r'):
                line = line.strip()
                line = line.split('\t')
                if not line or line[0].startswith('#'):
                    continue
                pred.append(line)
            pred_df = pd.DataFrame(pred[1:], columns=pred[0])

            keys = [alleles_key + ',' + pred_df.loc[i, 'Peptide'] for i in range(len(pred_df))]
            values = pred_df.to_dict(orient='records')
            self.save_to_db(keys=keys, values=values)
            self.pred_df = pd.DataFrame(db_data + values)

        return self.pred_df

    def score_df(self) -> pd.DataFrame:
        """
        Add features from mhcflurry predictions to feature_matrix. Affinity predictions added as log values.
        All non-log value clipped to a minimum of 1e-7.
        """

        self.pred_df = self.pred_df.iloc[pd.Series(self.peptides).map(
            {v: i for i, v in enumerate(self.pred_df['Peptide'])})]

        predictions = pd.DataFrame()
        assert list(self.pred_df['Peptide']) == list(self.peptides)
        alleles = list(self.pred_df.loc[:, 'BestAllele'].unique())
        for allele in alleles:
            if allele == 'NA':
                continue
            predictions[f'MixMHC2pred_{allele}_score'] = np.clip(self.pred_df[f'%Rank_{allele}'].replace('NA', 100).to_numpy(dtype=np.float32) / 100, a_min=EPSILON, a_max=None)
            predictions[f'MixMHC2pred_log{allele}_score'] = np.log(predictions[f'MixMHC2pred_{allele}_score'])
        # Best score column
        predictions['MixMHC2pred_score'] = np.clip(self.pred_df['%Rank_best'].replace('NA', 100).to_numpy(dtype=np.float32) / 100, a_min=EPSILON, a_max=None)
        predictions['MixMHC2pred_log_score'] = np.log(predictions['MixMHC2pred_score'])
        return predictions
