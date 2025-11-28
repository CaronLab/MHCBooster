
import tempfile
import subprocess

import numpy as np
import pandas as pd
from typing import List
from mhcnames import normalize_allele_name

from mhcbooster.utils.constants import EPSILON
from mhcbooster.predictors.base_predictor_helper import BasePredictorHelper


class MhcFlurryHelper(BasePredictorHelper):
    def __init__(self,
                 peptides: list[str],
                 alleles: list[str],
                 report_directory: str = None):
        super().__init__('MHCflurry', report_directory)

        if alleles is None or len(alleles) == 0:
            raise RuntimeError('Alleles are needed for MhcFlurry predictions.')
        if np.max(np.vectorize(len)(peptides)) > 16:
            raise RuntimeError('MhcFlurry cannot make predictions on peptides over length 16.')

        self.peptides = peptides
        self.alleles = self._format_class_I_alleles(alleles)

    def _format_class_I_alleles(self, alleles: List[str]):
        std_alleles = []
        for allele in set(alleles):
            try:
                std_alleles.append(normalize_allele_name(allele))
            except ValueError:
                print(f'ERROR: Allele {allele} not supported.')
        return [a.replace('*', '').replace(':', '') for a in std_alleles]

    def predict_df(self):
        # we will run MhcFlurry in a separate process so the Tensorflow space doesn't get messed up. I don't know why it
        # happens, but it does, and then either MhcFlurry or TensorFlow Decision Forests stops working.
        # I think perhaps because MhcFlurry uses some legacy code from TFv1 (I think), though this is only
        # a suspicion.
        print('Running MhcFlurry...')

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
                pep_file.write('allele,peptide\n')
                pep_file.write('\n'.join(pep_file_lines))
                pep_file.write('\n')
                pep_file_path = pep_file.name
            with tempfile.NamedTemporaryFile('w') as results:
                results_file_path = results.name

            command = f'mhcflurry-predict --out {results_file_path} {pep_file_path}'.split()
            p = subprocess.Popen(command)
            _ = p.communicate()

            pred_df = pd.read_csv(results_file_path, index_col=False)
            # pred_df['allele'] = self._format_class_I_alleles(pred_df['allele'].tolist())
            keys = [pred_df.loc[i, 'allele'] + ',' + pred_df.loc[i, 'peptide'] for i in range(len(pred_df))]
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

        alleles = list(self.pred_df.loc[:, 'allele'].unique())
        for allele in alleles:
            df = self.pred_df.loc[self.pred_df['allele'] == allele, :]
            df = df.iloc[pd.Series(self.peptides).map({v: i for i, v in enumerate(df['peptide'])})]
            assert list(df['peptide']) == list(self.peptides)
            predictions[f'{allele}_MhcFlurry_PresentationScore'] = df['mhcflurry_presentation_score'].clip(lower=EPSILON).to_numpy()
            predictions[f'{allele}_logMhcFlurry_PresentationScore'] = np.log(predictions[f'{allele}_MhcFlurry_PresentationScore'] + 0.01)
            predictions[f'{allele}_logMhcFlurry_Affinity'] = np.log(df['mhcflurry_affinity'].clip(lower=EPSILON)).to_numpy()
        df = self.pred_df.loc[self.pred_df['allele'] == alleles[0], :]
        df = df.iloc[pd.Series(self.peptides).map({v: i for i, v in enumerate(df['peptide'])})]
        # we only need one processing score column
        predictions[f'MhcFlurry_ProcessingScore'] = df['mhcflurry_processing_score'].clip(lower=EPSILON).to_numpy()
        predictions[f'logMhcFlurry_ProcessingScore'] = np.log(predictions[f'MhcFlurry_ProcessingScore'] + 0.01)
        return predictions
