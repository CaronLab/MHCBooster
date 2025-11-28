import os
import random
import tempfile
import mhcgnomes
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List
from itertools import islice
from uuid import uuid4
from mhcnames import normalize_allele_name

from mhcbooster.utils.allele import prepare_class_II_alleles, prepare_class_I_alleles
from mhcbooster.utils.peptide import remove_previous_and_next_aa, remove_modifications, replace_uncommon_aas
from mhcbooster.utils.job import Job
from tqdm.contrib.concurrent import process_map

from mhcbooster.utils.constants import EPSILON
from mhcbooster.predictors.base_predictor_helper import BasePredictorHelper

TMP_DIR = str(Path(tempfile.gettempdir(), 'pynetmhcpan').expanduser())
NETMHCPAN = Path(__file__).parent.parent / 'third_party' / 'netMHCpan-4.1' / 'netMHCpan'
NETMHCIIPAN = Path(__file__).parent.parent / 'third_party' / 'netMHCIIpan-4.3' / 'netMHCIIpan'


class NetMHCpanHelper(BasePredictorHelper):
    """
    example usage:
    cl_tools.make_binding_prediction_jobs()
    cl_tools.run_jubs()
    cl_tools.aggregate_netmhcpan_results()
    cl_tools.clear_jobs()
    """
    def __init__(self,
                 peptides: List[str] = None,
                 alleles: List[str] = None,
                 mhc_class: str = 'I',
                 n_threads: int = 0,
                 report_directory: str = '',
                 tmp_dir: str = TMP_DIR,
                 output_dir: str = None):
                 #netmhcpan_path: str = 'netMHCpan'):
        """
        Helper class to run NetMHCpan on multiple CPUs from Python. Can annotated a file with peptides in it.
        """
        assert mhc_class in ['I', 'II'], 'NetMHCpanHelper mhc_class must be I or II.'
        if mhc_class == 'I':
            super().__init__('NetMHCpan', report_directory)
        else:
            super().__init__('NetMHCIIpan', report_directory)
        if alleles is None or len(alleles) == 0:
            raise RuntimeError('Alleles are needed for NetMHCpan predictions.')

        if mhc_class == 'I':
            self.alleles = self._format_class_I_alleles(alleles)
            self.min_length = 8
        else:
            self.alleles = self._format_class_II_alleles(alleles)
            self.min_length = 9

        self.peptides = []
        if peptides is not None:
            self.add_peptides(peptides)
            self.netmhcpan_peptides = replace_uncommon_aas(self.peptides)
            self.reverse_lookup = {value: key for key, value in self.netmhcpan_peptides.items()}
        else:
            self.netmhcpan_peptides = dict()
        self.predictions = {x: {} for x in self.peptides}
        self.wd = Path(output_dir) if output_dir else Path(os.getcwd())
        self.temp_dir = Path(tmp_dir) / 'PyNetMHCpan'
        if self.wd and not self.wd.exists():
            self.wd.mkdir(parents=True)
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True)
        self.predictions_made = False
        self.not_enough_peptides = []
        if n_threads < 1 or n_threads > os.cpu_count():
            self.n_threads = os.cpu_count()
        else:
            self.n_threads = n_threads
        self.jobs = []
        # self.add_peptides(peptides)
        self.mhc_class: str = mhc_class

    def add_peptides(self, peptides: List[str]):
        peptides = remove_previous_and_next_aa(peptides)
        peptides = remove_modifications(peptides)
        for p in peptides:
            if len(p) < self.min_length:
                raise ValueError(f"One or more peptides is shorter than the minimum length of {self.min_length} mers")
        self.peptides += peptides

        self.predictions = {pep: {} for pep in self.peptides}

    def _format_class_I_alleles(self, alleles: List[str]):
        avail_allele_path = Path(__file__).parent.parent/'third_party'/'netMHCpan-4.1'/'Linux_x86_64'/'data'/'MHC_pseudo.dat'
        avail_alleles = [line.split()[0].replace(':', '') for line in open(avail_allele_path).readlines()]

        avail_alleles = [mhcgnomes.parse(allele).to_string() for allele in avail_alleles]
        std_alleles = prepare_class_I_alleles(alleles, avail_alleles)
        return [a.replace('*', '').replace(':', '') for a in std_alleles]

    def _format_class_II_alleles(self, alleles: List[str]):
        avail_allele_path = Path(__file__).parent.parent/'third_party'/'netMHCIIpan-4.3'/'data'/'pseudosequence.2023.all.X.dat'
        avail_alleles = [line.split()[0].replace('_', '') for line in open(avail_allele_path).readlines()]
        paired_alleles = prepare_class_II_alleles(alleles, avail_alleles)
        for i in range(len(paired_alleles)):
            allele = paired_alleles[i]
            allele = normalize_allele_name(allele)
            if allele.startswith('HLA-DRA1*01:01'):
                allele = allele.split('-')[-1].replace(':', '').replace('*', '_')
            else:
                allele = allele.replace(':', '').replace('*', '')
            paired_alleles[i] = allele
        return paired_alleles

    def _make_binding_prediction_jobs(self):
        if not self.peptides:
            print("ERROR: You need to add some peptides first!")
            return
        self.jobs = []

        # split peptide list into chunks
        if self.netmhcpan_peptides:
            peptides = np.unique(list(self.netmhcpan_peptides.values()))
        else:
            peptides = np.unique(self.peptides)
        np.random.shuffle(peptides)  # shuffle to speed up

        job_number = 1
        for allele in self.alleles:
            keys = [allele + ',' + pep for pep in peptides]
            db_data, matched_mask = self.try_load_from_db(keys=keys)
            if len(db_data) > 0:
                for i, pep in enumerate(peptides[matched_mask]):
                    self.predictions[self.reverse_lookup[pep]][allele] = db_data[i]
            if len(db_data) == len(keys):
                continue

            unmatched_peptides = peptides[~matched_mask]
            if len(unmatched_peptides) > 500:
                peptide_iter = iter(unmatched_peptides)
                chunks = list(iter(lambda: tuple(islice(peptide_iter, 500)), ()))
            else:
                chunks = [unmatched_peptides.tolist()]
            print(f'Peptide list broken into {len(chunks)} chunks.')

            for chunk in chunks:
                if len(chunk) < 1:
                    continue
                fname = Path(self.temp_dir, f'peplist_{job_number}.csv')
                # save the new peptide list, this will be given to netMHCpan
                with open(str(fname), 'w') as f:
                    f.write('\n'.join(chunk))
                # run netMHCpan
                if self.mhc_class == 'I':
                    command = f'{NETMHCPAN} -p -f {fname} -a {allele} -BA'.split(' ')
                else:
                    command = f'{NETMHCIIPAN} -inptype 1 -f {fname} -a {allele} -BA'.split(' ')

                job = Job(command=command, working_directory=self.temp_dir)
                self.jobs.append(job)
                job_number += 1

    @staticmethod
    def _run_job(job: Job):
        job.run()
        return job

    def _run_jobs(self):
        self.jobs = process_map(self._run_job, self.jobs, max_workers=self.n_threads, chunksize=1)
        for job in self.jobs:
            if job.returncode != 0:
                raise ChildProcessError(f'{job.stdout.decode()}\n\n{job.stderr.decode()}')
            out = (job.stdout.decode() + job.stderr.decode()).split('\n')
            if 'error' in (' '.join(out[-5:])).lower():
                raise ChildProcessError(f'{job.stdout.decode()}\n\n{job.stderr.decode()}')

    def _parse_netmhc_output(self, stdout: str):
        lines = stdout.split('\n')
        if self.mhc_class == 'I':
            allele_idx = 1
            peptide_idx = 2
            el_score_idx = 11
            el_rank_idx = 12
            aff_score_idx = 13
            aff_rank_idx = 14
            aff_nM_idx = 15
            strong_cutoff = 0.5
            weak_cutoff = 2.0
        else:
            allele_idx = 1
            peptide_idx = 2
            el_score_idx = 8
            el_rank_idx = 9
            aff_score_idx = 11
            aff_nM_idx = 13
            aff_rank_idx = 12
            strong_cutoff = 2.0
            weak_cutoff = 10.0

        keys = []
        values = []
        for line in lines:
            line = line.strip()
            line = line.split()
            if not line or line[0] == '#' or not line[0].isnumeric():
                continue
            allele = line[allele_idx].replace('*', '').replace(':', '')
            peptide = line[peptide_idx]
            el_rank = float(line[el_rank_idx])
            el_score = float(line[el_score_idx])
            aff_rank = float(line[aff_rank_idx])
            aff_score = float(line[aff_score_idx])
            aff_nM  = float(line[aff_nM_idx])

            if float(el_rank) <= strong_cutoff:
                binder = 'Strong'
            elif float(el_rank) <= weak_cutoff:
                binder = 'Weak'
            else:
                binder = 'Non-binder'

            key = allele + ',' + peptide
            value = {'el_rank': el_rank, 'el_score': el_score, 'aff_rank': aff_rank, 'aff_score': aff_score,
                     'aff_nM': aff_nM, 'binder': binder}
            self.predictions[self.reverse_lookup[peptide]][allele] = value
            keys.append(key)
            values.append(value)
        self.save_to_db(keys=keys, values=values)


    def _aggregate_netmhcpan_results(self):
        for job in self.jobs:
            if job.returncode != 0:
                print(job.stdout.decode())
                print(job.stderr.decode())
                print('ERROR: There was a problem in NetMHCpan. See the above about for possible information.')
                exit(1)
            self._parse_netmhc_output(job.stdout.decode())

    def _clear_jobs(self):
        self.jobs = []

    def make_predictions(self):
        self.temp_dir = self.temp_dir / str(uuid4())
        self.temp_dir.mkdir(parents=True)
        self._make_binding_prediction_jobs()
        self._run_jobs()
        self._aggregate_netmhcpan_results()
        self._clear_jobs()

    def predict_df(self):
        self.make_predictions()
        df_columns = ['Peptide', 'Allele', 'EL_score', 'EL_Rank', 'Aff_Score', 'Aff_Rank', 'Aff_nM', 'Binder']
        data = []
        for allele in self.alleles:
            for pep in self.peptides:
                # netmhc_pep = self.netmhcpan_peptides[pep]
                data.append([pep,
                             allele,
                             self.predictions[pep][allele]['el_score'],
                             self.predictions[pep][allele]['el_rank'],
                             self.predictions[pep][allele]['aff_score'],
                             self.predictions[pep][allele]['aff_rank'],
                             self.predictions[pep][allele]['aff_nM'],
                             self.predictions[pep][allele]['binder']])
        self.pred_df = pd.DataFrame(data=data, columns=df_columns)
        return self.pred_df

    def score_df(self) -> pd.DataFrame:
        predictions = pd.DataFrame()
        alleles = list(self.pred_df.loc[:, 'Allele'].unique())
        for allele in alleles:
            df = self.pred_df.loc[self.pred_df['Allele'] == allele, :]
            predictions[f'{allele}_NetMHCpan_Aff_Score'] = df['Aff_Score'].clip(lower=EPSILON).to_numpy()
            predictions[f'{allele}_logNetMHCpan_Aff_Score'] = np.log(predictions[f'{allele}_NetMHCpan_Aff_Score'] + 0.01)
            predictions[f'{allele}_NetMHCpan_EL_score'] = df['EL_score'].clip(lower=EPSILON).to_numpy()
            predictions[f'{allele}_logNetMHCpan_Aff_nM'] = np.log(df['Aff_nM'].clip(lower=EPSILON).to_numpy())

        return predictions

    def format_pred_result_for_saving(self) -> pd.DataFrame:
        """
        convert the netmhcpan predictions to a wide-format dataframe and add columns for "best" predictions per peptide.
        :return:
        """
        def binder_type(x):
            x = list(x)
            if 'Strong' in x:
                return 'Strong'
            elif 'Weak' in x:
                return 'Weak'
            else:
                return 'Non-binder'

        predictions = pd.DataFrame()
        predictions['Peptide'] = np.array(self.peptides)
        alleles = list(self.pred_df.loc[:, 'Allele'].unique())
        for allele in alleles:
            df = self.pred_df.loc[self.pred_df['Allele'] == allele, :].copy(deep=True)
            assert list(df['Peptide']) == list(self.peptides)
            for pred in df.columns:
                if pred in ['Allele', 'Peptide', 'Binder']:
                    continue
                predictions[f'{allele}_NetMHCpan_{pred}'] = df[pred].to_numpy()
            predictions[f'{allele}_NetMHCpan_Binder'] = df['Binder'].to_numpy()
            predictions[f'{allele}_logNetMHCpan_Aff_nM'] = np.log(predictions[f'{allele}_NetMHCpan_Aff_nM'].to_numpy())
        binder_columns = [x for x in predictions.columns if 'Binder' in x]
        predictions['Binder'] = predictions.loc[:, binder_columns].apply(binder_type, axis=1).to_numpy()

        return predictions