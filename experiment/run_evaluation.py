import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mhcbooster.utils.peptide import replace_uncommon_aas, remove_charge, remove_previous_and_next_aa, remove_modifications
from mhcbooster.predictors.netmhcpan_helper import NetMHCpanHelper


class Evaluation:
    def __init__(self, dataset_name, min_len, max_len):
        self.result_folder = Path(__file__).parent/dataset_name
        self.min_len = min_len
        self.max_len = max_len
        self.sequences = set()

    def base_eval_pep(self, folder, pep_file_suffix, sep, pep_col, qvalue_col, label_col=None, target_label=None, mod_col=None):
        paths = list(Path(folder).rglob(pattern='*' + pep_file_suffix))
        paths.sort()

        result_stat = pd.DataFrame(columns=['File name', 'Peptides', 'Peptides (filtered)'])
        for i, path in enumerate(paths):
            if 'combined' in str(path):
                continue
            if path.name == pep_file_suffix:
                filename = path.parent.name
            else:
                filename = path.name.replace(pep_file_suffix, '')
            filename = filename.replace('_MHCBooster', '').replace('_MhcValidator', '').replace('_edited', '')
            if len(list(open(path))) == 0:
                result_stat.loc[i] = [filename, 0, 0]
                continue

            pep_df = pd.read_csv(path, sep=sep)
            if label_col and target_label:
                pep_df = pep_df[pep_df[label_col] == target_label]
            if qvalue_col:
                pep_df = pep_df[pep_df[qvalue_col] <= FDR]

            # prot_col = 'proteinIds' if 'proteinIds' in pep_df.columns else 'protein' if 'protein' in pep_df.columns else 'Proteins'
            # pep_df = pep_df[pep_df[prot_col].str.contains('EBV') & ~pep_df[prot_col].str.contains('HUMAN')]

            peptides = pep_df[pep_col].to_numpy()
            peptides = remove_previous_and_next_aa(peptides)
            peptides = remove_charge(peptides)
            if mod_col:
                pep_mod_keys = np.char.add(peptides.astype(str), pep_df[mod_col].fillna('').to_numpy().astype(str))
                unique_indices = np.unique(pep_mod_keys, return_index=True)[1]
                peptides = peptides[unique_indices]
            else:
                peptides = np.unique(peptides)

            peptides = remove_modifications(peptides)
            pep_filtered = np.array(peptides)
            if len(pep_filtered) > 0:
                pep_filtered = pep_filtered[
                    (np.char.str_len(pep_filtered) >= self.min_len) * (np.char.str_len(pep_filtered) <= self.max_len)]
            result_stat.loc[i] = [filename, len(pep_filtered), len(np.unique(pep_filtered))]
            self.sequences.update(pep_filtered)
        print(len(self.sequences))
        print(','.join(self.sequences))
        self.sequences.clear()
        return result_stat

    def eval_mokapot(self):
        mokapot_folder = self.result_folder/'mokapot'
        if mokapot_folder.exists():
            result_df = self.base_eval_pep(mokapot_folder, pep_file_suffix='.mokapot.peptides.txt', sep='\t', pep_col='Peptide',
                                      qvalue_col='mokapot q-value', label_col='Label', target_label=True, mod_col=None)
            result_df.columns = ['File name', 'mokapot_pep', 'mokapot_seq']
            return result_df

    def eval_percolator(self):
        percolator_folder = self.result_folder/'percolator'
        if percolator_folder.exists():
            paths = Path(percolator_folder).rglob(pattern='*' + '_pep_target.pout')
            for i, path in enumerate(paths):
                # Compatibility for multi-prot
                with open(path, 'r') as file:
                    content = file.read().replace(';\t', ';')
                with open(str(path) + '.m', 'w') as file:
                    file.write(content)
            result_df = self.base_eval_pep(percolator_folder, pep_file_suffix='_pep_target.pout.m', sep='\t', pep_col='peptide',
                                      qvalue_col='q-value', label_col=None, mod_col=None)
            result_df.columns = ['File name', 'percolator_pep', 'percolator_seq']
            return result_df

    def eval_fragpipe(self):
        fragpipe_folder = self.result_folder/'fragpipe'
        if fragpipe_folder.exists():
            paths = Path(fragpipe_folder).rglob(pattern='*' + '_pep_target.pout')
            for i, path in enumerate(paths):
                # Compatibility for multi-prot
                with open(path, 'r') as file:
                    content = file.read().replace(';\t', ';')
                with open(str(path) + '.m', 'w') as file:
                    file.write(content)
            result_df = self.base_eval_pep(fragpipe_folder, pep_file_suffix='_pep_target.pout.m', sep='\t', pep_col='peptide',
                                      qvalue_col='q-value', label_col=None, mod_col=None)
            result_df.columns = ['File name', 'fragpipe_pep', 'fragpipe_seq']
            return result_df

    def eval_philosopher(self):
        philosopher_folder = self.result_folder/'philosopher'
        if philosopher_folder.exists():
            result_df = self.base_eval_pep(philosopher_folder, pep_file_suffix='_peptide.tsv', sep='\t', pep_col='Peptide',
                                      qvalue_col=None, label_col=None, mod_col='Assigned Modifications')
            result_df.columns = ['File name', 'philosopher_pep', 'philosopher_seq']
            return result_df

    def eval_ms2rescore(self):
        ms2rescore_folder = self.result_folder/'ms2rescore'
        if ms2rescore_folder.exists():
            result_df = self.base_eval_pep(ms2rescore_folder, pep_file_suffix='_result.csv', sep=',', pep_col='peptidoform',
                                           qvalue_col='peptide_qvalue', label_col='is_decoy', target_label=False, mod_col=None)
            result_df.columns = ['File name', 'ms2rescore_pep', 'ms2rescore_seq']
            return result_df

    def eval_mhcbooster_old(self):
        mhcbooster_folder = self.result_folder/'mhcbooster_old'
        if mhcbooster_folder.exists():
            result_df = self.base_eval_pep(mhcbooster_folder, pep_file_suffix='.MhcValidator_annotated.tsv', sep='\t', pep_col='Peptide',
                                      qvalue_col='mhcv_pep-level_q-value', label_col='mhcv_label', target_label=1, mod_col=None)
            result_df.columns = ['File name', 'mhcbooster_pep', 'mhcbooster_seq']
            return result_df

    def eval_mhcvalidator(self):
        mhcbooster_folder = self.result_folder/'mhcvalidator'
        if mhcbooster_folder.exists():
            result_df = self.base_eval_pep(mhcbooster_folder, pep_file_suffix='.MhcValidator_annotated.tsv', sep='\t', pep_col='Peptide',
                                      qvalue_col='mhcv_pep-level_q-value', label_col='mhcv_label', target_label=1, mod_col=None)
            result_df.columns = ['File name', 'mhcvalidator_pep', 'mhcvalidator_seq']
            return result_df

    def eval_mhcbooster(self, folder='mhcbooster'):
        mhcbooster_folder = self.result_folder / folder
        if mhcbooster_folder.exists():
            result_df = self.base_eval_pep(mhcbooster_folder, pep_file_suffix='peptide.tsv', sep='\t',
                                           pep_col='peptide',
                                           qvalue_col='pep_qvalue', label_col='label', target_label='Target',
                                           mod_col=None)
            result_df.columns = ['File name', f'{folder}_pep', f'{folder}_seq']
            return result_df

    def run(self):
        result_dfs = []
        result_dfs.append(self.eval_percolator())
        # philosopher_df = self.eval_philosopher()
        # mokapot_df = self.eval_mokapot()
        # ms2rescore_df = self.eval_ms2rescore()
        result_dfs.append(self.eval_fragpipe())
        # result_dfs.append(self.eval_mhcvalidator())
        # mhcbooster_df = self.eval_mhcbooster_old()
        # if mhcbooster_df is None or mhcbooster_df.empty:
        #     mhcbooster_df = self.eval_mhcbooster()
        # result_dfs.append(self.eval_mhcbooster(folder='mhcvalidator'))
        result_dfs.append(self.eval_mhcbooster(folder='mhcbooster'))
        # result_dfs.append(self.eval_mhcbooster(folder='mhcbooster_nomhc'))

        result_df = pd.DataFrame()
        for df in result_dfs:
            if df is not None:
                result_df = pd.merge(result_df, df, on='File name', how='outer') if not result_df.empty else df
        # result_df.to_csv(os.path.join(self.result_folder, 'result_stats.tsv'), sep='\t', index=False)
        result_df.to_csv(os.path.join(self.result_folder, 'result_stats.tsv'), sep='\t', index=False)
        print(result_df)

if __name__ == '__main__':
    FDR = 0.01
    # evaluation = Evaluation('JY_low_microIP_I', min_len=8, max_len=14)
    # evaluation = Evaluation('RA_Fractionation', min_len=8, max_len=15)
    # evaluation = Evaluation('JY_100K', min_len=8, max_len=15)
    evaluation = Evaluation('JY_1_10_25M_rerun/msfragger', min_len=8, max_len=14)
    # evaluation.result_folder = Path('/mnt/d/data/JY_500M/old')
    # evaluation.result_folder = Path('/mnt/d/data/JY_500M/new')
    # evaluation.result_folder = Path('/mnt/d/data/JY100M_Val_DDA_102824')
    # evaluation = Evaluation('JY_Fractionation', min_len=8, max_len=15)
    # evaluation.result_folder = Path('/mnt/d/data/JY_PC-9_50M_ClassI_MS_DDA')
    # evaluation.result_folder = Path('/mnt/d/data/JY_EL4_Class1_DDA_SK_MS_013125')
    # evaluation = Evaluation('RA_Fractionation', min_len=8, max_len=15)
    # evaluation.min_len = 12
    # evaluation.max_len = 21
    # evaluation.result_folder = Path('/mnt/e/data/BigIP_JY_500M_HLA-I')
    evaluation.result_folder = Path('/mnt/e/data/PXD058436')
    # evaluation.eval_percolator()
    # evaluation.eval_fragpipe()
    # evaluation.eval_mhcbooster()
    evaluation.run()
    # print(evaluation.eval_mhcbooster('test_alpha2_rt'))
    # print(evaluation.eval_mhcbooster('test_alpha2_rt_re'))
    # evaluation = Evaluation('PXD019643/HLA-I', min_len=8, max_len=15)
    # evaluation.run()
    # evaluation = Evaluation('RA_Fractionation_Replicate_1', min_len=8, max_len=15)
    # evaluation.run()

    # result_dfs = []
    # for mhcbooster_folder in evaluation.result_folder.iterdir():
    #     folder_name = mhcbooster_folder.name
    #     result_df = evaluation.base_eval_pep(mhcbooster_folder, pep_file_suffix='peptide.tsv', sep='\t',
    #                                          pep_col='peptide',
    #                                          qvalue_col='pep_qvalue', label_col='label', target_label='Target',
    #                                          mod_col=None)
    #     result_df.columns = ['File name', f'{folder_name}_pep', f'{folder_name}_seq']
    #     result_dfs.append(result_df)
    # 
    # result_df = pd.DataFrame()
    # for df in result_dfs:
    #     if df is not None:
    #         result_df = pd.merge(result_df, df, on='File name', how='outer') if not result_df.empty else df
    # result_df.to_csv(os.path.join(evaluation.result_folder, 'result_stats.tsv'), sep='\t', index=False)
    # print(result_df)
