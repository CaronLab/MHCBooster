import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyteomics.parser import length

from mhcbooster.utils.peptide import replace_uncommon_aas, remove_charge, remove_previous_and_next_aa, remove_modifications
from mhcbooster.predictors.netmhcpan_helper import NetMHCpanHelper


def run_netmhcpan(peptides):
    # netmhcpan = NetMHCpanHelper(peptides=peptides, alleles=ALLELES, n_threads=N_THREADS)
    netmhcpan = NetMHCpanHelper(peptides=peptides, alleles=ALLELES, mhc_class='I')
    # netmhcpan = NetMHCpanHelper(alleles=ALLELES, n_threads=N_THREADS)
    # netmhcpan.min_length = MIN_LENGTH
    # netmhcpan.max_length = MAX_LENGTH
    # netmhcpan.add_peptides(peptides)
    # netmhcpan.netmhcpan_peptides = replace_uncommon_aas(netmhcpan.peptides)
    # netmhcpan.predictions = {x: {} for x in netmhcpan.peptides}

    pred_df = netmhcpan.predict_df()
    assert len(pred_df) / len(ALLELES) == len(peptides)

    return pred_df


def draw_peptide_mer_distribution(peptides: np.ndarray, binding_status: np.ndarray, save_path: str=None) -> None:
    nAAs = np.char.str_len(peptides)
    strong_mer_list = np.zeros(MAX_LENGTH - MIN_LENGTH + 1)
    weak_mer_list = np.zeros(MAX_LENGTH - MIN_LENGTH + 1)
    non_mer_list = np.zeros(MAX_LENGTH - MIN_LENGTH + 1)
    for i in range(MAX_LENGTH - MIN_LENGTH + 1):
        strong_mer_list[i] = np.sum((nAAs == i + MIN_LENGTH) * (binding_status == 'Strong'))
        weak_mer_list[i] = np.sum((nAAs == i + MIN_LENGTH) * (binding_status == 'Weak'))
        non_mer_list[i] = np.sum((nAAs == i + MIN_LENGTH) * (binding_status == 'Non-binder'))

    plt.figure(figsize=(6, 9))
    bar_strong = plt.bar(np.arange(MIN_LENGTH, MAX_LENGTH + 1), strong_mer_list,
                         label='Strong', color='blue', alpha=0.4)
    bar_weak = plt.bar(np.arange(MIN_LENGTH, MAX_LENGTH + 1), weak_mer_list, bottom=strong_mer_list,
                       label='Weak', color='green', alpha=0.4)
    bar_non = plt.bar(np.arange(MIN_LENGTH, MAX_LENGTH + 1), non_mer_list, bottom=strong_mer_list+weak_mer_list,
                      label='Non-binder', color='orange', alpha=0.4)
    plt.xlabel('N mers')
    plt.ylabel('N peptides')
    plt.legend()

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                int(height),  # value
                ha='center',  # Horizontal alignment
                va='center'  # Vertical alignment
            )

    add_value_labels(bar_strong)
    add_value_labels(bar_weak)
    add_value_labels(bar_non)

    if save_path:
        plt.savefig(save_path)
        plt.clf()
        plt.close()
    else:
        plt.show()


def _base_eval_pep(folder, pep_file_suffix, sep, pep_col, qvalue_col, label_col=None, target_label=None, mod_col=None):
    paths = list(Path(folder).rglob(pattern='*'+pep_file_suffix))
    paths.sort()
    result_stat = pd.DataFrame(columns=['File name', 'Peptides', 'Peptides (filtered)', 'Strong binders', 'Weak binders', 'Non-binders', 'Binder percent'])
    for i, path in enumerate(paths):
        if path.name == 'combined_peptide.tsv':
            continue
        filename = path.stem
        if length(list(open(path))) == 0:
            pep_df = []
        else:
            pep_df = pd.read_csv(path, sep=sep)
            if label_col:
                pep_df = pep_df[pep_df[label_col] == target_label]
            if qvalue_col:
                pep_df = pep_df[pep_df[qvalue_col] <= 0.01]

        if len(pep_df) == 0:
            print(f'{filename}\tPeptides:0\tPeptides (filtered):0\tStrong:0\tWeak:0\tNon-binder:0\tBinder percent:Nan')
            result_stat.loc[i] = [filename, 0, 0, 0, 0, 0, 'Nan']
            continue

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
        pep_filtered = pep_filtered[(np.char.str_len(pep_filtered) >= MIN_LENGTH) * (np.char.str_len(pep_filtered) <= MAX_LENGTH)]

        netmhcpan_pred = run_netmhcpan(peptides=pep_filtered)
        netmhcpan_pred.to_csv(str(path).replace(pep_file_suffix, '.netmhcpan.tsv'), sep='\t', index=False)

        bind_table = pd.DataFrame(columns=['Peptide'])
        for j, allele in enumerate(ALLELES):
            allele_df = netmhcpan_pred.loc[netmhcpan_pred['Allele'] == allele, ['Peptide', 'Binder']]
            if j != 0:
                allele_df = allele_df.drop_duplicates(subset=['Peptide'])
            bind_table = bind_table.merge(allele_df, on='Peptide', how='outer')
            bind_table.rename(columns={'Binder': allele}, inplace=True)

        bind_table['Binder'] = ''
        for j in range(len(bind_table)):
            if 'Strong' in bind_table.iloc[j, 1:].values:
                bind_table.loc[j, 'Binder'] = 'Strong'
            elif 'Weak' in bind_table.iloc[j, 1:].values:
                bind_table.loc[j, 'Binder'] = 'Weak'
            else:
                bind_table.loc[j, 'Binder'] = 'Non-binder'
        bind_table = bind_table[bind_table.columns[:1].append(bind_table.columns[-1:]).append(bind_table.columns[1:-1])]

        bind_table.to_csv(str(path).replace(pep_file_suffix, '.netmhcpan_status.tsv'), sep='\t', index=False)
        n_strong = np.sum(bind_table['Binder'] == 'Strong')
        n_weak = np.sum(bind_table['Binder'] == 'Weak')
        n_non = np.sum(bind_table['Binder'] == 'Non-binder')
        binder_percent = round((n_strong + n_weak) * 100.0 / len(bind_table), 1)

        draw_peptide_mer_distribution(peptides=bind_table['Peptide'].values.astype(str),
                                      binding_status=bind_table['Binder'].values.astype(str),
                                      save_path=str(path).replace(pep_file_suffix,  '.netmhcpan_mer_ba_distribution.png'))

        print(f'{filename}\tPeptides:{len(peptides)}\tPeptides (filtered):{len(pep_filtered)}\tStrong:{n_strong}\tWeak:{n_weak}\tNon-binder:{n_non}\tBinder percent:{binder_percent}')
        result_stat.loc[i] = [filename, len(peptides), len(pep_filtered), n_strong, n_weak, n_non, binder_percent]
    result_stat.to_csv(os.path.join(folder, 'netmhcpan_result_stats.tsv'), sep='\t', index=False)


def eval_mokapot(folder):
    _base_eval_pep(folder, pep_file_suffix='.mokapot.peptides.txt', sep='\t', pep_col='Peptide', qvalue_col='mokapot q-value', label_col='Label', target_label=True, mod_col=None)

def eval_midiaid(folder):
    _base_eval_pep(folder, pep_file_suffix='.mokapot.peptides.txt', sep='\t', pep_col='peptide', qvalue_col='mokapot q-value', label_col='is_target', target_label=True, mod_col=None)

def eval_percolator(folder):
    paths = Path(folder).rglob(pattern='*' + '_pep_target.pout')
    for i, path in enumerate(paths):
        # Compatibility for multi-prot
        with open(path, 'r') as file:
            content = file.read().replace(';\t', ';')
        with open(str(path) + '.m', 'w') as file:
            file.write(content)
    _base_eval_pep(folder, pep_file_suffix='_pep_target.pout.m', sep='\t', pep_col='peptide', qvalue_col='q-value', label_col=None, mod_col=None)

def eval_philosopher(folder):
    _base_eval_pep(folder, pep_file_suffix='_peptide.tsv', sep='\t', pep_col='Peptide', qvalue_col=None, label_col=None, mod_col='Assigned Modifications')

def eval_ms2rescore(folder):
    _base_eval_pep(folder, pep_file_suffix='_result.csv', sep=',', pep_col='peptidoform', qvalue_col='peptide_qvalue', label_col='is_decoy', target_label=False, mod_col=None)


def eval_mhcvalidator(folder):
    _base_eval_pep(folder, pep_file_suffix='.MhcValidator_annotated.tsv', sep='\t', pep_col='Peptide', qvalue_col='mhcv_pep-level_q-value', label_col='mhcv_label', target_label=1, mod_col=None)

def eval_mhcbooster(folder):
    _base_eval_pep(folder, pep_file_suffix='peptide.tsv', sep='\t', pep_col='Peptide', qvalue_col='Pep Qvalue', label_col='Label', target_label='Target', mod_col=None)


if __name__ == '__main__':

    N_THREADS = os.cpu_count() // 2

    ALLELES = ['HLA-A0201', 'HLA-B0702', 'HLA-C0702']
    # ALLELES = ['HLA-A0220','HLA-A6801','HLA-B3503','HLA-B3901','HLA-C0401','HLA-C0702']
    # ALLELES = ['HLA-A0101', 'HLA-A0202', 'HLA-B5701', 'HLA-B4403', 'HLA-C0602', 'HLA-C1602']
    # ALLELES = ['HLA-A0101', 'HLA-A2415', 'HLA-B5701', 'HLA-C0602']
    # ALLELES = ['HLA-A0301', 'HLA-A6802', 'HLA-B0702', 'HLA-B1402', 'HLA-C0702', 'HLA-C0802']

    MIN_LENGTH = 8
    MAX_LENGTH = 14
    # MIN_LENGTH = 9
    # MAX_LENGTH = 25

    # eval_mokapot(folder='/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M_rerun/msfragger/mokapot')
    # eval_percolator(folder='/mnt/f/JY_1_2_5M/percolator')
    # eval_ms2rescore(folder='/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M_rerun/msfragger/ms2rescore')
    # eval_percolator(folder='/mnt/f/JY_1_2_5M/fragpipe')
    # eval_mhcvalidator(folder='/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M_rerun/msfragger/mhcvalidator')
    # eval_mhcbooster(folder='/mnt/f/JY_1_2_5M/mhcbooster')
    eval_percolator(folder='/mnt/f/directDIA/diatracer/percolator')
    eval_percolator(folder='/mnt/f/directDIA/diatracer/fragpipe')
    eval_mhcbooster(folder='/mnt/f/directDIA/diatracer/mhcbooster')
    # eval_midiaid(folder='/mnt/f/MIDIA_test/midiaid')

