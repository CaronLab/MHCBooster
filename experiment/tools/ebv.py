import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from mhcbooster.utils.peptide import replace_uncommon_aas, remove_charge, remove_previous_and_next_aa, remove_modifications


def get_gene_map(fasta_path, prot_tag):
    gene_map = {}
    description_map = {}
    with open(fasta_path, 'r') as file:
        for line in file:
            if not line.startswith('>'):
                continue
            if line.startswith('>rev'):
                continue
            line_split = line.split(' ')
            prot_name = line.split(' ')[0][1:]
            if prot_tag not in prot_name:
                continue
            gene_name = ''
            for s in line_split:
                if s.startswith('GN='):
                    gene_name = s.split('=')[1]
                    break
            gene_map[prot_name] = gene_name
            description = []
            for i in range(1, len(line_split)):
                if line_split[i].startswith('OS='):
                    break
                description.append(line_split[i])
            description = ' '.join(description)
            if gene_name in description_map and len(description_map[gene_name]) > len(description):
                continue
            description_map[gene_name] = description
    return gene_map, description_map


def get_identified(pep_dfs, q_col, pep_col, prot_col, fdr, min_len, max_len, label_col=None, target_label=None):
    peptide_df = pd.DataFrame(columns=['Peptide', 'Proteins', 'Q-value'])
    for pep_df in pep_dfs:
        if label_col is not None:
            pep_df = pep_df[pep_df[label_col] == target_label]

        identified_df = pep_df[pep_df[q_col] <= fdr]
        if len(identified_df) == 0:
            continue

        peptides = identified_df[pep_col].to_numpy()
        proteins = identified_df[prot_col].to_numpy()
        q_value = identified_df[q_col].to_numpy()
        if 'mapped_protein' in identified_df.columns:
            all_proteins = []
            for p, m in zip(proteins, identified_df['mapped_protein']):
                if p.startswith('rev'):
                    p = ''
                if pd.isna(m):
                    all_proteins.append(p)
                else:
                    ms = m.split(';')
                    ms = [m for m in ms if not m.startswith('rev')]
                    all_proteins.append(p + ';' + ';'.join(ms))
            proteins = np.array(all_proteins)

        peptides = remove_previous_and_next_aa(peptides)
        peptides = remove_charge(peptides)
        peptides = remove_modifications(peptides)
        peptides, indices = np.unique(peptides, return_index=True)
        proteins = proteins[indices]
        q_value = q_value[indices]

        mask = (np.char.str_len(peptides) >= min_len) * (np.char.str_len(peptides) <= max_len)
        peptides = peptides[mask]
        proteins = proteins[mask]
        q_value = q_value[mask]

        identified_df = pd.DataFrame({'Peptide': peptides, 'Proteins': proteins, 'Q-value': q_value})
        peptide_df = pd.concat([peptide_df, identified_df], ignore_index=True)
    peptide_df = peptide_df.groupby('Peptide', as_index=False).agg({'Proteins': 'first', 'Q-value': 'min'})
    return peptide_df

def fill_identified_psm_count(peptide_df, psm_dfs, q_col, pep_col, fdr, min_len, max_len, label_col=None, target_label=None):
    peptide_df['PSM_Count'] = 0
    for psm_df in psm_dfs:
        if label_col is not None:
            psm_df = psm_df[psm_df[label_col] == target_label]
        psm_df = psm_df[psm_df[q_col] <= fdr].copy()
        if len(psm_df) == 0:
            continue
        psm_df['Peptide'] = remove_modifications(remove_charge(remove_previous_and_next_aa(psm_df[pep_col].to_numpy())))

        mask = psm_df['Peptide'].str.len().between(min_len, max_len)
        psm_df = psm_df.loc[mask]

        ebv_psm_count_df = psm_df.groupby('Peptide').size().reset_index(name='PSM_Count')
        peptide_df =  peptide_df.merge(ebv_psm_count_df[['Peptide', 'PSM_Count']], on='Peptide',
                                       how='left', suffixes=('', '_add'))
        if 'PSM_Count_add' in peptide_df.columns:
            peptide_df['PSM_Count'] = peptide_df['PSM_Count'] + peptide_df['PSM_Count_add'].fillna(0).astype(int)
            peptide_df = peptide_df.drop('PSM_Count_add', axis=1)

    return peptide_df


def get_binders(mhcbooster_folders):
    binders = pd.DataFrame(columns=['Peptide', 'EL_Rank', 'Allele'])
    for mhcbooster_folder in mhcbooster_folders:
        netmhcpan_paths = Path(mhcbooster_folder).rglob('app_prediction.netmhcpan.tsv')
        for path in netmhcpan_paths:
            df = pd.read_csv(path, sep='\t')
            df = df[['Peptide', 'EL_Rank', 'Allele']]

            # Get best allele for each peptide based on minimum EL_Rank
            best_alleles = df.loc[df.groupby('Peptide')['EL_Rank'].idxmin()]
            best_alleles = best_alleles[['Peptide', 'EL_Rank', 'Allele']]
            best_alleles.columns = ['Peptide', 'EL_Rank', 'Best_Allele']

            # Get all alleles with EL_Rank <= 2
            binding_alleles = df[df['EL_Rank'] <= 2].groupby('Peptide')['Allele'].agg(
                lambda x: ','.join(sorted(set(x)))).reset_index()
            binding_alleles.columns = ['Peptide', 'Binding_Alleles']

            # Combine best allele with all binding alleles
            df = best_alleles.merge(binding_alleles, on='Peptide', how='left')
            binders = pd.concat([binders, df], ignore_index=True)

        binders = binders.groupby('Peptide').agg({
            'EL_Rank': 'first',
            'Best_Allele': 'first',
            'Binding_Alleles': 'first'
        }).reset_index()

    binders = binders.groupby('Peptide').agg({
        'EL_Rank': 'first',
        'Best_Allele': 'first',
        'Binding_Alleles': 'first'
    }).reset_index()
    binders['Binder'] = 'Non-binder'
    binders.loc[binders['EL_Rank'] <= 2, 'Binder'] = 'Weak'
    binders.loc[binders['EL_Rank'] <= 0.5, 'Binder'] = 'Strong'
    binders = binders[['Peptide', 'EL_Rank', 'Binder', 'Best_Allele', 'Binding_Alleles']]
    return binders

def search_best_scores(peptide_df, psm_dfs, pep_col, rt_col, msms_col):
    peptide_df['rt_score'] = 100.0
    peptide_df['ms2_score'] = 0.0

    for psm_df in psm_dfs:
        psm_df[pep_col] = remove_modifications(remove_charge(remove_previous_and_next_aa(psm_df[pep_col])))
        best_scores = psm_df.loc[psm_df.groupby(pep_col)[msms_col].idxmax()]

        # Merge with peptide_df
        peptide_df = peptide_df.merge(
            best_scores[[pep_col, rt_col, msms_col]],
            left_on='Peptide',
            right_on=pep_col,
            how='left'
        )

        # Update scores if new ones are higher
        mask = peptide_df[msms_col] > peptide_df['ms2_score']
        peptide_df.loc[mask, 'ms2_score'] = peptide_df.loc[mask, msms_col]

        if peptide_df[rt_col].dtype == float:
            mask = peptide_df[rt_col] < peptide_df['rt_score']
            peptide_df.loc[mask, 'rt_score'] = peptide_df.loc[mask, rt_col]

        # Remove temporary columns
        peptide_df = peptide_df.drop([msms_col, rt_col], axis=1, errors='ignore')

    return peptide_df

def anno_gene(df):
    genes = []
    gene_descriptions = []
    for p in df['Proteins']:
        ps = sorted(p.split(';'))
        if len(ps) == 1:
            ps = sorted(p.split(','))
        first_p = ''
        for pi in ps:
            if pi.startswith('rev_'):
                continue
            first_p = pi
            break
        if first_p == '':
            genes.append('')
            gene_descriptions.append('')
        else:
            genes.append(gene_map[first_p])
            gene_descriptions.append(gene_description_map[gene_map[first_p]])
    df['Gene'] = genes
    df['Gene_Description'] = gene_descriptions
    return df

def save_ebv(ebv_df, tool_name):
    # Calculate gene statistics
    # gene_stats = ebv_df.groupby('Gene').agg({
    #     'Peptide': 'count',
    #     'Binder': lambda x: (sum(x == 'Strong'), sum(x == 'Weak')),
    # }).reset_index()
    # gene_stats.columns = ['Gene', 'Total', ('Strong_binders', 'Weak_binders')]
    # gene_stats[['Strong_binders', 'Weak_binders']] = pd.DataFrame(
    #     gene_stats[('Strong_binders', 'Weak_binders')].tolist(), index=gene_stats.index)
    # gene_stats['Non-binders'] = gene_stats['Total'] - gene_stats['Strong_binders'] - gene_stats['Weak_binders']
    #
    # # Create stacked bar plot
    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=gene_stats, x='Gene', y='Strong_binders', color='red', label='Strong Binders')
    # sns.barplot(data=gene_stats, x='Gene', y='Weak_binders', color='orange', label='Weak Binders',
    #             bottom=gene_stats['Strong_binders'])
    # sns.barplot(data=gene_stats, x='Gene', y='Non-binders', color='blue', label='Non-binders',
    #             bottom=gene_stats['Strong_binders'] + gene_stats['Weak_binders'])
    #
    # plt.title(f'EBV Gene Statistics - {tool_name}')
    # plt.xticks(rotation=45, ha='right')
    # plt.ylabel('Number of Peptides')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    save_folder = Path('/mnt/d/workspace/mhc-booster/experiment/EBV/test')
    save_folder.mkdir(exist_ok=True, parents=True)
    ebv_df.to_csv(save_folder / f'EBV_{tool_name}.tsv', sep='\t', index=False)

def save_all(result_df, tool_name):
    save_folder = Path('/mnt/d/workspace/mhc-booster/experiment/EBV/test')
    save_folder.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(save_folder / f'ALL_{tool_name}.tsv', sep='\t', index=False)


def eval_percolator(score_folders, pout_folders, tool_name='FragPipe'):
    result_df = pd.DataFrame()
    for i, (score_folder, pout_folder) in enumerate(zip(score_folders, pout_folders)):
        score_folder, pout_folder = Path(score_folder), Path(pout_folder)
        psm_score_dfs, psm_dfs, pep_dfs = [], [], []
        for pin_file in score_folder.rglob('features.tsv'):
            psm_score_df = pd.read_csv(pin_file, sep='\t')
            psm_score_dfs.append(psm_score_df)

            if tool_name == 'FragPipe':
                pep_path = pout_folder / pin_file.parent.name.replace('_MHCBooster', '_edited_pep_target.pout')
                psm_path = pout_folder / pin_file.parent.name.replace('_MHCBooster', '_edited_psm_target.pout')
            else:
                pep_path = pout_folder / pin_file.parent.name.replace('_MHCBooster', '_pep_target.pout')
                psm_path = pout_folder / pin_file.parent.name.replace('_MHCBooster', '_psm_target.pout')
            with open(pep_path, 'r') as file:
                content = file.read().replace(';\t', ';')
            with open(str(pep_path) + '.m', 'w') as file:
                file.write(content)
            with open(psm_path, 'r') as file:
                content = file.read().replace(';\t', ';')
            with open(str(psm_path) + '.m', 'w') as file:
                file.write(content)
            pep_dfs.append(pd.read_csv(str(pep_path) + '.m', sep='\t'))
            psm_dfs.append(pd.read_csv(str(psm_path) + '.m', sep='\t'))

        peptide_df = get_identified(psm_dfs, 'q-value', 'peptide', 'proteinIds', FDR, 8, 14)
        # peptide_df = get_identified(pep_dfs, 'q-value', 'peptide', 'proteinIds', FDR, 8, 14)
        peptide_df = search_best_scores(peptide_df, psm_score_dfs, 'Peptide', 'Prosit_2019_irt_rt_error', 'Prosit_2023_intensity_timsTOF_entropy_score')
        peptide_df = fill_identified_psm_count(peptide_df, psm_dfs, 'q-value', 'peptide', FDR, 8, 14)
        peptide_df['From'] = i
        result_df = pd.concat([result_df, peptide_df], ignore_index=True)

    result_df = result_df.groupby('Peptide', as_index=False).agg({
        'Proteins': 'first', 'Q-value': 'min', 'rt_score': 'min', 'ms2_score': 'max', 'PSM_Count': 'sum', 'From': lambda x: ','.join(x.astype(str))})
    result_df['ID_Frequency'] = result_df['From'].str.count(',') + 1
    result_df = result_df.merge(binders, on='Peptide', how='left')
    # result_df = anno_gene(result_df)
    psm_count = np.sum(result_df['PSM_Count'])
    ebv_df = result_df[result_df['Proteins'].str.contains('EBV') & ~result_df['Proteins'].str.contains('HUMAN')]
    ebv_psm_count = np.sum(ebv_df['PSM_Count'])
    print(f'Sequences: {len(result_df)}. PSMs: {psm_count}. EBV Sequences: {len(ebv_df)}. EBV PSMs: {ebv_psm_count}')

    # draw_mz_rt_scores(result_df, ebv_df, tool_name)
    save_ebv(ebv_df, tool_name)
    save_all(result_df, tool_name)
    return result_df, ebv_df

def eval_mhcbooster(score_folders, peptide_folders):

    result_df = pd.DataFrame()
    for i, (score_folder, peptide_folder) in enumerate(zip(score_folders, peptide_folders)):
        score_folder, peptide_folder = Path(score_folder), Path(peptide_folder)
        psm_score_dfs, psm_dfs, pep_dfs = [], [], []
        for pin_file in score_folder.rglob('features.tsv'):
            psm_score_df = pd.read_csv(pin_file, sep='\t')
            psm_score_dfs.append(psm_score_df)

            pep_path = peptide_folder / pin_file.parent.name / 'peptide.tsv'
            pep_df = pd.read_csv(str(pep_path), sep='\t')
            pep_dfs.append(pep_df)
            psm_path = peptide_folder / pin_file.parent.name / 'psm.tsv'
            psm_df = pd.read_csv(str(psm_path), sep='\t')
            psm_dfs.append(psm_df)

        peptide_df = get_identified(psm_dfs, 'psm_qvalue', 'sequence', 'protein', FDR, 8, 14, label_col='label', target_label='Target')
        # peptide_df = get_identified(pep_dfs, 'pep_qvalue', 'sequence', 'protein', FDR, 8, 14, label_col='label', target_label='Target')
        peptide_df = search_best_scores(peptide_df, psm_score_dfs, 'Peptide', 'Prosit_2019_irt_rt_error', 'Prosit_2023_intensity_timsTOF_entropy_score')
        peptide_df = fill_identified_psm_count(peptide_df, psm_dfs, 'psm_qvalue', 'sequence', FDR, 8, 14, label_col='label', target_label='Target')
        peptide_df['From'] = i
        result_df = pd.concat([result_df, peptide_df], ignore_index=True)

    result_df = result_df.groupby('Peptide', as_index=False).agg({
        'Proteins': 'first', 'Q-value': 'min', 'rt_score': 'min', 'ms2_score': 'max', 'PSM_Count': 'sum', 'From': lambda x: ','.join(x.astype(str))})
    result_df['ID_Frequency'] = result_df['From'].str.count(',') + 1
    result_df = result_df.merge(binders, on='Peptide', how='left')
    # result_df = anno_gene(result_df)
    psm_count = np.sum(result_df['PSM_Count'])
    ebv_df = result_df[result_df['Proteins'].str.contains('EBV') & ~result_df['Proteins'].str.contains('HUMAN')]
    ebv_psm_count = np.sum(ebv_df['PSM_Count'])
    print(f'Sequences: {len(result_df)}. PSMs: {psm_count}. EBV Sequences: {len(ebv_df)}. EBV PSMs: {ebv_psm_count}')

    # draw_mz_rt_scores(result_df, ebv_df, 'MHCBooster')
    save_ebv(ebv_df, 'MHCBooster')
    save_all(result_df, 'MHCBooster')
    return result_df, ebv_df


def draw_mz_rt_scores(pep_df, ebv_df, title=None):
    plt.figure(figsize=(6, 6))
    strong_binders = ebv_df[ebv_df['Binder'] == 'Strong']
    weak_binders = ebv_df[ebv_df['Binder'] == 'Weak']
    non_binders = ebv_df[ebv_df['Binder'] == 'Non-binder']
    sns.scatterplot(x=pep_df['ms2_score'], y=pep_df['rt_score'], alpha=0.3, color='gray', label='All Peptides', edgecolor=None, linewidth=0)
    sns.scatterplot(x=strong_binders['ms2_score'], y=strong_binders['rt_score'], color='red', label=f'EBV Strong Binders ({len(strong_binders)})')
    sns.scatterplot(x=weak_binders['ms2_score'], y=weak_binders['rt_score'], color='orange', label=f'EBV Weak Binders ({len(weak_binders)})')
    sns.scatterplot(x=non_binders['ms2_score'], y=non_binders['rt_score'], color='blue', label=f'EBV Non-binders ({len(non_binders)})')
    plt.xlabel('Prosit_2023_intensity_timsTOF_entropy_score')
    plt.ylabel('Prosit_2019_irt_rt_error')
    plt.xlim(0, 1)
    plt.ylim(0, 90)
    plt.title(title + f' (Binders: {len(strong_binders) + len(weak_binders)}, Non-binders: {len(non_binders)})')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    FDR = 0.01
    percolator_folders = [
        # '/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M_rerun/msfragger/percolator',
        # '/mnt/d/data/JY_500M/old/percolator',
        # '/mnt/d/data/JY_500M/new/percolator',
        # '/mnt/d/data/JY_500M/new_test/percolator',
        '/mnt/d/workspace/mhc-booster/experiment/paper/JY_Fractionation/percolator',
        # '/mnt/d/data/JY100M_Val_DDA_102824/percolator',
        # '/mnt/d/data/JY_EL4_Class1_DDA_SK_MS_013125/percolator',
        # '/mnt/d/data/JY_PC-9_50M_ClassI_MS_DDA/percolator',
        # '/mnt/d/workspace/mhc-booster/experiment/RA_Fractionation/percolator',
        # '/mnt/e/data/Low-input_microIP_JY_Moh_DDA_new/percolator',
    ]
    fragpipe_folders = [
        # '/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M_rerun/msfragger/fragpipe',
        # '/mnt/d/data/JY_500M/old/fragpipe',
        # '/mnt/d/data/JY_500M/new/fragpipe',
        # '/mnt/d/data/JY_500M/new_test/fragpipe',
        '/mnt/d/workspace/mhc-booster/experiment/paper/JY_Fractionation/fragpipe',
        # '/mnt/d/data/JY100M_Val_DDA_102824/fragpipe',
        # '/mnt/d/data/JY_EL4_Class1_DDA_SK_MS_013125/fragpipe',
        # '/mnt/d/data/JY_PC-9_50M_ClassI_MS_DDA/fragpipe',
        # '/mnt/d/workspace/mhc-booster/experiment/RA_Fractionation/fragpipe',
        # '/mnt/e/data/Low-input_microIP_JY_Moh_DDA_new/fragpipe',
    ]
    mhcbooster_folders = [
        # '/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M_rerun/msfragger/mhcbooster',
        # '/mnt/d/data/JY_500M/old/mhcbooster',
        # '/mnt/d/data/JY_500M/new/mhcbooster',
        # '/mnt/d/data/JY_500M/new_test/mhcbooster',
        '/mnt/d/workspace/mhc-booster/experiment/paper/JY_Fractionation/mhcbooster',
        # '/mnt/d/data/JY100M_Val_DDA_102824/mhcbooster',
        # '/mnt/d/data/JY_EL4_Class1_DDA_SK_MS_013125/mhcbooster',
        # '/mnt/d/data/JY_PC-9_50M_ClassI_MS_DDA/mhcbooster',
        # '/mnt/d/workspace/mhc-booster/experiment/RA_Fractionation/mhcbooster',
        # '/mnt/e/data/Low-input_microIP_JY_Moh_DDA_new/mhcbooster',
    ]
    binders = get_binders(mhcbooster_folders)
    gene_map, gene_description_map = get_gene_map(
        '/mnt/d/data/JY_1_10_25M/2024-09-03-decoys-contam-Human_EBV_GD1_B95.fasta', '')
    eval_percolator(mhcbooster_folders, percolator_folders, 'Percolator')
    eval_percolator(mhcbooster_folders, fragpipe_folders, 'FragPipe')
    eval_mhcbooster(mhcbooster_folders, mhcbooster_folders)
    print('Finished.')

    binders = get_binders(mhcbooster_folders)
