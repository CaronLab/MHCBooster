import numpy as np
import pandas as pd
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mhcbooster.utils.peptide import remove_previous_and_next_aa, remove_modifications, remove_charge

def read_result(pep_tsv_paths, aliases=None, rt_col='Prosit_2019_irt_rt_error', msms_col='Prosit_2023_intensity_timsTOF_entropy_score', min_len=8, max_len=14):
    results = {}
    for i, pep_tsv_path in enumerate(pep_tsv_paths):
        result_df = pd.read_csv(pep_tsv_path, sep='\t')
        if 'label' in result_df.columns and 'pep_qvalue' in result_df.columns:
            result_df = result_df[(result_df['label'] == 'Target') * (result_df['pep_qvalue'] < 0.01)]
        result_df = result_df[result_df['sequence'].str.len().between(min_len, max_len)]

        seqs = result_df['sequence'].unique()

        prots = result_df['protein'].unique()

        all_mods = []
        for s in result_df['peptide']:
            mods = set(re.findall(r"\[(.*?)\]", s))
            all_mods.extend(mods)
        mod_map = dict(Counter(all_mods))

        if 'binder' in result_df.columns:
            binder_map = dict(Counter(result_df['binder']))
            binder_df = result_df[result_df['binder'] != 'Non-binder']
            allele_map = dict(Counter(binder_df['best_allele']))
        else:
            binder_map, allele_map = None, None

        if rt_col is None and msms_col is None:
            rt_scores, ms2_scores = None, None
        else:
            feature_tsv_path = Path(pep_tsv_path).parent / 'features.tsv'
            feature_df = pd.read_csv(feature_tsv_path, sep='\t')
            feature_df['peptide'] = remove_charge(remove_previous_and_next_aa(feature_df['Peptide']))
            best_scores = feature_df.loc[feature_df.groupby('peptide')[msms_col].idxmax()]
            best_scores = best_scores[['peptide', rt_col, msms_col]]
            result_df = result_df.merge(best_scores, on='peptide', how='left')
            rt_scores = result_df[rt_col]
            ms2_scores = result_df[msms_col]
        if aliases is not None:
            results[aliases[i]] = (seqs, prots, mod_map, binder_map, allele_map, rt_scores, ms2_scores)
        else:
            run_name = pep_tsv_path.parent
            run_name_split = run_name.name.split('_')
            if len(run_name_split) > 2 and run_name_split[-2].isnumeric():
                alias = run_name_split[-2]
            else:
                alias = run_name
            results[alias] = (seqs, prots, mod_map, binder_map, allele_map, rt_scores, ms2_scores)
    return results

def calculate_matching_percentage(seqs_list):
    def calculate_paired_matching_percentage(seqs_eval, seqs_ref):
        seqs_eval_set = set(seqs_eval)
        seqs_ref_set = set(seqs_ref)
        return len(seqs_eval_set.intersection(seqs_ref_set)) / len(seqs_ref_set)
    matching_matrix = np.zeros((len(seqs_list), len(seqs_list)))
    for i, ref_seqs in enumerate(seqs_list):
        for j, eval_seqs in enumerate(seqs_list):
            if i == j:
                matching_matrix[i, j] = 1
                continue
            matching_matrix[i, j] = calculate_paired_matching_percentage(eval_seqs, ref_seqs)
    return matching_matrix

def draw_to_pdf(results):
    # Peptide
    all_pep_seqs = [seqs for seqs, _, _, _, _, _, _ in results.values()]

    plt.bar(results.keys(), [len(seqs) for seqs in all_pep_seqs])
    plt.title('Peptide Identification Number')
    plt.xlabel('Runs')
    plt.ylabel('IDs')
    plt.xticks(rotation=90)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    pep_matching_matrix = calculate_matching_percentage(all_pep_seqs)
    plt.figure(figsize=(10, 10))
    plt.imshow(pep_matching_matrix, cmap='Blues', vmin=0, vmax=1.0)
    plt.colorbar()
    plt.title('Peptide Overlapping percentage')
    plt.xticks(range(len(results)), results.keys(), rotation=90)
    plt.yticks(range(len(results)), results.keys())
    plt.xlabel('Evaluation')
    plt.ylabel('Reference')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Protein
    all_prots = [prots for _, prots, _, _, _, _, _ in results.values()]
    plt.bar(results.keys(), [len(prots) for prots in all_prots])
    plt.title('Protein Identification Number')
    plt.xlabel('Runs')
    plt.ylabel('IDs')
    plt.xticks(rotation=90)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    prot_matching_matrix = calculate_matching_percentage(all_prots)
    plt.figure(figsize=(10, 10))
    plt.imshow(prot_matching_matrix, cmap='Blues', vmin=0, vmax=1.0)
    plt.colorbar()
    plt.title('Protein Overlapping percentage')
    plt.xticks(range(len(results)), results.keys(), rotation=90)
    plt.yticks(range(len(results)), results.keys())
    plt.xlabel('Evaluation')
    plt.ylabel('Reference')
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    # Mods
    all_mod_maps = [mod_map for _, _, mod_map, _, _, _, _ in results.values()]
    all_mods = set()
    for mod_map in all_mod_maps:
        all_mods.update(mod_map.keys())
    for mod in all_mods:
        counts = []
        for mod_map in all_mod_maps:
            if mod in mod_map:
                counts.append(mod_map[mod])
            else:
                counts.append(0)
        plt.plot(results.keys(), counts, label=mod)
    plt.title('Modification Number')
    plt.xlabel('Runs')
    plt.ylabel('Number')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    for mod in all_mods:
        percentage = []
        for i, mod_map in enumerate(all_mod_maps):
            if mod in mod_map:
                percentage.append(mod_map[mod] / len(all_pep_seqs[i]))
            else:
                percentage.append(0)
        plt.plot(results.keys(), percentage, label=mod)
    plt.title('Modification Percentage')
    plt.xlabel('Runs')
    plt.ylabel('Percentage')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Binders
    binder_keys = ['Strong', 'Weak', 'Non-binder']
    all_binder_maps = [binder_map for _, _, _, binder_map, _, _, _ in results.values()]
    if all_binder_maps[0] is not None:
        bottoms = np.zeros(len(all_binder_maps))
        all_counts = []
        for key in binder_keys:
            counts = []
            for binder_map in all_binder_maps:
                if key in binder_map:
                    counts.append(binder_map[key])
                else:
                    counts.append(0)
            plt.bar(results.keys(), counts, bottom=bottoms, label=key)
            bottoms = [bottoms[i] + counts[i] for i, count in enumerate(counts)]
            all_counts.append(counts)
        plt.xticks(rotation=90)
        plt.title('Binder Number')
        plt.xlabel('Runs')
        plt.ylabel('Number')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        all_percents = np.array(all_counts).astype(float)
        for j in range(all_percents.shape[1]):
            all_percents[:, j] = all_percents[:, j] * 100.0 / np.sum(all_percents[:, j])
        bottoms = np.zeros(all_percents.shape[1])
        for i, key in enumerate(binder_keys):
            plt.bar(results.keys(), all_percents[i], bottom=bottoms, label=key)
            bottoms = bottoms + all_percents[i]
        plt.xticks(rotation=90)
        plt.title('Binder Percentage')
        plt.xlabel('Runs')
        plt.ylabel('Percentage')
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Scores
    all_rt_scores = [rt_scores for _, _, _, _, _, rt_scores, _ in results.values()]
    if all_rt_scores[0] is not None:
        plt.violinplot(all_rt_scores, showmeans=False, showmedians=True, showextrema=False)
        plt.title('RT Score Distribution')
        plt.xticks(range(1, len(results) + 1), results.keys(), rotation=90)
        plt.ylabel('RT Score')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        all_msms_scores = [msms_scores for _, _, _, _, _, _, msms_scores in results.values()]
        plt.violinplot(all_msms_scores, showmeans=False, showmedians=True, showextrema=False)
        plt.title('MS2 Score Distribution')
        plt.xticks(range(1, len(results) + 1), results.keys(), rotation=90)
        plt.ylabel('MS2 Score')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

def draw_title_to_pdf(title):
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.text(0.5, 0.7, title, fontsize=24, ha="center", va="center", weight="bold")
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == '__main__':
    result_paths = ['/mnt/d/workspace/mhc-booster/experiment/paper/JY_1_10_25M/msfragger/mhcbooster',
                    '/mnt/d/workspace/mhc-booster/experiment/paper/JY_Fractionation/mhcbooster',
                    '/mnt/d/workspace/mhc-booster/experiment/JY_500M/mhcbooster_0521/mhcbooster',
                    '/mnt/e/data/BigIP_JY_500M_HLA-I/mhcbooster']
    aliases = ['JY_1_10_25M', 'JY_Fractionation', 'JY_500M', 'BigIP_JY_500M_HLA-I']
    pdf_output_path = Path('/mnt/d/workspace/mhc-booster/experiment/paper/qc_internal_test.pdf')

    def sort_results(results):
        seq_lens = [len(seqs) for seqs, _, _, _, _, _, _ in results.values()]
        sorted_idx = np.argsort(seq_lens)[::-1]
        items = list(results.items())
        sorted_items = [items[i] for i in sorted_idx]
        return dict(sorted_items)

    with PdfPages(pdf_output_path) as pdf:
        # Overall
        draw_title_to_pdf('Comparison Between Batches')
        combined_result_paths = [Path(result_path) / 'combined_peptide.tsv' for result_path in result_paths]
        # aliases = [Path(result_path).parent.name for result_path in result_paths]
        results = read_result(combined_result_paths, aliases=aliases, rt_col=None, msms_col=None)
        results = sort_results(results)
        draw_to_pdf(results)

        # Each batch
        for p, result_path in enumerate(result_paths):
            draw_title_to_pdf(aliases[p])
            result_tsv_paths = list(Path(result_path).rglob('peptide.tsv'))
            results = read_result(result_tsv_paths)
            results = sort_results(results)
            draw_to_pdf(results)