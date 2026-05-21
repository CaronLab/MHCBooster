import re
import numpy as np
import pandas as pd
import matplotlib.backends.backend_pdf as plt_pdf

from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap
from mhcbooster.utils.fdr import calculate_qs, calculate_peptide_level_qs, calculate_roc
from mhcnames import normalize_allele_name
from pyteomics.mass import calculate_mass, std_aa_mass
from mhcbooster.utils.constants import PROTON_MASS
from mhcbooster.utils.peptide import extract_mods, extract_mod_mass_diff
from mhcbooster.utils.log import read_from_log


class RunReporter:

    def __init__(self, report_directory, decoy_prefix):

        self.report_directory = Path(report_directory)
        self.pepxml_path = str((self.report_directory / 'peptide.pep.xml').resolve())
        self.decoy_prefix = decoy_prefix

        self.psm_df = pd.DataFrame()
        self.peptide_df = pd.DataFrame()
        self.sequence_df = pd.DataFrame()


    def add_run_result(self, spec_names, spec_indices, rts, ims, masses, charges, peptides, sequences, prev_aas, next_aas, labels, scores, proteins):
        psm_qvalues = calculate_qs(scores, labels)
        pep_qvalues, _, _, peps, _ = calculate_peptide_level_qs(scores, labels, peptides)
        seq_qvalues, _, _, seqs, _ = calculate_peptide_level_qs(scores, labels, sequences)
        pep_qvalue_lookup = {pep: q for pep, q in zip(peps, pep_qvalues)}
        seq_qvalue_lookup = {seq: q for seq, q in zip(seqs, seq_qvalues)}


        self.psm_df['Spectrum'] = spec_names
        self.psm_df['Spectrum File'] = self.pepxml_path
        self.psm_df['Index'] = spec_indices
        self.psm_df['Retention'] = rts * 60
        self.psm_df['Ion Mobility'] = ims
        self.psm_df['Peptide'] = peptides
        self.psm_df['Sequence'] = sequences
        self.psm_df['Charge'] = charges
        self.psm_df['Assigned Modifications'] = [extract_mods(peptide) for peptide in peptides]
        self.psm_df['Observed Modifications'] = ''
        self.psm_df['Prev AA'] = prev_aas
        self.psm_df['Next AA'] = next_aas
        self.psm_df['Label'] = np.array(['Target' if label == 1 else 'Decoy' for label in labels])
        self.psm_df['Peptide Length'] = self.psm_df['Sequence'].str.len().astype(str)
        self.psm_df['Observed Mass'] = masses
        self.psm_df['Observed M/Z'] = masses / charges + PROTON_MASS
        self.psm_df['Calibrated Observed Mass'] = masses
        self.psm_df['Calibrated Observed M/Z'] = masses / charges + PROTON_MASS
        self.psm_df['Calculated Mass'] = [calculate_mass(sequences[i]) + np.sum(extract_mod_mass_diff(peptides[i])) for i in range(len(sequences))]
        self.psm_df['Calculated M/Z'] = self.psm_df['Calculated Mass'] / self.psm_df['Charge'] + PROTON_MASS
        self.psm_df['Score'] = scores
        self.psm_df['Psm Qvalue'] = psm_qvalues
        self.psm_df['Pep Qvalue'] = np.array([pep_qvalue_lookup[pep] for pep in peptides])
        self.psm_df['Seq Qvalue'] = np.array([seq_qvalue_lookup[seq] for seq in sequences])
        self.psm_df['Protein'] = proteins
        self.psm_df['Protein ID'] = ''
        self.psm_df['Entry Name'] = ''
        self.psm_df['Gene'] = ''
        self.psm_df['Protein Description'] = ''
        self.psm_df['Mapped Proteins'] = ''
        self.psm_df['Mapped Genes'] = ''
        self.psm_df['Intensity'] = ''


    def add_app_score(self):
        app_score_paths = self.report_directory.glob('app_prediction.*.tsv')
        app_df = pd.DataFrame()
        app_df['Sequence'] = np.unique(self.psm_df['Sequence'])
        for app_score_path in app_score_paths:
            predictor = app_score_path.stem.split('.')[1]
            psm_app_df = pd.read_csv(app_score_path, sep='\t')
            if predictor == 'netmhcpan':
                seq_app_df = psm_app_df.loc[psm_app_df.groupby('Peptide')['EL_Rank'].idxmin(), ['Peptide', 'Allele', 'EL_Rank']]
                seq_app_df['netmhcpan_binder'] = seq_app_df['EL_Rank'].apply(lambda r: 'Strong' if r < 0.5 else ('Weak' if r < 2 else 'Non-binder'))
            if predictor == 'mhcflurry':
                seq_app_df = psm_app_df.loc[psm_app_df.groupby('peptide')['mhcflurry_affinity_percentile'].idxmin(), ['peptide', 'allele', 'mhcflurry_affinity_percentile']]
                seq_app_df['mhcflurry_binder'] = seq_app_df['mhcflurry_affinity_percentile'].apply(lambda r: 'Strong' if r < 0.5 else ('Weak' if r < 2 else 'Non-binder'))
            if predictor == 'bigmhc':
                seq_app_df = psm_app_df.loc[psm_app_df.groupby('pep')['BigMHC_EL'].idxmin(), ['pep', 'mhc', 'BigMHC_EL']]
                seq_app_df['BigMHC_EL'] *= 100
                seq_app_df['bigmhc_binder'] = seq_app_df['BigMHC_EL'].apply(lambda r: 'Strong' if r < 0.5 else ('Weak' if r < 2 else 'Non-binder'))
            if predictor == 'netmhciipan':
                seq_app_df = psm_app_df.loc[psm_app_df.groupby('Peptide')['EL_Rank'].idxmin(), ['Peptide', 'Allele', 'EL_Rank']]
                seq_app_df['netmhciipan_binder'] = seq_app_df['EL_Rank'].apply(lambda r: 'Strong' if r < 0.5 else ('Weak' if r < 2 else 'Non-binder'))
            if predictor == 'mixmhc2pred':
                psm_app_df['BestAllele'] = psm_app_df['BestAllele'].fillna('')
                psm_app_df['%Rank_best'] = psm_app_df['%Rank_best'].fillna(100)
                seq_app_df = psm_app_df.loc[psm_app_df.groupby('Peptide')['%Rank_best'].idxmin(), ['Peptide', 'BestAllele', '%Rank_best']]
                seq_app_df['mixmhc2pred_binder'] = seq_app_df['%Rank_best'].apply(lambda r: 'Strong' if r < 0.5 else ('Weak' if r < 2 else 'Non-binder'))
            seq_app_df.columns = ['Sequence', 'Best Allele', 'Min Rank'] + [seq_app_df.columns[3]]
            app_df = pd.merge(app_df, seq_app_df, on='Sequence', how='left', suffixes=(' left', ' right'))
            if 'Min Rank left' in app_df.columns:
                app_df['Min Rank'] = app_df[['Min Rank left', 'Min Rank right']].min(axis=1)
                app_df['Best Allele'] = np.where(app_df['Min Rank'] == app_df['Min Rank left'], app_df['Best Allele left'], app_df['Best Allele right'])
                app_df.drop(columns=['Min Rank left', 'Min Rank right', 'Best Allele left', 'Best Allele right'], inplace=True)
        if 'Best Allele' in app_df.columns:
            app_df['Best Allele'] = app_df['Best Allele'].apply(lambda a: normalize_allele_name(a.replace('__', '-')) if a != '' else a)
            app_df['Binder'] = (app_df[[col for col in app_df.columns if '_binder' in col]]
                                .apply(lambda b: 'Strong' if 'Strong' in b.values else ('Weak' if 'Weak' in b.values else 'Non-binder'), axis=1))
        self.psm_df = pd.merge(self.psm_df, app_df, on='Sequence', how='left')


    def generate_pep_xml(self, fasta_path):
        log_path = self.report_directory / 'input_files.txt'
        mzml_path = read_from_log('mzml=', log_path).replace('.mzML', '')
        header = ['<?xml version="1.0" encoding="UTF-8"?>\n',
                  '<?xml-stylesheet type="text/xsl" href="pepXML_std.xsl"?>\n',
                  '<msms_pipeline_analysis xmlns="http://regis-web.systemsbiology.net/pepXML" xsi:schemaLocation="http://sashimi.sourceforge.net/schema_revision/pepXML/pepXML_v122.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n',
                  '<analysis_summary analysis="MHCBooster">\n',
                  '</analysis_summary>\n',
                  f'<msms_run_summary base_name="{mzml_path}" raw_data_type="mzML" raw_data="mzML">\n']

        with open(self.pepxml_path, 'w') as pep_xml:
            pep_xml.writelines(header)
            pep_xml.write('<search_summary>\n')
            pep_xml.write(f'<search_database local_path="{fasta_path}" type="AA"/>\n')
            pep_xml.write(f'</search_summary>\n')

            for i, psm in self.psm_df.iterrows():
                sequence = psm['Sequence']
                modified_sequence = psm['Sequence']
                modified_mass = calculate_mass(sequence)
                mod_map = dict()
                if not pd.isna(psm['Assigned Modifications']) and len(psm['Assigned Modifications']) > 0:
                    for mod in psm['Assigned Modifications'].split(', '):
                        if mod.startswith('N-term'):
                            mass = re.findall(r"\((.*)\)", mod)[0]
                            modified_mass += float(mass)
                            mod_map[0] = float(mass) + 1.007825
                        else:
                            pos, res, mass = re.match(r"(\d+)([A-Z])\((.*)\)", mod).groups()
                            modified_mass += float(mass)
                            mod_map[int(pos)] = float(mass) + std_aa_mass[res]
                    sorted_idx = np.argsort(-1 * np.array(list(mod_map.keys())))
                    for j in range(len(mod_map)):
                        idx = sorted_idx[j]
                        pos = list(mod_map.keys())[idx]
                        mod_aa_mass = str(round(mod_map[pos]))
                        if pos != 0:
                            modified_sequence = modified_sequence[:pos] + f'[{mod_aa_mass}]' + modified_sequence[pos:]
                        else:
                            modified_sequence = f'n[{mod_aa_mass}]' + modified_sequence

                charge = psm['Charge']
                proteins = [protein for protein in psm['Protein'].split(';') if len(protein.strip()) > 0]
                score = psm['Score']

                spectrum_id = self.psm_df['Spectrum'].iloc[i]
                idx = self.psm_df['Index'].iloc[i]
                rt = float(self.psm_df['Retention'].iloc[i])
                im = self.psm_df['Ion Mobility'].iloc[i]
                scan_nrs = spectrum_id.rsplit('.', 3)[-3:-1]
                pep_xml.write(f'<spectrum_query assumed_charge="{charge}" spectrum="{spectrum_id}" index="{int(idx) + 1}" start_scan="{scan_nrs[0]}" end_scan="{scan_nrs[1]}" retention_time_sec="{rt}" ion_mobility="{im}">\n')
                pep_xml.write('<search_result>\n')
                pep_xml.write(f'<search_hit peptide="{sequence}" massdiff="0" calc_neutral_pep_mass="{round(modified_mass, 5)}" num_tot_proteins="{len(proteins)}" hit_rank="1" protein="{proteins[0]}">\n')

                if modified_sequence != sequence:
                    if modified_sequence.startswith('n'):
                        pep_xml.write(f'<modification_info modified_peptide="{modified_sequence}" mod_nterm_mass="{round(mod_map[0], 5)}">\n')
                    else:
                        pep_xml.write(f'<modification_info modified_peptide="{modified_sequence}">\n')
                    for pos, mass in mod_map.items():
                        if pos != 0:
                            pep_xml.write(f'<mod_aminoacid_mass mass="{round(mass, 5)}" position="{pos}"/>\n')
                    pep_xml.write(f'</modification_info>\n')

                for i in range(1, len(proteins)):
                    pep_xml.write(f'<alternative_protein protein="{proteins[i]}"/>\n')
                pep_xml.write('<analysis_result analysis="peptideprophet">\n')
                pep_xml.write(
                    f'<peptideprophet_result probability="{score}" all_ntt_prob="({score},{score},{score})">\n')
                pep_xml.write('</peptideprophet_result>\n')
                pep_xml.write('</analysis_result>\n')
                pep_xml.write('</search_hit>\n')
                pep_xml.write('</search_result>\n')
                pep_xml.write('</spectrum_query>\n')
            pep_xml.write('</msms_run_summary>\n')
            pep_xml.write('</msms_pipeline_analysis>\n')


    def generate_psm_report(self, psm_fdr=1, remove_decoy=False):
        if remove_decoy:
            psm_df = self.psm_df[self.psm_df['Label'] == 'Target'].copy()
        else:
            psm_df = self.psm_df.copy()
        psm_df = psm_df[psm_df['Psm Qvalue'] <= psm_fdr]
        psm_df.to_csv(self.report_directory / f'psm.tsv', index=False, sep='\t')
        return psm_df


    def generate_peptide_report(self, pep_fdr=1, remove_decoy=False, sequential=False, psm_fdr=1):
        if sequential:
            psm_df = self.psm_df[self.psm_df['Psm Qvalue'] <= psm_fdr]
        else:
            psm_df = self.psm_df
        if remove_decoy:
            psm_df = psm_df[psm_df['Label'] == 'Target']

        psm_df = psm_df[psm_df['Pep Qvalue'] <= pep_fdr]
        agg_dict = {
            'Sequence': 'first',
            'Prev AA': lambda x: ','.join(set(x)),
            'Next AA': lambda x: ','.join(set(x)),
            'Label': lambda x: 'Target' if 'Target' in set(x) else 'Decoy',
            'Peptide Length': 'first',
            'Charge': lambda x: ','.join([str(c) for c in sorted(set(x))]),
            'Score': 'max',
            'Pep Qvalue': 'max',
            'Protein': 'first',
            'Protein ID': 'first',
            'Entry Name': 'first',
            'Gene': 'first',
            'Protein Description': 'first',
            'Mapped Proteins': 'first',
            'Mapped Genes': 'first'
        }
        for col in psm_df.columns:
            if 'Binder' in col or 'Best Allele' in col or 'Min Rank' in col:
                agg_dict[col] = 'first'
        peptide_df = psm_df.groupby('Peptide', as_index=False).agg(agg_dict)
        pep_stat = psm_df['Peptide'].value_counts().reset_index().rename(columns={'count': 'Spectral Count'})
        self.peptide_df = peptide_df.merge(pep_stat, how='left', on='Peptide')
        self.peptide_df.to_csv(self.report_directory / f'peptide.tsv', index=False, sep='\t')
        return self.peptide_df


    def generate_sequence_report(self, seq_fdr=1, remove_decoy=False, sequential=False, psm_fdr=1):
        if sequential:
            psm_df = self.psm_df[self.psm_df['Psm Qvalue'] <= psm_fdr]
        else:
            psm_df = self.psm_df
        if remove_decoy:
            psm_df = psm_df[psm_df['Label'] == 'Target']

        psm_df = psm_df[psm_df['Seq Qvalue'] <= seq_fdr]
        agg_dict = {
            'Peptide': lambda x: ','.join(set(x)),
            'Prev AA': lambda x: ','.join(set(x)),
            'Next AA': lambda x: ','.join(set(x)),
            'Label': lambda x: 'Target' if 'Target' in set(x) else 'Decoy',
            'Peptide Length': 'first',
            'Charge': lambda x: ','.join([str(c) for c in sorted(set(x))]),
            'Score': 'max',
            'Seq Qvalue': 'max',
            'Protein': 'first',
            'Protein ID': 'first',
            'Entry Name': 'first',
            'Gene': 'first',
            'Protein Description': 'first',
            'Mapped Proteins': 'first',
            'Mapped Genes': 'first'
        }
        for col in psm_df.columns:
            if 'Binder' in col or 'Best Allele' in col or 'Min Rank' in col:
                agg_dict[col] = 'first'
        sequence_df = psm_df.groupby('Sequence', as_index=False).agg(agg_dict)
        seq_stat = psm_df['Sequence'].value_counts().reset_index().rename(columns={'count': 'Spectral Count'})
        self.sequence_df = sequence_df.merge(seq_stat, how='left', on='Sequence')
        self.sequence_df.to_csv(self.report_directory / f'sequence.tsv', index=False, sep='\t')
        return self.sequence_df


    def draw_result_figure(self, val_loss_list, psm_fdr=0.01):
        roc = calculate_roc(self.psm_df['Psm Qvalue'], self.psm_df['Label'])
        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        fig.suptitle(self.report_directory.name, fontsize=16)
        gs = GridSpec(2, 2, figure=fig)
        colormap = get_cmap("tab10")

        final = fig.add_subplot(gs[0, 0])
        final.plot(*roc, c=colormap(0), ms='3', ls='none', marker='.', alpha=0.6)
        n_psms_at_fdr = np.sum((self.psm_df['Psm Qvalue'] <= psm_fdr) & (self.psm_df['Label'] == 'Target'))
        final.vlines(psm_fdr, 0, n_psms_at_fdr, ls='--', lw=1, color='k', alpha=0.7)
        final.hlines(n_psms_at_fdr, 0, psm_fdr, ls='--', lw=1, color='k', alpha=0.7)
        final.set_xlim((0, max(0.05, 2 * psm_fdr)))
        final.set_title('Final q-values')
        final.set_xlabel('q-value')
        final.set_ylabel('PSMs')
        final.set_ylim((0, final.get_ylim()[1]))

        dist = fig.add_subplot(gs[0, 1])
        scores = self.psm_df['Score']
        labels = self.psm_df['Label']
        bins = np.arange(0, 1.025, 0.025)
        dist.hist(scores[labels == 'Target'], label='Target', bins=bins, alpha=0.5, color='g')
        dist.hist(scores[labels == 'Decoy'], label='Decoy', bins=bins, alpha=0.5, zorder=100, color='r')
        dist.set_title('Prediction distributions')
        dist.set_xlabel('Target probability')
        dist.set_ylabel('PSMs')
        dist.legend()

        loss = fig.add_subplot(gs[1, 1])
        min_x = []
        min_y = []
        for i, val_loss in enumerate(val_loss_list):
            loss.plot(range(1, len(val_loss) + 1), val_loss, c=colormap(i), marker=None, label=f'split {i+1}')
            min_y.append(np.min(val_loss))
            min_x.append(np.argmin(val_loss) + 1)
        loss.plot(min_x, min_y, ls='none', marker='x', ms='12', c='k', label='best models')
        loss.set_title('Validation loss')
        loss.set_xlabel('Epoch')
        loss.set_ylabel('Loss')
        loss.legend()
        plt.tight_layout()

        pdf_file = self.report_directory / 'training_report.pdf'
        pdf = plt_pdf.PdfPages(str(pdf_file), keep_empty=False)
        pdf.savefig(fig)
        pdf.close()
        plt.close(fig)


if __name__ == '__main__':
    # psm_df = pd.read_csv('/mnt/e/mhcb_v2.2_test/JY_Class1_1M_DDA_60min_Slot1-10_1_541_MHCBooster/psm.tsv', sep='\t')
    # run_reporter = RunReporter(report_directory='/mnt/e/mhcb_v2.2_test/JY_Class1_1M_DDA_60min_Slot1-10_1_541_MHCBooster/test', decoy_prefix='rev_')
    # psm_df = pd.read_csv('/mnt/d/data/JY_Fractionation_Replicate_1/mhcbooster_0305/JY_MHC1_T1_F1_iRT_DDA_Slot1-2_1_782_MHCBooster/psm.tsv', sep='\t')
    # run_reporter = RunReporter(report_directory='/mnt/d/data/JY_Fractionation_Replicate_1/mhcbooster_0306/JY_MHC1_T1_F1_iRT_DDA_Slot1-2_1_782_MHCBooster',
    #                            file_name='test', decoy_prefix='rev_')
    # psm_df['protein_id'] = ''
    # psm_df['entry_name'] = ''
    # psm_df['protein_description'] = ''
    # psm_df['mapped_protein'] = ''
    # run_reporter.psm_df = psm_df
    # run_reporter.generate_pep_xml(fasta_path='fasta_path')
    # run_reporter.add_app_score()
    # run_reporter.infer_protein('/mnt/d/data/Library/2025-02-26-decoys-contam-JY_var_splicing_0226.fasta.fas')
    # run_reporter.generate_psm_report()
    # run_reporter.generate_peptide_report(pep_fdr=0.01, remove_decoy=True, sequential=True, psm_fdr=0.01)
    # run_reporter.generate_sequence_report()

    root_path = Path('/mnt/f/paper_data/JY_DDA_HLA-I_dataset/mhcbooster_mhclib')
    for psm_path in root_path.rglob('psm.tsv'):
        pepxml_path = psm_path.parent / 'peptide.pep.xml'
        run_reporter = RunReporter(report_directory=psm_path.parent.absolute(), decoy_prefix='rev_')
        psm_df = pd.read_csv(psm_path, sep='\t')
        run_reporter.psm_df = psm_df
        run_reporter.generate_pep_xml(fasta_path='/mnt/f/paper_data/JY_DDA_HLA-I_dataset/SA1_1_with_decoys.fasta')
