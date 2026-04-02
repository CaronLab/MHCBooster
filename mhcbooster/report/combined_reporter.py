import tempfile
import subprocess
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from pathlib import Path
from copy import deepcopy
from pyteomics import fasta, protxml
from mhcbooster.utils.peptide import extract_mod_mass_diff
from mhcbooster.utils.log import read_from_log

class CombinedReporter:

    def __init__(self, result_folder, fasta_path, pep_fdr=0.01, seq_fdr=0.01, infer_protein=True,
                 decoy_prefix = 'rev_', remove_contaminant=False, control_combine_fdr=False, build_library=True, quant=True):

        self.result_folder = Path(result_folder)
        self.fasta_path = fasta_path
        self.pep_fdr = pep_fdr
        self.seq_fdr = seq_fdr
        self.infer_protein = infer_protein
        self.decoy_prefix = decoy_prefix
        self.remove_contaminant = remove_contaminant
        self.control_combine_fdr = control_combine_fdr
        self.build_library = build_library
        self.quant = quant

        if self.fasta_path is None:
            fasta_paths = list(self.result_folder.glob('*.fasta')) + list(self.result_folder.glob('*.fas'))
            if len(fasta_paths) > 0:
                self.fasta_path = fasta_paths[0]

        self.fasta_map = None
        self.seq_prot_df = None
        self.combined_high_confidence_sequences = None

    def prepare_fasta_map(self):
        # Prepare fasta: some sequences are missing in prot.xml
        fasta_map = {}
        if self.fasta_path is not None:
            for protein in fasta.read(self.fasta_path):
                description = protein.description.strip()
                protein_name = description.split(' ')[0]
                protein_description = description[len(protein_name) + 1:]
                fasta_map[protein_name] = protein_description
        return fasta_map


    def do_protein_inference(self):
        if self.fasta_path is None:
            return pd.DataFrame()

        ### Run ProteinProphet
        pep_xml_paths = [path.absolute().as_posix() for path in self.result_folder.rglob('peptide.pep.xml')]
        with tempfile.NamedTemporaryFile('w') as pep_xml_list:
            pep_xml_list.write('\n'.join(pep_xml_paths))
            pep_xml_list.flush()
            philosopher_exe_path = Path(__file__).parent.parent / 'third_party' / 'philosopher_v5.1.0_linux_amd64' / 'philosopher'
            subprocess.run(f'{philosopher_exe_path} workspace --init', cwd=self.result_folder, shell=True)
            subprocess.run(f'{philosopher_exe_path} proteinprophet --maxppmdiff 2000000 --output combined {pep_xml_list.name}', cwd=self.result_folder, shell=True)
            subprocess.run(f'{philosopher_exe_path} workspace --clean --nocheck', cwd=self.result_folder, shell=True)
        if not (self.result_folder / 'combined.prot.xml').exists():
            print('Protein inference is skipped.')
            return pd.DataFrame()

        # Generate sequence-protein map from prot.xml
        seq_prot_map = {}
        for prot in protxml.read(str(self.result_folder / 'combined.prot.xml')):
            protein = prot['protein'][0]
            prot_desc_split = [t for t in protein['protein_description'].split(' ') if t.startswith('GN=')]
            gene_name = 'UNANNOTATED' if len(prot_desc_split) == 0 else prot_desc_split[0].replace('GN=', '')

            protein_list = [{
                'protein_name': protein['protein_name'],
                'protein_description': protein['protein_description'],
                'gene_name': self.decoy_prefix + gene_name if protein['protein_name'].startswith(
                    self.decoy_prefix) else gene_name,
                'n_related_peptides': len(protein['peptide'])
            }]
            if protein['n_indistinguishable_proteins'] > 1:
                for dup_prot in protein['indistinguishable_protein']:
                    prot_desc_split = [t for t in dup_prot['protein_description'].split(' ') if t.startswith('GN=')]
                    gene_name = 'UNANNOTATED' if len(prot_desc_split) == 0 else prot_desc_split[0].replace('GN=', '')
                    protein_list.append({
                        'protein_name': dup_prot['protein_name'],
                        'protein_description': dup_prot['protein_description'],
                        'gene_name': self.decoy_prefix + gene_name if dup_prot['protein_name'].startswith(
                            self.decoy_prefix) else gene_name,
                        'n_related_peptides': len(protein['peptide'])
                    })
            for peptide in prot['protein'][0]['peptide']:
                sequence = peptide['peptide_sequence']
                group_weight = peptide['group_weight']
                n_sibling_peptides = peptide['n_sibling_peptides']
                for prot_map in protein_list:
                    prot_map['group_weight'] = group_weight
                    prot_map['n_sibling_peptides'] = n_sibling_peptides

                if sequence not in seq_prot_map.keys():
                    seq_prot_map[sequence] = []
                protein_names = [protein['protein_name'] for protein in seq_prot_map[sequence]]
                for protein in protein_list:
                    if protein['protein_name'] not in protein_names:
                        seq_prot_map[sequence].append(deepcopy(protein))
                        protein_names.append(protein['protein_name'])
        (self.result_folder / 'combined.prot.xml').unlink()

        # find the best protein for each sequence
        seq_idx_map = {}
        for sequence in seq_prot_map.keys():
            protein_list = seq_prot_map[sequence]
            max_weight = max(protein['group_weight'] for protein in protein_list)
            indices = [i for i, protein in enumerate(protein_list) if protein['group_weight'] == max_weight]
            if len(indices) == 1:
                seq_idx_map[sequence] = indices[0]
                continue

            n_related_peptides = [protein_list[i]['n_related_peptides'] for i in indices]
            max_related_peptides = max(n_related_peptides)
            indices = [indices[i] for i in range(len(n_related_peptides)) if
                       n_related_peptides[i] == max_related_peptides]
            if len(indices) == 1:
                seq_idx_map[sequence] = indices[0]
                continue

            protein_names = [protein_list[i]['protein_name'] for i in indices]
            index = indices[protein_names.index(min(protein_names))]
            seq_idx_map[sequence] = index

        sequences = list(seq_prot_map.keys())
        proteins = [None] * len(sequences)
        protein_ids = [None] * len(sequences)
        entry_names = [None] * len(sequences)
        genes = [None] * len(sequences)
        protein_descriptions = [None] * len(sequences)
        mapped_proteins = [None] * len(sequences)
        mapped_genes = [None] * len(sequences)

        # Use list comprehensions and set operations to avoid redundant calculations
        for i, sequence in enumerate(sequences):
            protein_list = seq_prot_map[sequence]
            best_index = seq_idx_map[sequence]
            best_protein = protein_list[best_index]

            protein_name = best_protein['protein_name']
            proteins[i] = protein_name
            protein_split = protein_name.split('|')
            if len(protein_split) < 3:
                protein_split = protein_split + [''] * (3 - len(protein_split))
            protein_ids[i] = self.decoy_prefix + protein_split[1] if protein_name.startswith(self.decoy_prefix) else protein_split[1]
            entry_names[i] = self.decoy_prefix + protein_split[2] if protein_name.startswith(self.decoy_prefix) else protein_split[2]
            genes[i] = best_protein['gene_name']
            protein_descriptions[i] = best_protein['protein_description']

            mapped_proteins_list = list(set([protein_list[j]['protein_name'] for j in range(len(protein_list)) if j != best_index]))
            mapped_target_proteins = [protein_name for protein_name in mapped_proteins_list if not protein_name.startswith(self.decoy_prefix)]
            mapped_decoy_proteins = [protein_name for protein_name in mapped_proteins_list if protein_name.startswith(self.decoy_prefix)]
            mapped_proteins[i] = ','.join(sorted(mapped_target_proteins) + sorted(mapped_decoy_proteins))

            mapped_genes_list = list(set([protein_list[j]['gene_name'] for j in range(len(protein_list)) if j != best_index]))
            mapped_target_genes = [gene_name for gene_name in mapped_genes_list if not gene_name.startswith(self.decoy_prefix)]
            mapped_decoy_genes = [gene_name for gene_name in mapped_genes_list if gene_name.startswith(self.decoy_prefix)]
            mapped_genes[i] = ','.join(sorted(mapped_target_genes) + sorted(mapped_decoy_genes))

        seq_prot_df = pd.DataFrame({
            'Sequence': sequences,
            'Protein': proteins,
            'Protein ID': protein_ids,
            'Entry Name': entry_names,
            'Gene': genes,
            'Protein Description': protein_descriptions,
            'Mapped Proteins': mapped_proteins,
            'Mapped Genes': mapped_genes
        })
        return seq_prot_df


    def update_proteins(self, file_name):
        if self.fasta_path is None:
            return
        for result_path in self.result_folder.rglob(file_name):
            print(f'Inserting protein inference results to {result_path.name}')
            result_df = pd.read_csv(result_path, sep='\t')
            protein_idx = result_df.columns.get_loc('Protein')
            protein_col = result_df['Protein']
            result_df = result_df.drop(['Protein', 'Protein ID', 'Entry Name', 'Gene', 'Protein Description',
                                        'Mapped Proteins', 'Mapped Genes'], axis=1)
            merged_df = pd.DataFrame()
            merged_df['Sequence'] = result_df['Sequence']
            merged_df = pd.merge(merged_df, self.seq_prot_df, how='left', on='Sequence')
            for i in range(1, len(merged_df.columns)):
                result_df.insert(protein_idx + i - 1, merged_df.columns[i], merged_df.iloc[:, i])

            for i, row in result_df.iterrows():
                if pd.isna(row['Protein']):
                    proteins = sorted([protein for protein in protein_col[i].split(';') if len(protein.strip()) > 0])
                    if len(proteins) == 0:
                        continue
                    protein = proteins[0]
                    protein_description = ''
                    gene = ''
                    protein_id, entry_name = '', ''
                    if protein in self.fasta_map.keys():
                        protein_description = self.fasta_map[protein]
                        prot_desc_split = [t for t in protein_description.split(' ') if t.startswith('GN=')]
                        gene = 'UNANNOTATED' if len(prot_desc_split) == 0 else prot_desc_split[0][3:]
                        gene = self.decoy_prefix + gene if protein.startswith(self.decoy_prefix) else gene
                        prot_name_split = protein.split('|')
                        if len(prot_name_split) == 3:
                            protein_id = self.decoy_prefix + prot_name_split[1] if protein.startswith(self.decoy_prefix) else prot_name_split[1]
                            entry_name = self.decoy_prefix + prot_name_split[2] if protein.startswith(self.decoy_prefix) else prot_name_split[2]

                    mapped_proteins = proteins[1:]
                    mapped_target_proteins = [protein_name for protein_name in mapped_proteins if not protein_name.startswith(self.decoy_prefix)]
                    mapped_decoy_proteins = [protein_name for protein_name in mapped_proteins if protein_name.startswith(self.decoy_prefix)]
                    mapped_proteins = sorted(mapped_target_proteins) + sorted(mapped_decoy_proteins)
                    mapped_genes = []
                    for mapped_protein in mapped_proteins:
                        description = self.fasta_map[mapped_protein]
                        prot_desc_split = [t for t in description.split(' ') if t.startswith('GN=')]
                        gene_name = 'UNANNOTATED' if len(prot_desc_split) == 0 else prot_desc_split[0][3:]
                        gene_name = self.decoy_prefix + gene_name if mapped_protein.startswith(self.decoy_prefix) else gene_name
                        mapped_genes.append(gene_name)
                    mapped_genes = list(set(mapped_genes))
                    mapped_protein = ','.join(mapped_proteins)
                    mapped_gene = ','.join(mapped_genes)
                    result_df.loc[i, ['Protein', 'Protein Description', 'Gene', 'Protein ID', 'Entry Name', 'Mapped Proteins', 'Mapped Genes']] = \
                        [protein, protein_description, gene, protein_id, entry_name, mapped_protein, mapped_gene]
            result_df.to_csv(result_path, sep='\t', index=False)


    def remove_contaminants(self, file_name):
        contam_fasta = fasta.read(str(Path(__file__).parent / 'contaminant.fasta.fas'))
        contam_proteins = [p.description.split(' ')[0] for p in contam_fasta]
        for result_path in self.result_folder.rglob(file_name):
            print(f'Removing contaminants in {result_path.parent.name} {result_path.name}', end='\t')
            result_df = pd.read_csv(result_path, sep='\t')
            result_df_wo_contam = result_df[-result_df['Protein'].isin(contam_proteins)]
            print(f'{len(result_df) - len(result_df_wo_contam)} {file_name.split(".")[0]}s removed.')
            result_df_wo_contam.to_csv(result_path, sep='\t', index=False)


    def combine_result(self, file_name):

        common_cols = ['Peptide', 'Sequence', 'Prev AA', 'Next AA', 'Peptide Length', 'Charge',
                       'Protein', 'Protein ID', 'Entry Name', 'Gene', 'Protein Description',
                       'Mapped Proteins', 'Mapped Genes']
        if 'peptide' in file_name:
            fdr = self.pep_fdr
            group_key = 'Peptide'
        else:
            fdr = self.seq_fdr
            group_key = 'Sequence'
        group_type = group_key.lower()
        print(f'Generating combined {group_type} result to combined_{group_type}.tsv')

        result_paths = list(self.result_folder.rglob(file_name))
        all_dfs = []
        for result_path in result_paths:
            file_name = Path(result_path).parent.name
            result_df = pd.read_csv(result_path, sep='\t')
            fdr_col = [col for col in result_df.columns if 'Qvalue' in col][0]
            result_df = result_df[(result_df['Label'] == 'Target') * (result_df[fdr_col] <= fdr)]
            if not pd.api.types.is_string_dtype(result_df['Charge']):
                result_df['Charge'] = result_df['Charge'].astype(str)

            cols_to_drop = set(result_df.columns) & {'Label', 'Score', 'Min Rank', fdr_col}
            cols_to_drop |= {col for col in result_df.columns if '_binder' in col}
            result_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            result_df = result_df.rename(columns={'Spectral Count': file_name + ' Spectral Count'})
            if 'Intensity' in result_df.columns:
                result_df = result_df.rename(columns={'Intensity': file_name + ' Intensity'})
            if 'Binder' in result_df.columns:
                result_df = result_df.rename(columns={'Binder': file_name + ' Binder'})
                result_df = result_df.rename(columns={'Best Allele': file_name + ' Allele'})
            all_dfs.append(result_df)

        combined_df = pd.concat(all_dfs, ignore_index=True).merge(pd.DataFrame(columns=common_cols), on=common_cols, how='outer')
        agg_dict = {
            'Peptide': lambda x: ','.join(sorted(set(','.join(x.dropna()).split(',')))),
            'Sequence': 'first',
            'Prev AA': lambda x: ','.join(sorted(set(','.join(x.dropna()).split(',')))),
            'Next AA': lambda x: ','.join(sorted(set(','.join(x.dropna()).split(',')))),
            'Peptide Length': 'first',
            'Charge': lambda x: ','.join(sorted(set(','.join(x.dropna()).split(',')))),
            'Protein': 'first',
            'Protein ID': 'first',
            'Entry Name': 'first',
            'Gene': 'first',
            'Protein Description': 'first',
            'Mapped Proteins': 'first',
            'Mapped Genes': 'first'
        }
        agg_dict.pop(group_key)
        for col in combined_df.columns[13:]:
            agg_dict[col] = lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
        combined_df = combined_df.groupby(group_key, as_index=False).agg(agg_dict)
        intensity_cols = [col for col in combined_df.columns if 'Intensity' in col]
        binder_cols = [col for col in combined_df.columns if 'Binder' in col]
        allele_cols = [col for col in combined_df.columns if 'Allele' in col]
        spectral_count_cols = [col for col in combined_df.columns if 'Spectral Count' in col]
        for col in spectral_count_cols:
            combined_df[col] = combined_df[col].astype('Int64')
        cols = list(combined_df.columns[:13]) + spectral_count_cols + intensity_cols + binder_cols + allele_cols
        combined_df = combined_df[cols]

        print(f'Saving unfiltered combined {group_type}s to combined_{group_type}_unfiltered.tsv.')
        combined_df.to_csv(self.result_folder / f'combined_{group_type}_unfiltered.tsv', sep='\t', index=False)
        if self.combined_high_confidence_sequences is not None:
            combined_len = len(combined_df)
            combined_df = combined_df[combined_df['sequence'].isin(self.combined_high_confidence_sequences)]
            print(f'Saving {len(combined_df)} {group_type}s to combined_{group_type}.tsv ({combined_len} before filtering)')
        else:
            print(f'Saving {len(combined_df)} {group_type}s to combined_{group_type}.tsv')
        combined_df.to_csv(self.result_folder / f'combined_{group_type}.tsv', sep='\t', index=False)
        print('Done.')


    def get_philosopher_reference(self):
        peptide_paths = list(self.result_folder.rglob('peptide.pep.xml'))
        if self.fasta_path is None or len(peptide_paths) == 0 or self.pep_fdr != 0.01:
            return
        philosopher_exe_path = Path(__file__).parent.parent / 'third_party' / 'philosopher_v5.1.0_linux_amd64' / 'philosopher'
        for peptide_path in peptide_paths:
            sample_path = peptide_path.parent
            subprocess.run(f'{philosopher_exe_path} workspace --init', cwd=sample_path, shell=True)
            subprocess.run(f'{philosopher_exe_path} database --annotate {self.fasta_path}', cwd=sample_path, shell=True)
            subprocess.run(f'{philosopher_exe_path} filter --sequential --prot 1 --pep 0.01 --tag rev_ --pepxml peptide.pep.xml --protxml ../combined.prot.xml --razor', cwd=sample_path, shell=True)

        pepxml_paths = ' '.join([str(path) for path in peptide_paths])
        sample_names = ' '.join([path.parent.name for path in peptide_paths])
        subprocess.run(f'{philosopher_exe_path} workspace --init', cwd=self.result_folder, shell=True)
        subprocess.run(f'{philosopher_exe_path} iprophet --decoy rev_ --nonsp --output combined {pepxml_paths}', cwd=self.result_folder, shell=True)
        subprocess.run(f'{philosopher_exe_path} abacus --razor --reprint --tag rev_ --protein --peptide {sample_names}', cwd=self.result_folder, shell=True)

        for peptide_path in peptide_paths:
            sample_path = peptide_path.parent
            subprocess.run(f'{philosopher_exe_path} workspace --clean --nocheck', cwd=sample_path, shell=True)
        subprocess.run(f'{philosopher_exe_path} workspace --clean --nocheck', cwd=self.result_folder, shell=True)
        try:
            reference_df = pd.read_csv(self.result_folder / 'combined_peptide.tsv', sep='\t')
            if 'Sequence' in reference_df.columns:
                self.combined_high_confidence_sequences = reference_df['Sequence'].to_numpy()
        except FileNotFoundError:
            print('Failed to generate the combined peptide list using Philosopher.')
            return


    def build_easypqp_library(self):
        unimod_path = Path(__file__).parent / 'unimod.xml'
        pepxml_paths = list(self.result_folder.rglob('peptide.pep.xml'))
        psms_paths = []
        peaks_paths = []
        commands = []

        # convert pep.xml to .psmpkl and .peakpkl
        for pepxml_path in pepxml_paths:
            input_log_path = pepxml_path.parent / 'input_files.txt'
            mzml_path = read_from_log('mzml=', input_log_path)
            assert mzml_path is not None, f'Failed to find mzml file in {input_log_path}'
            mzml_name = Path(mzml_path).stem.replace('_uncalibrated', '').replace('_calibrated', '')
            psms_path = pepxml_path.parent / f'{mzml_name}.psmpkl'
            peaks_path = pepxml_path.parent / f'{mzml_name}.peakpkl'
            psms_paths.append(str(psms_path.resolve()))
            peaks_paths.append(str(peaks_path.resolve()))
            command = f'easypqp convert --pepxml {pepxml_path} --spectra {mzml_path} --unimod {unimod_path} --psms {psms_path} --peaks {peaks_path}'
            commands.append(command)

        combined_psm_path = self.result_folder / 'combined_psm.tsv'
        combined_pep_path = self.result_folder / 'combined_peptide.tsv'
        with open(combined_psm_path, 'w') as f:
            header_written = False
            for pepxml_path in pepxml_paths:
                psm_path = pepxml_path.parent / f'psm.tsv'
                with open(psm_path) as psm_f:
                    header = psm_f.readline()
                    if not header_written:
                        f.write(header)
                        header_written = True
                    for line in psm_f:
                        f.write(line)

        easypqp_library_path = self.result_folder / 'easypqp_library.tsv'
        command = f'easypqp library --psmtsv {combined_psm_path} --peptidetsv {combined_pep_path} --out {easypqp_library_path} --rt_lowess_fraction 0.0'
        command = command + ' ' + ' '.join(psms_paths) + ' ' + ' '.join(peaks_paths)
        commands.append(command)

        for command in commands:
            subprocess.run(command, shell=True, cwd=self.result_folder)

        # The following code is modified from the FragPipe script
        library_df = pd.read_csv(easypqp_library_path, sep='\t')
        frag_df = library_df['Annotation'].str.extract(r'^([abcxyz])(\d{1,2})(?:-(.*))?\^(\d+)$')
        frag_df.columns = 'FragmentType', 'FragmentSeriesNumber', 'FragmentLossType', 'FragmentCharge'
        frag_df = frag_df.reindex(
            columns=['FragmentType', 'FragmentCharge', 'FragmentSeriesNumber', 'FragmentLossType'], copy=False)

        def interp(t):
            return interp1d(t.iloc[:, 1], t.iloc[:, 0], bounds_error=False)

        rt = library_df['NormalizedRetentionTime'].squeeze()
        align_files = list(self.result_folder.glob('easypqp_rt_alignment_*.alignment_pkl'))
        avg_experimental_rt0 = np.nanmean([interp(pd.read_pickle(f))(rt) for f in align_files], axis=0)
        for e in align_files:
            e.unlink()
        avg_experimental_rt = pd.Series(avg_experimental_rt0, name='AverageExperimentalRetentionTime')
        library_df['ModifiedPeptideSequence'] = library_df['ModifiedPeptideSequence'].str.replace('.(UniMod:', '(UniMod:', regex=False)

        df_lib = pd.concat([library_df, frag_df, avg_experimental_rt], axis=1)
        df_psm = pd.read_csv(combined_psm_path, sep='\t', na_values='')

        df_psm['AllMappedProteins'] = df_psm.apply(
            lambda x: f"{x['Protein']};{x['Mapped Proteins']}" if pd.notna(x['Mapped Proteins']) else x['Protein'],
            axis=1)
        t = dict(zip(df_psm['Peptide'], df_psm['AllMappedProteins']))
        df_lib['AllMappedProteins'] = df_lib['PeptideSequence'].map(t)

        df_psm['AllMappedGenes'] = df_psm.apply(
            lambda x: f"{x['Gene']};{x['Mapped Genes']}" if pd.notna(x['Mapped Genes']) else x['Gene'], axis=1)
        t = dict(zip(df_psm['Peptide'], df_psm['AllMappedGenes']))
        df_lib['AllMappedGenes'] = df_lib['PeptideSequence'].map(t)

        tt = df_psm.apply(lambda x: 1 if pd.isna(x['Mapped Genes']) else 0, axis=1)
        ttt = dict(zip(df_psm['Peptide'], tt))

        df_lib['Proteotypic'] = df_lib['PeptideSequence'].map(ttt)
        df_lib['Proteotypic'] = df_lib['Proteotypic'].fillna(0)
        df_lib['Proteotypic'] = df_lib['Proteotypic'].astype(int)

        library_path = self.result_folder / 'library.tsv'
        df_lib.to_csv(library_path, sep='\t', index=False)

        for path in self.result_folder.glob('easypqp_*'):
            path.unlink()
        for path in psms_paths:
            Path(path).unlink()
        for path in peaks_paths:
            Path(path).unlink()
        for path in self.result_folder.rglob('*_run_peaks.tsv'):
            path.unlink()


    def add_ionquant_intensity(self):
        psm_paths = list(self.result_folder.rglob('psm.tsv'))
        if len(psm_paths) == 0:
            print('No psm.tsv found. Skipping ionquant.')
            return

        modmasses = set()
        ionmob = False
        spec_dir = str(Path(read_from_log('mzml=', psm_paths[0].parent / 'input_files.txt')).parent.resolve())
        for psm_path in psm_paths:
            # backup psm.tsv and peptide.tsv
            print(f'Backing up psm.tsv and peptide.tsv in {psm_path.parent}')
            psm_bak_path = psm_path.parent / 'psm_backup.tsv'
            psm_df = pd.read_csv(psm_path, sep='\t')
            psm_df.to_csv(psm_bak_path, sep='\t', index=False)
            pep_path = psm_path.parent / 'peptide.tsv'
            pep_bak_path = psm_path.parent / 'peptide_backup.tsv'
            pep_df = pd.read_csv(pep_path, sep='\t')
            pep_df.to_csv(pep_bak_path, sep='\t', index=False)

            psm_df.rename(columns={'Peptide': 'Modified Peptide', 'Sequence': 'Peptide'}, inplace=True)
            psm_df['Is Unique'] = psm_df['Mapped Proteins'].isna().map(lambda x: 'true' if x else 'false')
            psm_df['Protein Start'] = 0
            psm_df['Protein End'] = 0
            psm_df['Expectation'] = 1
            psm_df['Hyperscore'] = 1
            psm_df['PeptideProphet Probability'] = psm_df['Score']
            psm_df.to_csv(psm_path, sep='\t', index=False)
            for peptide in psm_df['Modified Peptide']:
                modmasses.update(set(extract_mod_mass_diff(peptide)))
            if not ionmob and psm_df['Ion Mobility'].notna().any():
                ionmob = True

        modlist_path = self.result_folder / 'ionquant_modmasses.txt'
        with open(modlist_path, 'w') as f:
            for modmass in sorted(modmasses):
                f.write(f'{modmass}\n')

        filelist_path = self.result_folder / 'ionquant_filelist.txt'
        with open(filelist_path, 'w') as f:
            f.write('flag\tvalue\n')
            for psm_path in psm_paths:
                f.write(f'--psm\t{psm_path}\n')
            f.write(f'--specdir\t{spec_dir}\n')

        java_exe_path = Path(__file__).parent.parent / 'third_party' / 'jre-17.0.14' / 'bin' / 'java'
        ionquant_exe_path = Path(__file__).parent.parent / 'third_party' / 'IonQuant-1.11.11' / 'IonQuant-1.11.11.jar'
        msfragger_path = Path(__file__).parent.parent / 'third_party' / 'MSFragger-4.1'
        bruker_path = msfragger_path / 'ext' / 'bruker'
        thermo_path = msfragger_path / 'ext' / 'thermo'
        command = (f'{java_exe_path} -jar -Dlibs.bruker.dir={bruker_path} -Dlibs.thermo.dir={thermo_path}'
                   f' {ionquant_exe_path} --isotol 20.0 --ionmobility {int(ionmob)} --multidir .'
                   f' --filelist {filelist_path} --modlist {modlist_path}')
        subprocess.run(command, shell=True, cwd='/tmp')
        modlist_path.unlink()
        filelist_path.unlink()

        for psm_path in psm_paths:
            pep_path = psm_path.parent / 'peptide.tsv'
            seq_path = psm_path.parent / 'sequence.tsv'
            psm_bak_path = psm_path.parent / 'psm_backup.tsv'
            pep_bak_path = psm_path.parent / 'peptide_backup.tsv'
            psm_bak_df = pd.read_csv(psm_bak_path, sep='\t')
            pep_bak_df = pd.read_csv(pep_bak_path, sep='\t')

            psm_df = pd.read_csv(psm_path, sep='\t')
            quant_df = pd.DataFrame({
                'Spectrum': psm_df['Spectrum'],
                'Peptide': psm_df['Modified Peptide'],
                'Sequence': psm_df['Peptide'],
                'Intensity': psm_df['Intensity']
            })

            if 'Intensity' in psm_bak_df.columns:
                psm_bak_df.drop(columns=['Intensity'], inplace=True)
            psm_bak_df = psm_bak_df.merge(quant_df, on=['Spectrum', 'Peptide', 'Sequence'], how='left')
            psm_bak_df.to_csv(psm_path, sep='\t', index=False)
            psm_bak_path.unlink()

            pep_quant_df = quant_df.groupby('Peptide', as_index=False).agg({'Intensity': 'max'})
            if 'Intensity' in pep_bak_df.columns:
                pep_bak_df.drop(columns=['Intensity'], inplace=True)
            pep_bak_df = pep_bak_df.merge(pep_quant_df, on='Peptide', how='left')
            pep_bak_df.to_csv(pep_path, sep='\t', index=False)
            pep_bak_path.unlink()

            seq_quant_df = quant_df.groupby('Sequence', as_index=False).agg({'Intensity': 'max'})
            seq_df = pd.read_csv(seq_path, sep='\t')
            if 'Intensity' in seq_df.columns:
                seq_df.drop(columns=['Intensity'], inplace=True)
            seq_df = seq_df.merge(seq_quant_df, on='Sequence', how='left')
            seq_df.to_csv(seq_path, sep='\t', index=False)


    def run(self):

        if self.infer_protein:
            self.fasta_map = self.prepare_fasta_map()
            self.seq_prot_df = self.do_protein_inference()
            self.update_proteins('psm.tsv')
            self.update_proteins('peptide.tsv')
            self.update_proteins('sequence.tsv')

        if self.remove_contaminant:
            self.remove_contaminants('psm.tsv')
            self.remove_contaminants('peptide.tsv')
            self.remove_contaminants('sequence.tsv')

        if self.quant:
            self.add_ionquant_intensity()

        if self.control_combine_fdr:
            self.get_philosopher_reference()
        self.combine_result('peptide.tsv')
        self.combine_result('sequence.tsv')

        if self.build_library:
            self.build_easypqp_library()


if __name__ == '__main__':
    combined_reporter = CombinedReporter(result_folder='/mnt/e/data/JY_HLA-II/mhcbooster_comb',
                                         fasta_path='/mnt/d/data/JY_1_10_25M/2024-09-03-decoys-contam-Human_EBV_GD1_B95.fasta',
                                         infer_protein=True,
                                         remove_contaminant=False,
                                         control_combine_fdr=False)
    combined_reporter.run()
    # combined_reporter.get_philosopher_reference()
    # combined_reporter.combine_result('peptide.tsv')
    # combined_reporter.combine_result('sequence.tsv')