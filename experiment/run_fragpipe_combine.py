
import subprocess

from pathlib import Path

def run_combine(result_folder, fasta_path):
    peptide_paths = list(result_folder.rglob('*.pep.xml'))

    philosopher_exe_path = Path(__file__).parent.parent / 'mhcbooster' / 'third_party' / 'philosopher_v5.1.0_linux_amd64' / 'philosopher'

    for peptide_path in peptide_paths:
        sample_path = peptide_path.parent
        subprocess.run(f'{philosopher_exe_path} workspace --clean --nocheck', cwd=sample_path, shell=True)
    subprocess.run(f'{philosopher_exe_path} workspace --clean --nocheck', cwd=result_folder, shell=True)

    for peptide_path in peptide_paths:
        sample_path = peptide_path.parent
        subprocess.run(f'{philosopher_exe_path} workspace --init', cwd=sample_path, shell=True)
        subprocess.run(f'{philosopher_exe_path} database --annotate {fasta_path}', cwd=sample_path, shell=True)
        subprocess.run \
            (f'{philosopher_exe_path} filter --sequential --prot 1 --pep 0.01 --tag rev_ --pepxml {peptide_path.name} --protxml ../combined.prot.xml --razor', cwd=sample_path, shell=True)

    pepxml_paths = ' '.join([str(path) for path in peptide_paths])
    sample_names = ' '.join([path.parent.name for path in peptide_paths])
    subprocess.run(f'{philosopher_exe_path} workspace --init', cwd=result_folder, shell=True)
    subprocess.run(f'{philosopher_exe_path} iprophet --decoy rev_ --nonsp --output combined {pepxml_paths}', cwd=result_folder, shell=True)
    # subprocess.run(f'{philosopher_exe_path} abacus --razor --reprint --tag rev_ --protein --peptide {sample_names}', cwd=result_folder, shell=True)

    # for peptide_path in peptide_paths:
    #     sample_path = peptide_path.parent
    #     subprocess.run(f'{philosopher_exe_path} workspace --clean --nocheck', cwd=sample_path, shell=True)
    # subprocess.run(f'{philosopher_exe_path} workspace --clean --nocheck', cwd=result_folder, shell=True)

if __name__ == '__main__':
    result_folder = Path('/mnt/e/data/PXD058436/Search_all_II_0930')
    fasta_path = Path('/mnt/d/data/Library/2024-10-15-decoys-contam-Human_EBV_GD1_B95_Leukemia.fasta.fas')
    run_combine(result_folder, fasta_path)
