import subprocess
import re

import numpy as np
import pandas as pd
from pathlib import Path
from pyopenms import MzMLFile, MSExperiment


def split(result_path):
    result_path = Path(result_path)
    result_df = pd.read_csv(result_path, sep='\t')

    if result_df['ln(precursor_ppm)'].isna().any():
        result_df.dropna(subset=['ln(precursor_ppm)'], inplace=True)

    file_names = result_df['FileName'].drop_duplicates().tolist()
    for file_name in file_names:
        run_df = result_df[result_df['FileName'] == file_name]
        run_path = (result_path.parent/file_name).with_suffix('.pin')
        run_df.to_csv(run_path, sep='\t', index=False)
    result_path.rename(result_path.with_suffix('.pin.tmp'))


def get_sage_command(param_path, fasta_path, mzml_path):
    sage_exe_path = Path(__file__).parent.parent / 'third_party' / 'sage' / 'target' / 'release' / 'sage'
    output_path = Path(mzml_path) / 'sage'
    commands = []
    for mzml_file in Path(mzml_path).glob('*.mzML'):
        filename = mzml_file.stem
        command = f'{sage_exe_path} {param_path} -f {fasta_path} --write-pin {mzml_file} -o {output_path / filename}'
        commands.append(command)
        command = f'mv {output_path / filename / "results.sage.pin"} {output_path / (filename + ".pin")}'
        commands.append(command)
    return commands

def convert_to_msfragger_pin(result_path):
    for path in Path(result_path).rglob('*.pin'):
        if path.name == 'results.sage.pin':
            continue
        print(f'Converting {path.name} to MSFragger pin format...')
        result_df = pd.read_csv(path, sep='\t')
        result_df = result_df.rename(columns={'posterior_error': 'log10_evalue'})
        if 'FileName' in result_df.columns:
            result_df = result_df.drop(columns=['FileName'])
        if 'ln(precursor_ppm)' in result_df.columns:
            result_df['ln(precursor_ppm)'] = result_df['ln(precursor_ppm)'].fillna(np.log(20))
        charges = np.ones(len(result_df), dtype=int)
        for col in result_df.columns:
            if 'z=other' in col:
                continue
            if 'z=' in col:
                charges[result_df[col] == 1] = int(re.findall(r'\d+', col)[0])
        result_df['Peptide'] = result_df['Peptide'].apply(lambda x: x.replace('+', '').replace('[42.0106]-', 'n[42.0106]'))
        result_df['Peptide'] = '-.' + result_df['Peptide'] + charges.astype(str) + '.-'
        if len(result_df) > 0:
            if not str(result_df.loc[0, 'ScanNr']).isnumeric():
                mzml_path = str(path.parent.parent / path.stem) + '.mzML'
                # Build nativeID -> 1-based index map using pyopenms
                exp = MSExperiment()
                MzMLFile().load(mzml_path, exp)
                scannr_map = {}
                for i in range(exp.getNrSpectra()):
                    scannr_map[exp.getSpectrum(i).getNativeID()] = i + 1
                result_df['ScanNr'] = result_df['ScanNr'].apply(lambda x: str(scannr_map[x]))
            else:
                result_df['ScanNr'] = result_df['ScanNr'].astype(str)
        result_df['SpecId'] = (path.stem + '.' + result_df['ScanNr'] + '.' + result_df['ScanNr']
                               + '.' + charges.astype(str)) + '_1'
        result_df.to_csv(path, sep='\t', index=False)
        print('Done.')


def run_sage(param_path, fasta_path, mzml_path):
    commands = get_sage_command(param_path, fasta_path, mzml_path)
    for command in commands:
        subprocess.run(command, shell=True)
    convert_to_msfragger_pin(mzml_path)

if __name__ == '__main__':
    param_path = Path('/mnt/d/workspace/mhc-booster/pipeline_setup/sage_orbi.json')
    fasta_path = Path('/mnt/d/data/HL-60/2024-11-18-decoys-contam-2024_11_18_human.fasta.fas')
    mzml_path = Path('/mnt/d/data/HL-60/mzML')
    run_sage(param_path, fasta_path, mzml_path)
