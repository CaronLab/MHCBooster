import logging
import tempfile

from pathlib import Path
import pandas as pd
from ms2rescore.rescoring_engines import mokapot
from ms2rescore.report.charts import score_histogram
from ms2rescore.feature_generators.im2deep import IM2DeepFeatureGenerator
from ms2rescore.feature_generators.basic import BasicFeatureGenerator
from ms2rescore.feature_generators.ms2pip import MS2PIPFeatureGenerator
from ms2rescore.feature_generators.deeplc import DeepLCFeatureGenerator
from psm_utils.io import read_file

logging.basicConfig(level=logging.INFO)


def run_ms2rescore(pin_folder, mzml_folder, output_folder):
    pin_files = list(pin_folder.rglob('*.pin'))
    mzml_files = list(mzml_folder.rglob('*.mzML'))
    output_folder.mkdir(parents=True, exist_ok=True)
    mzml_map = {}
    for mzml_file in mzml_files:
        key = mzml_file.stem.replace('_uncalibrated', '')
        mzml_map[key] = str(mzml_file)

    for pin_file in pin_files:
        if 'edited' in pin_file.name:
            continue
        print(pin_file.stem)
        file_data = list(open(pin_file))
        pep_col_idx = -1
        for i, key in enumerate(file_data[0].split('\t')):
            if 'peptide' in key.lower():
                pep_col_idx = i
                break

        with tempfile.NamedTemporaryFile('w', delete=False) as tmp_file:
            tmp_file.write(file_data[0])
            for i in range(1, len(file_data)):
                line_split = file_data[i].split('\t')
                pep_seq = line_split[pep_col_idx]
                if pep_seq[2] == 'n':
                    pep_seq = pep_seq[:2] + pep_seq[12:]
                formatted_seq = pep_seq[:-3] + '/' + pep_seq[-3:]
                line_split[pep_col_idx] = formatted_seq
                line_split[0] = line_split[0][:-4]
                tmp_file.write('\t'.join(line_split))

            psm_list = read_file(tmp_file.name, filetype="percolator")
            psm_list["spectrum_id"] = [str(spec_id).split('.')[-2] for spec_id in psm_list["spectrum_id"]]
            psm_list.rename_modifications({
                '42.0106': 'Acetyl',
                '57.0215': 'Carbamidomethyl',
                '79.9663':  'Phospho',
                '39.9950':  'Pyro-carbamidomethyl',
                '-18.0106': 'Glu->pyro-Glu',
                '-17.0265': 'Gln->pyro-Glu',
                '15.9949':  'Oxidation',
                '114.0429': 'GG',
                '119.0041': 'Cysteinyl',
                '229.1629': 'TMT6plex'
            })
            for psm in psm_list:
                psm.retention_time = psm.rescoring_features['retentiontime']

        basic_fgen = BasicFeatureGenerator()
        basic_fgen.add_features(psm_list)


        ms2pip_fgen = MS2PIPFeatureGenerator(
            model="HCD",
            ms2_tolerance=0.02,
            spectrum_path=mzml_map[pin_file.stem],
            spectrum_id_pattern=r".*scan=(\d+)$",
            processes=1,
        )

        ms2pip_fgen.add_features(psm_list)

        deeplc_fgen = DeepLCFeatureGenerator(
            lower_score_is_better=False,
            calibration_set_size=0.15,
            spectrum_path=None,
            processes=1,
            deeplc_retrain=False,
        )
        deeplc_fgen.add_features(psm_list)

        # maxquant_fgen = MaxQuantFeatureGenerator()
        # maxquant_fgen.add_features(psm_list)

        # im2deep_fgen = IM2DeepFeatureGenerator(
        #     spectrum_path="/mnt/d/gui_test/PXD052187/mzml/IP0040_11MAI2022_JY_MHC1_HUMAN_S4_2PELLETS_KK_SERIAL_DIL_PT_1_R1_uncalibrated.mzML",
        #     spectrum_id_pattern=r".*scan=(\d+)$")
        # im2deep_fgen.add_features(psm_list)

        mokapot.rescore(psm_list)
        fig = score_histogram(psm_list.to_dataframe())
        fig.show()

        psm_list = psm_list.to_dataframe()
        psm_list['peptide_qvalue'] = 1
        for i in range(len(psm_list)):
            psm_list.loc[i, 'peptide_qvalue'] = psm_list.loc[i, 'metadata']['peptide_qvalue']

        save_path = output_folder / (pin_file.stem + '_result.csv')
        psm_list.to_csv(save_path, index=False)

if __name__ == '__main__':
    pin_folder = Path('/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M_rerun/msfragger/pin')
    mzml_folder = Path('/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M_rerun/msfragger/mzml')
    output_folder = Path('/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M_rerun/msfragger/ms2rescore')
    run_ms2rescore(pin_folder, mzml_folder, output_folder)