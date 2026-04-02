
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mhcbooster.main_mhcbooster import run_mhcbooster

alleles = ['HLA-A0201', 'HLA-B0702', 'HLA-C0702'] # HLA-A0201 HLA-B0702 HLA-C0702
# HLA-DPA1*01:03 HLA-DPB1*02:01 HLA-DPB1*04:02 HLA-DQA1*01:03 HLA-DQA1*03:01 HLA-DQB1*03:02 HLA-DQB1*06:03 HLA-DRB1*04:04 HLA-DRB1*13:01 HLA-DRB4*04 HLA-DRB5*02
# alleles = ['HLA-DPA1*01:03, HLA-DPB1*02:01, HLA-DPB1*04:02, HLA-DQA1*01:03, HLA-DQA1*03:01, HLA-DQB1*03:02, HLA-DQB1*06:03, HLA-DRB1*04:04, HLA-DRB1*13:01, HLA-DRB4*04, HLA-DRB5*02']
pin_files = Path('/mnt/f/JY_DIA/search0102').rglob('*.pin')
mzml_folder = Path('/mnt/f/JY_DIA/diatracer')
output_folder = Path('/mnt/f/JY_DIA/mhcbooster')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

auto_predict_predictor = False
fasta_path = '/mnt/d/data/Library/2024-10-15-decoys-contam-Human_EBV_GD1_B95_Leukemia.fasta.fas'
rt_predictors = ['Prosit_2019_irt', 'Prosit_2024_irt_cit']
ms2_predictors = ['ms2pip_timsTOF2024', 'Prosit_2023_intensity_timsTOF']
# ccs_predictors = ['IM2Deep']
app_predictors = ['mhcflurry', 'netmhcpan']

# ms2_predictors = []
ccs_predictors = []
app_predictors = []

run_mhcbooster(pin_files, sequence_encoding=True, alleles=alleles, mhc_class='I', app_predictors=app_predictors,
    auto_predict_predictor=auto_predict_predictor, rt_predictors=rt_predictors, ms2_predictors=ms2_predictors,
    ccs_predictors=ccs_predictors, fine_tune=False, fasta_path=fasta_path, mzml_folder=mzml_folder,
    output_folder=output_folder)