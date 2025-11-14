
# MHCBooster
Immunopeptide identification in data-dependent acquisition (DDA) is crucial for
understanding immune responses and developing targeted therapies.
However, enhancing both the quantity and quality of identified epitopes,
particularly from low-input samples, presents a significant challenge.
To address this, we developed MHCBooster,
an AI-powered tool that leverages deep learning models to enhance immunopeptide identification.
By providing reliability measurements across key dimensions like retention time (RT), MS2, ion mobility (IM),
and antigen processing and presentation (APP) for both MHC-I and MHC-II peptides,
MHCBooster increases the sensitivity and specificity in epitope identifications,
especially in lower-input scenarios.
MHCBooster features a graphical user interface and is also available via command line or as a Python package. 

### Install MHCBooster

```terminal
# create a conda environment for MHCBooster
conda create -n mhcbooster python==3.10
conda activate mhcbooster

# install MHCBooster
pip install mhcbooster==2.1.2


# WSL (Ubuntu 20.04)
sudo apt update
sudo apt install libxcb-xinerama0 libxcb-cursor0 libx11-xcb1 qt5-qmake qtbase5-dev qtchooser qtbase5-dev-tools   # GUI
sudo apt install tcsh    # NetMHCpan env
mhcflurry-downloads fetch models_class1_presentation    # MHCflurry env
```

### Install third party tools
MHCBooster utilizes a variety of tools for RT, MS2, and CCS scoring. 
Some of these tools are governed by strict licenses and must be manually downloaded and installed.
**By downloading and installing the third-party packages, you agree to abide by their respective licenses.**
Third-party tools only need to be installed once.

#### Install by MHCBooster GUI (preferred)

1. Start GUI
```terminal
mhcbooster-gui
```
2. Click the **Configuration** tab.
3. Click the **Download** button to obtain the required tools.
2. Specify the installation paths by clicking the **Browse** button.
3. Click the **Install to MHCBooster** button to initiate automatic unzipping and installation.

A progress bar will appear during the installation process. Once the installation is complete,
the progress bar will disappear, and the tool paths will be updated to their actual installation locations. 
Each time MHCBooster starts, it will scan the installation folder and display the correct paths in the input box,
confirming successful installation.

#### Install by command line 
1. Download the packages manually.
```text
# Download links
MSFragger (MSFragger-4.1.zip): https://msfragger-upgrader.nesvilab.org/upgrader/
AutoRT (AutoRT-master.zip): https://github.com/bzhanglab/AutoRT/archive/refs/heads/master.zip
BigMHC (bigmhc-master.zip): https://github.com/KarchinLab/bigmhc/archive/refs/heads/master.zip
NetMHCpan (netMHCpan-4.1b.Linux.tar.gz): https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=4.1&packageversion=4.1b&platform=Linux
NetMHCIIpan (netMHCIIpan-4.3e.Linux.tar.gz): https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCIIpan&version=4.3&packageversion=4.3e&platform=Linux
MixMHC2pred (MixMHC2pred-2.0.zip): https://github.com/GfellerLab/MixMHC2pred/releases/download/v2.0.2.2/MixMHC2pred-2.0.zip

# Note: Please download these packages with a browser.
#       Downloading by scripts will lead to wrong file names and could not be identified by package-installer.
```
2. Install by scripts
```terminal
# Install packages from download folder (WSL/Linux style)
mhcbooster-package-installer /path/to/folder_of_third_party_zip_files
```

### Run MHCBooster by GUI

1. Start GUI
```terminal
mhcbooster-gui
```
2. Follow instructions on the GUI page

### Run MHCBooster by command line
```terminal
mhcbooster -n 23 --app_predictors NetMHCpan MHCflurry --alleles HLA-A0201 HLA-B0702 HLA-C0702 --mhc_class I \
 --rt_predictors Prosit_2024_irt_cit --ms2_predictors ms2pip_timsTOF2024 --ccs_predictors IM2Deep \
 --encode_peptide_sequences --infer_protein --remove_decoy --psm_fdr 0.01 --pep_fdr 0.01 --seq_fdr 0.01 \
 --koina_server_url koina.wilhelmlab.org:443 --input /path/to/pin_folder --output_dir /path/to/output_folder \
 --fasta_path /path/to/fasta_with_decoy --mzml_dir /path/to/mzml
```

### Run MHCBooster by Python API

```python
from pathlib import Path
from mhcbooster.main_mhcbooster import run_mhcbooster

pin_files = Path('/path/to/pin_folder').rglob('*.pin')
mzml_folder = '/path/to/mzml_folder'
output_folder = '/path/to/output_folder'
fasta_path = '/path/to/fasta'

alleles = ['HLA-A0201', 'HLA-B0702', 'HLA-C0702']
app_predictors = ['mhcflurry', 'netmhcpan']

auto_predict_predictor = True
rt_predictors = []
ms2_predictors = []
ccs_predictors = []

run_mhcbooster(pin_files, sequence_encoding=True, alleles=alleles, mhc_class='I', app_predictors=app_predictors,
    auto_predict_predictor=auto_predict_predictor, rt_predictors=rt_predictors, ms2_predictors=ms2_predictors,
    ccs_predictors=ccs_predictors, fine_tune=False, fasta_path=fasta_path, mzml_folder=mzml_folder,
    output_folder=output_folder)
```