
import os
import subprocess
import sys
import select
import tempfile
import webbrowser
import time
import psutil
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QPixmap, QIcon, QTextCursor, QFont
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                               QVBoxLayout, QLineEdit, QFileDialog, QHBoxLayout,
                               QCheckBox, QGridLayout, QSpinBox, QGroupBox,
                               QMessageBox, QTextEdit, QTabWidget, QStackedWidget, QRadioButton,
                               QDoubleSpinBox, QProgressBar, QTextBrowser)


ROOT_PATH = Path(__file__).parent.parent.parent
sys.path.append(ROOT_PATH.as_posix())
from mhcbooster import __version__
from mhcbooster.utils.package_installer import *
from mhcbooster.adapter.msfragger_adapter import get_msfragger_command
from mhcbooster.adapter.sage_adapter import get_sage_command


def grid_layout(label, elements, n_same_row=4):
    g_layout = QGridLayout()
    g_layout.setHorizontalSpacing(8)
    g_layout.setVerticalSpacing(2)
    for i, checkbox in enumerate(elements):
        row = i // n_same_row  # Every 5 checkboxes will be placed in a new row
        col = i % n_same_row  # Columns will repeat after every 5 checkboxes (like a 5-column grid)
        g_layout.addWidget(checkbox, row, col)
        g_layout.setColumnMinimumWidth(col, 220)
    h_layout = QHBoxLayout()
    h_layout.addWidget(label)
    h_layout.setAlignment(label, Qt.AlignTop)
    h_layout.addLayout(g_layout)
    h_layout.setAlignment(Qt.AlignLeft)
    return h_layout


class MhcBoosterGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("""
                QGroupBox {
                    border: 1px solid #A9A9A9;
                    margin-top: 1ex;
                    padding: 5px;
                    font: bold 12px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left; /* position at the top left edge */
                    padding: 0 3px; /* padding from the border */
                    left: 10px;
                }
                QTabWidget::pane {
                    border: 1px solid lightgray;  /* Remove the tab box */
                    border-left: none;
                    border-right: none;
                    border-bottom: none;
                }
            """)

        # GUI window
        self.setWindowTitle(f'MHCBooster {__version__}')
        self.setWindowIcon(QIcon(str(Path(__file__).parent/'caronlab_icon.png')))
        # self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 20, 30, 10) # left, top, right, bottom
        layout.setSpacing(10)

        ### INTRODUCTION
        logo_lab_label = QLabel()
        logo_pix_map = QPixmap(str(Path(__file__).parent/'caronlab.png')).scaled(160, 120, Qt.KeepAspectRatio)
        logo_lab_label.setPixmap(logo_pix_map)
        logo_lab_label.resize(logo_pix_map.size())
        intro_label = QLabel('<p style="line-height: 1.2;"><b>MHCBooster: An AI-powered Software to Boost DDA-based Immunopeptide Identification.</b><br>'
                             'Authors: Ruimin Wang, Etienne Caron.<br>'
                             'CaronLab: <a href="https://www.caronlab.org/">https://www.caronlab.org/</a><br>'
                             'Affiliations: Yale School of Medicine, Department of Immunobiology.</p>')
        intro_label.setOpenExternalLinks(True)
        intro_layout = QHBoxLayout()
        intro_layout.setAlignment(Qt.AlignLeft)
        intro_layout.addWidget(logo_lab_label)
        intro_layout.addSpacing(50)
        intro_layout.addWidget(intro_label)
        layout.addLayout(intro_layout)


        self.tab_widget = QTabWidget()
        self.add_main_tab()
        # self.add_reporter_tab()
        self.add_config_tab()
        layout.addWidget(self.tab_widget)

        ### Footnote
        foot_label = QLabel('CaronLab 2024')
        foot_label.setAlignment(Qt.AlignRight)
        layout.addWidget(foot_label)

        self.setLayout(layout)

        self.restore_settings()

    def closeEvent(self, event, /):
        self.save_settings()
        event.accept()

    def add_main_tab(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 20, 10, 0) # left, top, right, bottom
        main_layout.setSpacing(20)

        ### FILE MANAGEMENT
        file_groupbox = QGroupBox('Input / Output')
        file_group_layout = QVBoxLayout()

        input_format_layout = QHBoxLayout()
        input_format_layout.setContentsMargins(0, 0, 0, 0)
        input_format_layout.setSpacing(30)
        self.psm_radiobutton = QRadioButton('Run from PSM')
        self.raw_radiobutton = QRadioButton('Run from RAW (MSFragger)')
        self.psm_radiobutton.setChecked(True)
        input_format_layout.addWidget(self.psm_radiobutton)
        input_format_layout.addWidget(self.raw_radiobutton)
        input_format_layout.setAlignment(Qt.AlignLeft)
        file_group_layout.addLayout(input_format_layout)

        psm_group_layout = QVBoxLayout()
        psm_group_layout.setContentsMargins(0, 0, 0, 0)
        psm_group_layout.setSpacing(5)

        # PIN folder
        psm_path_label = QLabel('PSM folder: \t')
        self.psm_inputbox = QLineEdit()
        self.psm_inputbox.setPlaceholderText("Select input folder containing .pin files from Comet or MSFragger...")
        self.psm_button = QPushButton("Select")
        self.psm_button.clicked.connect(self.open_folder_dialog)
        psm_layout = QHBoxLayout()
        psm_layout.addWidget(psm_path_label)
        psm_layout.addWidget(self.psm_inputbox)
        psm_layout.addWidget(self.psm_button)
        psm_group_layout.addLayout(psm_layout)

        # Fasta file
        psm_fasta_label = QLabel('FASTA file: \t')
        self.psm_fasta_inputbox = QLineEdit()
        self.psm_fasta_inputbox.setPlaceholderText('Select .fasta or .fasta.fas file with decoys (for protein inference)...')
        self.psm_fasta_button = QPushButton("Select")
        self.psm_fasta_button.clicked.connect(self.open_file_dialog)
        psm_fasta_layout = QHBoxLayout()
        psm_fasta_layout.addWidget(psm_fasta_label)
        psm_fasta_layout.addWidget(self.psm_fasta_inputbox)
        psm_fasta_layout.addWidget(self.psm_fasta_button)
        psm_group_layout.addLayout(psm_fasta_layout)

        # MzML folder
        mzml_label = QLabel('mzML folder: \t')
        self.psm_mzml_inputbox = QLineEdit()
        self.psm_mzml_inputbox.setPlaceholderText('Select mzML folder containing .mzML files with the same name as PSM files (for MS2, CCS rescoring)...')
        self.mzml_button = QPushButton("Select")
        self.mzml_button.clicked.connect(self.open_folder_dialog)
        mzml_layout = QHBoxLayout()
        mzml_layout.addWidget(mzml_label)
        mzml_layout.addWidget(self.psm_mzml_inputbox)
        mzml_layout.addWidget(self.mzml_button)
        psm_group_layout.addLayout(mzml_layout)

        # Output folder
        psm_output_label = QLabel('Output folder: \t')
        self.psm_output_inputbox = QLineEdit()
        self.psm_output_inputbox.setPlaceholderText('Select output folder...')
        self.psm_output_button = QPushButton("Select")
        self.psm_output_button.clicked.connect(self.open_folder_dialog)
        psm_output_layout = QHBoxLayout()
        psm_output_layout.addWidget(psm_output_label)
        psm_output_layout.addWidget(self.psm_output_inputbox)
        psm_output_layout.addWidget(self.psm_output_button)
        psm_group_layout.addLayout(psm_output_layout)

        raw_group_layout = QVBoxLayout()
        raw_group_layout.setContentsMargins(0, 0, 0, 0)
        raw_group_layout.setSpacing(5)

        # RAW folder
        raw_path_label = QLabel('RAW folder: \t')
        self.raw_inputbox = QLineEdit()
        self.raw_inputbox.setPlaceholderText("Select input folder containing .raw/.d files...")
        self.raw_button = QPushButton("Select")
        self.raw_button.clicked.connect(self.open_folder_dialog)
        raw_layout = QHBoxLayout()
        raw_layout.addWidget(raw_path_label)
        raw_layout.addWidget(self.raw_inputbox)
        raw_layout.addWidget(self.raw_button)
        raw_group_layout.addLayout(raw_layout)

        # fasta file
        raw_fasta_label = QLabel('FASTA file: \t')
        self.raw_fasta_inputbox = QLineEdit()
        self.raw_fasta_inputbox.setPlaceholderText('Select .fasta or .fasta.fas file...')
        self.raw_fasta_button = QPushButton("Select")
        self.raw_fasta_button.clicked.connect(self.open_file_dialog)
        raw_fasta_layout = QHBoxLayout()
        raw_fasta_layout.addWidget(raw_fasta_label)
        raw_fasta_layout.addWidget(self.raw_fasta_inputbox)
        raw_fasta_layout.addWidget(self.raw_fasta_button)
        raw_group_layout.addLayout(raw_fasta_layout)

        # parameter file
        param_label = QLabel('Parameter file: \t')
        self.raw_param_inputbox = QLineEdit()
        self.raw_param_inputbox.setPlaceholderText('Select fragger.params file...')
        self.param_button = QPushButton("Select")
        self.param_button.clicked.connect(self.open_file_dialog)
        param_layout = QHBoxLayout()
        param_layout.addWidget(param_label)
        param_layout.addWidget(self.raw_param_inputbox)
        param_layout.addWidget(self.param_button)
        raw_group_layout.addLayout(param_layout)

        # Output folder
        raw_output_label = QLabel('Output folder: \t')
        self.raw_output_inputbox = QLineEdit()
        self.raw_output_inputbox.setPlaceholderText('Select output folder...')
        self.raw_output_button = QPushButton("Select")
        self.raw_output_button.clicked.connect(self.open_folder_dialog)
        raw_output_layout = QHBoxLayout()
        raw_output_layout.addWidget(raw_output_label)
        raw_output_layout.addWidget(self.raw_output_inputbox)
        raw_output_layout.addWidget(self.raw_output_button)
        raw_group_layout.addLayout(raw_output_layout)

        pin_group_widget = QWidget()
        pin_group_widget.setLayout(psm_group_layout)
        raw_group_widget = QWidget()
        raw_group_widget.setLayout(raw_group_layout)

        self.input_stacked_widget = QStackedWidget()
        self.input_stacked_widget.addWidget(pin_group_widget)
        self.input_stacked_widget.addWidget(raw_group_widget)

        file_group_layout.addWidget(self.input_stacked_widget)
        self.psm_radiobutton.clicked.connect(lambda: self.input_stacked_widget.setCurrentIndex(0))
        self.raw_radiobutton.clicked.connect(lambda: self.input_stacked_widget.setCurrentIndex(1))
        file_groupbox.setLayout(file_group_layout)
        main_layout.addWidget(file_groupbox)

        ### MHC specific SCORES
        mhc_groupbox = QGroupBox('MHC Predictors')
        mhc_group_layout = QVBoxLayout()
        # mhc_group_layout.insertSpacing(0, 5)
        mhc_group_layout.setSpacing(5)

        # APP score
        self.mhc_class = None
        mhc_I_label = QLabel('MHC-I Score:\t')
        mhc_I_models = ['NetMHCpan', 'MHCflurry', 'BigMHC']
        self.checkboxes_mhc_I = [QCheckBox(model) for model in mhc_I_models]
        for checkbox in self.checkboxes_mhc_I:
            checkbox.toggled.connect(self.on_mhc_I_checkbox_toggled)
        mhc_I_layout = grid_layout(mhc_I_label, self.checkboxes_mhc_I)
        mhc_group_layout.addLayout(mhc_I_layout)
        mhc_II_label = QLabel('MHC-II Score:\t')
        mhc_II_models = ['NetMHCIIpan', 'MixMHC2pred']
        self.checkboxes_mhc_II = [QCheckBox(model) for model in mhc_II_models]
        for checkbox in self.checkboxes_mhc_II:
            checkbox.toggled.connect(self.on_mhc_II_checkbox_toggled)
        mhc_II_layout = grid_layout(mhc_II_label, self.checkboxes_mhc_II)
        mhc_group_layout.addLayout(mhc_II_layout)

        # Alleles
        allele_label = QLabel('Alleles: \t   ')
        self.allele_inputbox = QLineEdit()
        self.allele_inputbox.setPlaceholderText('Input alleles separated with space (e.g. HLA-A0101 DQB1*05:01) or Select allele map file...')
        self.allele_button = QPushButton("Select")
        self.allele_button.clicked.connect(self.open_file_dialog)
        allele_layout = QHBoxLayout()
        allele_layout.addWidget(allele_label)
        allele_layout.addWidget(self.allele_inputbox)
        allele_layout.addWidget(self.allele_button)
        mhc_group_layout.addLayout(allele_layout)

        mhc_groupbox.setLayout(mhc_group_layout)
        main_layout.addWidget(mhc_groupbox)

        ### GENERAL SCORES
        gs_groupbox = QGroupBox('General Predictors')
        gs_group_layout = QVBoxLayout()
        # gs_group_layout.insertSpacing(0, 5)
        gs_group_layout.setSpacing(5)

        # Auto select
        ap_layout = QHBoxLayout()
        self.ap_checkbox = QCheckBox('Auto-predict best combination')
        self.ap_checkbox.toggled.connect(self.on_autopred_checkbox_toggled)
        ap_layout.addWidget(self.ap_checkbox)
        gs_group_layout.addLayout(ap_layout)

        # RT score
        rt_label = QLabel('RT Score: \t')
        rt_models = ['AutoRT', 'Deeplc_hela_hf', 'AlphaPeptDeep_rt_generic', 'Chronologer_RT',
                     'Prosit_2019_irt', 'Prosit_2024_irt_cit', 'Prosit_2020_irt_TMT']
        self.checkboxes_rt = [QCheckBox(model) for model in rt_models]
        rt_layout = grid_layout(rt_label, self.checkboxes_rt)
        gs_group_layout.addLayout(rt_layout)

        # MS2 score
        ms2_label = QLabel('MS2 Score:\t')
        unsuitable_ms2_models = ['UniSpec', 'Prosit_2024_intensity_XL_NMS2', 'Prosit_2023_intensity_XL_CMS2',
                                 'Prosit_2023_intensity_XL_CMS3']
        ms2_models = ['AlphaPeptDeep_ms2_generic', 'ms2pip_HCD2021', 'ms2pip_Immuno_HCD', 'ms2pip_timsTOF2023',
                      'ms2pip_timsTOF2024', 'ms2pip_iTRAQphospho', 'ms2pip_TTOF5600', 'ms2pip_CID_TMT',
                      'Prosit_2019_intensity', 'Prosit_2020_intensity_HCD', 'Prosit_2020_intensity_CID',
                      'Prosit_2023_intensity_timsTOF', 'Prosit_2024_intensity_cit', 'Prosit_2020_intensity_TMT']

        self.checkboxes_ms2 = [QCheckBox(model) for model in ms2_models]
        self.checkboxes_ms2.insert(7, QLabel(''))
        ms2_layout = grid_layout(ms2_label, self.checkboxes_ms2)
        gs_group_layout.addLayout(ms2_layout)

        # CCS score
        ccs_label = QLabel('CCS Score:\t')
        ccs_models = ['IM2Deep', 'AlphaPeptDeep_ccs_generic']
        self.checkboxes_ccs = [QCheckBox(model) for model in ccs_models]
        ccs_layout = grid_layout(ccs_label, self.checkboxes_ccs)
        gs_group_layout.addLayout(ccs_layout)

        gs_groupbox.setLayout(gs_group_layout)
        main_layout.addWidget(gs_groupbox)


        ### RUN PARAMS
        rp_groupbox = QGroupBox('Run Parameters')
        rp_group_layout = QVBoxLayout()
        # rp_group_layout.insertSpacing(0, 5)
        rp_group_layout.setSpacing(5)

        p1_layout = QHBoxLayout()
        p1_layout.setAlignment(Qt.AlignLeft)

        # Peptide encoding
        self.pe_checkbox = QCheckBox('Peptide Encoding')
        self.pe_checkbox.setChecked(True)
        p1_layout.addWidget(self.pe_checkbox)
        p1_layout.addSpacing(30)

        # Fine tune
        self.ft_checkbox = QCheckBox('Fine tune')
        p1_layout.addWidget(self.ft_checkbox)
        p1_layout.addSpacing(30)

        # Protein Inference
        self.pi_checkbox = QCheckBox('Protein Inference')
        self.pi_checkbox.setChecked(True)
        p1_layout.addWidget(self.pi_checkbox)
        p1_layout.addSpacing(30)

        # Remove Decoy
        self.rd_checkbox = QCheckBox('Remove Decoy')
        p1_layout.addWidget(self.rd_checkbox)
        p1_layout.addSpacing(30)

        # Remove Contamination
        self.rc_checkbox = QCheckBox('Remove Contaminant')
        p1_layout.addWidget(self.rc_checkbox)
        p1_layout.addSpacing(30)

        # Peptide length
        self.pl_checkbox = QCheckBox('Filter by length')
        self.pl_min_spinbox = QSpinBox()
        self.pl_min_spinbox.setRange(8, 30)
        self.pl_min_spinbox.setValue(8)
        self.pl_max_spinbox = QSpinBox()
        self.pl_max_spinbox.setRange(8, 30)
        self.pl_max_spinbox.setValue(15)
        p1_layout.addWidget(self.pl_checkbox)
        p1_layout.addWidget(self.pl_min_spinbox)
        p1_layout.addWidget(QLabel('-'))
        p1_layout.addWidget(self.pl_max_spinbox)
        p1_layout.addSpacing(30)
        rp_group_layout.addLayout(p1_layout)

        p2_layout = QHBoxLayout()
        p2_layout.setAlignment(Qt.AlignLeft)

        # FDR filtering
        psm_fdr_label = QLabel('PSM FDR:')
        self.psm_fdr_spinbox = QDoubleSpinBox()
        self.psm_fdr_spinbox.setRange(0.000001, 1)
        self.psm_fdr_spinbox.setValue(0.01)
        self.psm_fdr_spinbox.setSingleStep(0.01)
        pep_fdr_label = QLabel('Peptide FDR:')
        self.pep_fdr_spinbox = QDoubleSpinBox()
        self.pep_fdr_spinbox.setRange(0.000001, 1)
        self.pep_fdr_spinbox.setValue(0.01)
        self.pep_fdr_spinbox.setSingleStep(0.01)
        seq_fdr_label = QLabel('Sequence FDR:')
        self.seq_fdr_spinbox = QDoubleSpinBox()
        self.seq_fdr_spinbox.setRange(0.000001, 1)
        self.seq_fdr_spinbox.setValue(0.01)
        self.seq_fdr_spinbox.setSingleStep(0.01)
        self.cfdr_checkbox = QCheckBox('Control combine FDR')
        self.cfdr_checkbox.setChecked(True)

        fdr_layout = QHBoxLayout()
        fdr_layout.setSpacing(15)
        psm_fdr_layout = QHBoxLayout()
        psm_fdr_layout.setSpacing(2)
        psm_fdr_layout.addWidget(psm_fdr_label)
        psm_fdr_layout.addWidget(self.psm_fdr_spinbox)
        pep_fdr_layout = QHBoxLayout()
        pep_fdr_layout.setSpacing(2)
        pep_fdr_layout.addWidget(pep_fdr_label)
        pep_fdr_layout.addWidget(self.pep_fdr_spinbox)
        seq_fdr_layout = QHBoxLayout()
        seq_fdr_layout.setSpacing(2)
        seq_fdr_layout.addWidget(seq_fdr_label)
        seq_fdr_layout.addWidget(self.seq_fdr_spinbox)
        fdr_layout.addLayout(psm_fdr_layout)
        fdr_layout.addLayout(pep_fdr_layout)
        fdr_layout.addLayout(seq_fdr_layout)
        fdr_layout.addWidget(self.cfdr_checkbox)
        fdr_layout.addSpacing(10)
        p2_layout.addLayout(fdr_layout)

        # Koina
        koina_label = QLabel('Koina server URL: ')
        self.koina_inputbox = QLineEdit('koina.wilhelmlab.org:443')
        self.koina_inputbox.setFixedWidth(160)
        p2_layout.addWidget(koina_label)
        p2_layout.addWidget(self.koina_inputbox)
        p2_layout.addSpacing(10)

        # Max thread
        self.thread_label = QLabel('Threads: ')
        self.thread_spinbox = QSpinBox()
        self.thread_spinbox.setRange(1, os.cpu_count() - 1)
        self.thread_spinbox.setValue(os.cpu_count() - 1)
        p2_layout.addWidget(self.thread_label)
        p2_layout.addWidget(self.thread_spinbox)
        p2_layout.addSpacing(30)
        rp_group_layout.addLayout(p2_layout)

        rp_groupbox.setLayout(rp_group_layout)
        main_layout.addWidget(rp_groupbox)

        ### Logger
        run_layout = QVBoxLayout()
        run_layout.setSpacing(10)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(120)
        run_layout.addWidget(self.log_output)

        ### Execution
        self.button_run = QPushButton("RUN")
        self.button_run.clicked.connect(self.on_exec_clicked)
        run_layout.addWidget(self.button_run)
        main_layout.addLayout(run_layout)

        self.worker_thread = MhcBoosterWorker(commands=None)
        self.worker_thread.message.connect(self.add_log)
        self.worker_thread.finished.connect(self.worker_stop)

        main_tab = QWidget()
        main_tab.setLayout(main_layout)
        self.tab_widget.addTab(main_tab, 'Run')


    def add_config_tab(self):
        config_layout = QVBoxLayout()
        config_layout.setContentsMargins(10, 20, 10, 0) # left, top, right, bottom
        config_layout.setSpacing(20)
        config_layout.setAlignment(Qt.AlignTop)

        third_party_groupbox = QGroupBox('Third-party tools')
        third_party_layout = QVBoxLayout()
        third_party_layout.setSpacing(5)

        # Introduction
        introduction_label = QLabel('MHCBooster utilizes a variety of tools for RT, MS2, and CCS scoring.'
                                    ' Some of these tools are governed by strict licenses and must be manually'
                                    ' downloaded and installed. Please input the paths to the downloaded'
                                    ' zip files. And they will be automatically installed by pressing'
                                    ' the \'Install to MHCBooster\' button.')
        introduction_label.setWordWrap(True)
        introduction_label_layout = QHBoxLayout()
        introduction_label_layout.setContentsMargins(0, 5, 0, 10)
        introduction_label_layout.addWidget(introduction_label)
        third_party_layout.addLayout(introduction_label_layout)

        # MSFragger
        msfragger_label = QLabel('MSFragger path: \t')
        self.msfragger_inputbox = QLineEdit()
        self.msfragger_inputbox.setPlaceholderText("Select the path to MSFragger-4.1.zip ...")
        self.msfragger_browse_button = QPushButton("Browse")
        self.msfragger_browse_button.clicked.connect(self.open_file_dialog)
        self.msfragger_download_button = QPushButton("Download")
        self.msfragger_download_button.clicked.connect(lambda: webbrowser.open('https://msfragger-upgrader.nesvilab.org/upgrader/'))
        msfragger_layout = QHBoxLayout()
        msfragger_layout.addWidget(msfragger_label)
        msfragger_layout.addWidget(self.msfragger_inputbox)
        msfragger_layout.addWidget(self.msfragger_browse_button)
        msfragger_layout.addWidget(self.msfragger_download_button)
        third_party_layout.addLayout(msfragger_layout)

        # AutoRT
        autort_label = QLabel('AutoRT path: \t')
        self.autort_inputbox = QLineEdit()
        self.autort_inputbox.setPlaceholderText("Select the path to AutoRT-master.zip ...")
        self.autort_browse_button = QPushButton("Browse")
        self.autort_browse_button.clicked.connect(self.open_file_dialog)
        self.autort_download_button = QPushButton("Download")
        self.autort_download_button.clicked.connect(lambda: webbrowser.open('https://github.com/bzhanglab/AutoRT/archive/refs/heads/master.zip'))
        autort_layout = QHBoxLayout()
        autort_layout.addWidget(autort_label)
        autort_layout.addWidget(self.autort_inputbox)
        autort_layout.addWidget(self.autort_browse_button)
        autort_layout.addWidget(self.autort_download_button)
        third_party_layout.addLayout(autort_layout)

        # BigMHC
        bigmhc_label = QLabel('BigMHC path: \t')
        self.bigmhc_inputbox = QLineEdit()
        self.bigmhc_inputbox.setPlaceholderText("Select the path to bigmhc-master.zip ...")
        self.bigmhc_browse_button = QPushButton("Browse")
        self.bigmhc_browse_button.clicked.connect(self.open_file_dialog)
        self.bigmhc_download_button = QPushButton("Download")
        self.bigmhc_download_button.clicked.connect(lambda: webbrowser.open('https://github.com/KarchinLab/bigmhc/archive/refs/heads/master.zip'))
        bigmhc_layout = QHBoxLayout()
        bigmhc_layout.addWidget(bigmhc_label)
        bigmhc_layout.addWidget(self.bigmhc_inputbox)
        bigmhc_layout.addWidget(self.bigmhc_browse_button)
        bigmhc_layout.addWidget(self.bigmhc_download_button)
        third_party_layout.addLayout(bigmhc_layout)

        # NetMHCpan
        netmhcpan_label = QLabel('NetMHCpan path:  ')
        self.netmhcpan_inputbox = QLineEdit()
        self.netmhcpan_inputbox.setPlaceholderText("Select the path to netMHCpan-4.1b.Linux.tar.gz ...")
        self.netmhcpan_browse_button = QPushButton("Browse")
        self.netmhcpan_browse_button.clicked.connect(self.open_file_dialog)
        self.netmhcpan_download_button = QPushButton("Download")
        self.netmhcpan_download_button.clicked.connect(lambda: webbrowser.open('https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=4.1&packageversion=4.1b&platform=Linux'))
        netmhcpan_layout = QHBoxLayout()
        netmhcpan_layout.addWidget(netmhcpan_label)
        netmhcpan_layout.addWidget(self.netmhcpan_inputbox)
        netmhcpan_layout.addWidget(self.netmhcpan_browse_button)
        netmhcpan_layout.addWidget(self.netmhcpan_download_button)
        third_party_layout.addLayout(netmhcpan_layout)

        # NetMHCIIpan
        netmhcIIpan_label = QLabel('NetMHCIIpan path:')
        self.netmhcIIpan_inputbox = QLineEdit()
        self.netmhcIIpan_inputbox.setPlaceholderText("Select the path to netMHCIIpan-4.3e.Linux.tar.gz ...")
        self.netmhcIIpan_browse_button = QPushButton("Browse")
        self.netmhcIIpan_browse_button.clicked.connect(self.open_file_dialog)
        self.netmhcIIpan_download_button = QPushButton("Download")
        self.netmhcIIpan_download_button.clicked.connect(lambda: webbrowser.open('https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCIIpan&version=4.3&packageversion=4.3e&platform=Linux'))
        netmhcIIpan_layout = QHBoxLayout()
        netmhcIIpan_layout.addWidget(netmhcIIpan_label)
        netmhcIIpan_layout.addWidget(self.netmhcIIpan_inputbox)
        netmhcIIpan_layout.addWidget(self.netmhcIIpan_browse_button)
        netmhcIIpan_layout.addWidget(self.netmhcIIpan_download_button)
        third_party_layout.addLayout(netmhcIIpan_layout)

        # MixMHC2pred
        mixmhc2pred_label = QLabel('MixMHC2pred path:')
        self.mixmhc2pred_inputbox = QLineEdit()
        self.mixmhc2pred_inputbox.setPlaceholderText("Select the path to MixMHC2pred-2.0.zip ...")
        self.mixmhc2pred_browse_button = QPushButton("Browse")
        self.mixmhc2pred_browse_button.clicked.connect(self.open_file_dialog)
        self.mixmhc2pred_download_button = QPushButton("Download")
        self.mixmhc2pred_download_button.clicked.connect(lambda: webbrowser.open('https://github.com/GfellerLab/MixMHC2pred/releases/download/v2.0.2.2/MixMHC2pred-2.0.zip'))
        mixmhc2pred_layout = QHBoxLayout()
        mixmhc2pred_layout.addWidget(mixmhc2pred_label)
        mixmhc2pred_layout.addWidget(self.mixmhc2pred_inputbox)
        mixmhc2pred_layout.addWidget(self.mixmhc2pred_browse_button)
        mixmhc2pred_layout.addWidget(self.mixmhc2pred_download_button)
        third_party_layout.addLayout(mixmhc2pred_layout)

        # IonQuant
        ionquant_label = QLabel('IonQuant path: \t')
        self.ionquant_inputbox = QLineEdit()
        self.ionquant_inputbox.setPlaceholderText("Select the path to IonQuant-1.11.11.zip ...")
        self.ionquant_browse_button = QPushButton("Browse")
        self.ionquant_browse_button.clicked.connect(self.open_file_dialog)
        self.ionquant_download_button = QPushButton("Download")
        self.ionquant_download_button.clicked.connect(lambda: webbrowser.open('https://msfragger-upgrader.nesvilab.org/ionquant/'))
        ionquant_layout = QHBoxLayout()
        ionquant_layout.addWidget(ionquant_label)
        ionquant_layout.addWidget(self.ionquant_inputbox)
        ionquant_layout.addWidget(self.ionquant_browse_button)
        ionquant_layout.addWidget(self.ionquant_download_button)
        third_party_layout.addLayout(ionquant_layout)

        extract_layout = QHBoxLayout()
        extract_layout.setContentsMargins(0, 10, 0, 0)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 7)  # Set range to 0 to make it indeterminate
        self.progress_bar.setTextVisible(False)  # Hide the text inside the progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.extract_button = QPushButton("Install to MHCBooster")
        self.extract_button.clicked.connect(self.on_install_clicked)
        self.refresh_third_party_status()
        extract_layout.addWidget(self.progress_bar)
        extract_layout.addWidget(self.extract_button)
        extract_layout.setAlignment(Qt.AlignRight)
        third_party_layout.addLayout(extract_layout)

        third_party_groupbox.setLayout(third_party_layout)
        config_layout.addWidget(third_party_groupbox)

        license_groupbox = QGroupBox('License')
        license_layout = QVBoxLayout()
        license_text = QLabel('MHCBooster is an open-source software tool released under the GNU General Public'
                              ' License (GPL) version 3. This means that you are free to use, modify, and distribute'
                              ' the software, as long as you adhere to the terms and conditions set forth by the GPL-3'
                              ' license. Additionally, when using MHCBooster, please ensure that you comply with the'
                              ' licenses of any third-party tools or libraries integrated with the software. These'
                              ' third-party components may be subject to different licensing agreements, and it is'
                              ' your responsibility to review and follow the relevant terms for each of them. By using'
                              ' MHCBooster, you agree to abide by the obligations of both the GPL-3 and any applicable'
                              ' third-party licenses.')
        license_text.setWordWrap(True)
        license_layout.addWidget(license_text)
        license_groupbox.setLayout(license_layout)
        config_layout.addWidget(license_groupbox)

        cite_groupbox = QGroupBox('How to cite')
        cite_layout = QVBoxLayout()
        cite = ('MHCBooster: in developing; '
                '            <a href="https://doi.org/10.1038/s41467-024-54734-9">https://doi.org/10.1038/s41467-024-54734-9</a><br>'
                '<br>'
                '<br>'
                'MSFragger: <a href="https://doi.org/10.1038/nmeth.4256">https://doi.org/10.1038/nmeth.4256</a>; '
                '       <a href="https://doi.org/10.1038/s41467-020-17921-y">https://doi.org/10.1038/s41467-020-17921-y</a><br>'
                'NetMHCpan: <a href="https://doi.org/10.1093/nar/gkaa379">https://doi.org/10.1093/nar/gkaa379</a><br>'
                'MHCflurry: <a href="https://doi.org/10.1016/j.cels.2020.06.010">https://doi.org/10.1016/j.cels.2020.06.010</a><br>'
                'BigMHC: <a href="https://doi.org/10.1038/s42256-023-00694-6">https://doi.org/10.1038/s42256-023-00694-6</a><br>'
                'NetMHCIIpan: <a href="https://doi.org/10.1126/sciadv.adj6367">https://doi.org/10.1126/sciadv.adj6367</a><br>'
                'MixMHC2pred: <a href="https://doi.org/10.1038/s41587-019-0289-6">https://doi.org/10.1038/s41587-019-0289-6</a>; '
                '       <a href="https://doi.org/10.1016/j.immuni.2023.03.009">https://doi.org/10.1016/j.immuni.2023.03.009</a><br>'
                'AutoRT: <a href="https://doi.org/10.1038/s41467-020-15456-w">https://doi.org/10.1038/s41467-020-15456-w</a><br>'
                'DeepLC: <a href="https://doi.org/10.1038/s41592-021-01301-5">https://doi.org/10.1038/s41592-021-01301-5</a><br>'
                'AlphaPeptDeep: <a href="https://doi.org/10.1038/s41467-022-34904-3">https://doi.org/10.1038/s41467-022-34904-3</a><br>'
                'Prosit: <a href="https://doi.org/10.1038/s41592-019-0426-7">https://doi.org/10.1038/s41592-019-0426-7</a>; '
                '       <a href="https://doi.org/10.1038/s41467-021-23713-9">https://doi.org/10.1038/s41467-021-23713-9</a>; <br>'
                '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                '       <a href="https://doi.org/10.1038/s41467-024-48322-0">https://doi.org/10.1038/s41467-024-48322-0</a>; '
                '       <a href="https://doi.org/10.1021/acs.analchem.1c05435">https://doi.org/10.1021/acs.analchem.1c05435</a><br>'
                'Chronologer: <a href="https://doi.org/10.1101/2023.05.30.542978">https://doi.org/10.1101/2023.05.30.542978</a><br>'
                'MS2PIP: <a href="https://doi.org/10.1093/nar/gkad335">https://doi.org/10.1093/nar/gkad335</a><br>'
                'IM2Deep: <a href="https://doi.org/10.1021/acs.jproteome.4c00609">https://doi.org/10.1021/acs.jproteome.4c00609</a><br>'
                'Koina: <a href="https://doi.org/10.1101/2024.06.01.596953">https://doi.org/10.1101/2024.06.01.596953</a><br>'
                'ProteinProphet: <a href="https://doi.org/10.1021/ac0341261">https://doi.org/10.1021/ac0341261</a><br>'
                'Philosopher: <a href="https://doi.org/10.1038/s41592-020-0912-y">https://doi.org/10.1038/s41592-020-0912-y</a><br>'
                )
        cite_text = QTextBrowser()
        cite_text.setOpenExternalLinks(True)
        cite_text.setHtml(cite)
        cite_text.setReadOnly(True)
        cite_layout.addWidget(cite_text)
        cite_groupbox.setLayout(cite_layout)
        config_layout.addWidget(cite_groupbox)

        config_tab = QWidget()
        config_tab.setLayout(config_layout)
        self.tab_widget.addTab(config_tab, 'Configuration')


    def open_folder_dialog(self):
        sender = self.sender()
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.Directory)

        if file_dialog.exec():
            selected_path = file_dialog.selectedFiles()[0]
            if sender == self.psm_button:
                self.psm_inputbox.setText(selected_path)
            elif sender == self.mzml_button:
                self.psm_mzml_inputbox.setText(selected_path)
            elif sender == self.psm_output_button:
                self.psm_output_inputbox.setText(selected_path)
                self.raw_output_inputbox.setText(selected_path)
            elif sender == self.raw_button:
                self.raw_inputbox.setText(selected_path)
            elif sender == self.raw_output_button:
                self.raw_output_inputbox.setText(selected_path)
                self.psm_output_inputbox.setText(selected_path)

    def open_file_dialog(self):
        sender = self.sender()
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec():
            selected_path = file_dialog.selectedFiles()[0]
            if sender == self.allele_button:
                self.allele_inputbox.setText(selected_path)
            elif sender == self.psm_fasta_button:
                self.psm_fasta_inputbox.setText(selected_path)
                self.raw_fasta_inputbox.setText(selected_path)
            elif sender == self.raw_fasta_button:
                self.raw_fasta_inputbox.setText(selected_path)
                self.psm_fasta_inputbox.setText(selected_path)
            elif sender == self.param_button:
                self.raw_param_inputbox.setText(selected_path)
            elif sender == self.msfragger_browse_button:
                self.msfragger_inputbox.setText(selected_path)
            elif sender == self.ionquant_browse_button:
                self.ionquant_inputbox.setText(selected_path)
            elif sender == self.autort_browse_button:
                self.autort_inputbox.setText(selected_path)
            elif sender == self.bigmhc_browse_button:
                self.bigmhc_inputbox.setText(selected_path)
            elif sender == self.netmhcpan_browse_button:
                self.netmhcpan_inputbox.setText(selected_path)
            elif sender == self.netmhcIIpan_browse_button:
                self.netmhcIIpan_inputbox.setText(selected_path)
            elif sender == self.mixmhc2pred_browse_button:
                self.mixmhc2pred_inputbox.setText(selected_path)


    def on_mhc_I_checkbox_toggled(self, checked):
        if checked:
            self.mhc_class = 'I'
            self.pl_min_spinbox.setRange(8, 15)
            self.pl_max_spinbox.setRange(8, 15)
            self.pl_min_spinbox.setValue(min(max(8, self.pl_min_spinbox.value()), 15))
            self.pl_max_spinbox.setValue(min(max(8, self.pl_max_spinbox.value()), 15))
            for checkbox in self.checkboxes_mhc_II:
                checkbox.setChecked(False)

    def on_mhc_II_checkbox_toggled(self, checked):
        if checked:
            self.mhc_class = 'II'
            self.pl_min_spinbox.setRange(9, 30)
            self.pl_max_spinbox.setRange(9, 30)
            self.pl_min_spinbox.setValue(min(max(9, self.pl_min_spinbox.value()), 30))
            self.pl_max_spinbox.setValue(min(max(9, self.pl_max_spinbox.value()), 30))
            for checkbox in self.checkboxes_mhc_I:
                checkbox.setChecked(False)

    def on_autopred_checkbox_toggled(self, checked):
        if checked:
            for checkbox in self.checkboxes_rt + self.checkboxes_ms2 + self.checkboxes_ccs:
                checkbox.setDisabled(True)
        else:
            for checkbox in self.checkboxes_rt + self.checkboxes_ms2 + self.checkboxes_ccs:
                checkbox.setDisabled(False)

    def show_message(self, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Information")
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.exec()

    def load_default_config(self):
        pass

    def save_config(self):
        pass

    def load_config(self):
        pass

    def save_params(self):
        date = datetime.now().strftime("%y_%m_%d")
        param_filename = f'mhcbooster-{date}.params'
        with open(Path(self.output_inputbox.text()) / param_filename, 'w') as f:
            f.write('')

    def on_exec_clicked(self):
        if self.button_run.text() == 'RUN':
            self.log_output.setText('')
            self.run()
        else:
            self.add_log('Termination triggered...')
            self.worker_stop()

    def on_install_clicked(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        if Path(self.msfragger_inputbox.text()).exists() and self.msfragger_inputbox.text().endswith('.zip'):
            install_msfragger(self.msfragger_inputbox.text())
        self.progress_bar.setValue(1)
        if Path(self.autort_inputbox.text()).exists() and self.autort_inputbox.text().endswith('.zip'):
            install_autort(self.autort_inputbox.text())
        self.progress_bar.setValue(2)
        if Path(self.bigmhc_inputbox.text()).exists() and self.bigmhc_inputbox.text().endswith('.zip'):
            install_bigmhc(self.bigmhc_inputbox.text())
        self.progress_bar.setValue(3)
        if Path(self.netmhcpan_inputbox.text()).exists() and self.netmhcpan_inputbox.text().endswith('.gz'):
            install_netmhcpan(self.netmhcpan_inputbox.text())
        self.progress_bar.setValue(4)
        if Path(self.netmhcIIpan_inputbox.text()).exists() and self.netmhcIIpan_inputbox.text().endswith('.gz'):
            install_netmhcIIpan(self.netmhcIIpan_inputbox.text())
        self.progress_bar.setValue(5)
        if Path(self.mixmhc2pred_inputbox.text()).exists() and self.mixmhc2pred_inputbox.text().endswith('.zip'):
            install_mixmhc2pred(self.mixmhc2pred_inputbox.text())
        self.progress_bar.setValue(6)
        if Path(self.ionquant_inputbox.text()).exists() and self.ionquant_inputbox.text().endswith('.zip'):
            install_ionquant(self.ionquant_inputbox.text())
        self.progress_bar.setValue(7)
        self.refresh_third_party_status()
        time.sleep(1)
        self.progress_bar.setVisible(False)

    def refresh_third_party_status(self):
        third_party_folder = Path(__file__).parent.parent/'third_party'
        for path in third_party_folder.iterdir():
            if 'AutoRT' in path.name:
                self.autort_inputbox.setText(str(path))
            if 'bigmhc' in path.name:
                self.bigmhc_inputbox.setText(str(path))
            if 'MixMHC2pred' in path.name:
                self.mixmhc2pred_inputbox.setText(str(path))
            if 'MSFragger' in path.name:
                self.msfragger_inputbox.setText(str(path))
            if 'IonQuant' in path.name:
                self.ionquant_inputbox.setText(str(path))
            if 'netMHCIIpan' in path.name:
                self.netmhcIIpan_inputbox.setText(str(path))
            if 'netMHCpan' in path.name:
                self.netmhcpan_inputbox.setText(str(path))

    def worker_start(self):
        self.button_run.setText('STOP')
        self.worker_thread.start()

    def worker_stop(self):
        self.worker_thread.stop()
        self.button_run.setText('RUN')


    def run(self):
        self.add_log(f'Running MhcBooster {__version__}...')

        commands = []
        if self.raw_radiobutton.isChecked():
            param_path = self.raw_param_inputbox.text()
            fasta_path = self.raw_fasta_inputbox.text()
            raw_path = self.raw_inputbox.text()
            n_threads = self.thread_spinbox.value()
            msfragger_commands = get_msfragger_command(param_path=param_path, fasta_path=fasta_path, raw_path=raw_path, num_threads=n_threads)
            commands += msfragger_commands
            self.psm_inputbox.setText(self.raw_inputbox.text())
            self.psm_fasta_inputbox.setText(self.raw_fasta_inputbox.text())
            self.psm_mzml_inputbox.setText(self.raw_inputbox.text())
            self.psm_output_inputbox.setText(self.raw_output_inputbox.text())

        # File
        psm_folder = Path(self.psm_inputbox.text())
        fasta_path = self.psm_fasta_inputbox.text()
        mzml_folder = self.psm_mzml_inputbox.text()
        output_folder = Path(self.psm_output_inputbox.text()).resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
        pin_files = list(psm_folder.rglob('*.pin'))
        if len(pin_files) == 0 and len(commands) == 0:
            self.show_message('No pin files found')
            return

        # Run params
        auto_pred = self.ap_checkbox.isChecked()
        pe = self.pe_checkbox.isChecked()
        fine_tune = self.ft_checkbox.isChecked()
        infer_protein = self.pi_checkbox.isChecked()
        remove_contaminant = self.rc_checkbox.isChecked()
        remove_decoy = self.rd_checkbox.isChecked()
        min_pep_length, max_pep_length = None, None
        if self.pl_checkbox.isChecked():
            min_pep_length = self.pl_min_spinbox.value()
            max_pep_length = self.pl_max_spinbox.value()
        psm_fdr = self.psm_fdr_spinbox.value()
        pep_fdr = self.pep_fdr_spinbox.value()
        seq_fdr = self.seq_fdr_spinbox.value()
        cfdr = self.cfdr_checkbox.isChecked()
        koina_server_url = self.koina_inputbox.text()
        n_threads = self.thread_spinbox.value()

        # App score
        app_predictors = []
        mhc_class = None
        for checkbox in self.checkboxes_mhc_I:
            if checkbox.isChecked():
                app_predictors.append(checkbox.text())
                mhc_class = 'I'
        for checkbox in self.checkboxes_mhc_II:
            if checkbox.isChecked():
                app_predictors.append(checkbox.text())
                mhc_class = 'II'

        allele_param = self.allele_inputbox.text()
        if len(app_predictors) > 0 and len(allele_param) == 0:
            self.show_message('Input alleles cannot be empty')
            return

        # RT score
        rt_predictors = []
        for checkbox in self.checkboxes_rt:
            if checkbox.isChecked():
                rt_predictors.append(checkbox.text())

        # MS2 score
        ms2_predictors = []
        for checkbox in self.checkboxes_ms2:
            if isinstance(checkbox, QLabel):
                continue
            if checkbox.isChecked():
                ms2_predictors.append(checkbox.text())

        # CCS score
        ccs_predictors = []
        for checkbox in self.checkboxes_ccs:
            if checkbox.isChecked():
                ccs_predictors.append(checkbox.text())

        app_predictor_param = ' '.join(app_predictors)
        rt_predictor_param = ' '.join(rt_predictors)
        ms2_predictor_param = ' '.join(ms2_predictors)
        ccs_predictor_param = ' '.join(ccs_predictors)
        cli_path = str(Path(__file__).parent/'mhcbooster_cli.py')
        command = f'python {cli_path} -n {n_threads}'
        if len(app_predictor_param) > 0 and len(allele_param) > 0:
            command += f' --app_predictors {app_predictor_param}'
            command += f' --alleles {allele_param}'
            command += f' --mhc_class {mhc_class}'
        if len(rt_predictor_param) > 0:
            command += f' --rt_predictors {rt_predictor_param}'
        if len(ms2_predictor_param) > 0:
            command += f' --ms2_predictors {ms2_predictor_param}'
        if len(ccs_predictor_param) > 0:
            command += f' --ccs_predictors {ccs_predictor_param}'
        if auto_pred:
            command += f' --auto_pred'
        if pe:
            command += f' --encode_peptide_sequences'
        if fine_tune:
            command += f' --fine_tune'
        if infer_protein:
            command += f' --infer_protein'
        if remove_contaminant:
            command += f' --remove_contaminant'
        if remove_decoy:
            command += f' --remove_decoy'
        if min_pep_length and max_pep_length:
            command += f' --min_pep_len {min_pep_length} --max_pep_len {max_pep_length}'
        if psm_fdr is not None:
            command += f' --psm_fdr {psm_fdr}'
        if pep_fdr is not None:
            command += f' --pep_fdr {pep_fdr}'
        if seq_fdr is not None:
            command += f' --seq_fdr {seq_fdr}'
        if cfdr:
            command += f' --control_combine_fdr'
        if len(koina_server_url) > 0:
            command += f' --koina_server_url {koina_server_url}'
        command += f' --input {psm_folder} --output_dir {output_folder}'
        if len(fasta_path) > 0:
            command += f' --fasta_path {fasta_path}'
        if len(mzml_folder) > 0:
            command += f' --mzml_dir {mzml_folder}'
        print(command)
        commands.append(command)
        self.worker_thread.commands = commands
        self.worker_start()


    def add_log(self, message):
        print(message)
        # if '\r' in message:
        #     print('Bingo!')
        #     self.log_output.moveCursor(QTextCursor.StartOfLine)
        self.log_output.append(message)
        self.log_output.moveCursor(QTextCursor.End)
        self.log_output.ensureCursorVisible()

    def save_settings(self):
        settings = QSettings('MHCBooster')

        settings.setValue('psm_radiobutton', self.psm_radiobutton.isChecked())
        settings.setValue('psm_inputbox', self.psm_inputbox.text())
        settings.setValue('psm_fasta_inputbox', self.psm_fasta_inputbox.text())
        settings.setValue('psm_mzml_inputbox', self.psm_mzml_inputbox.text())
        settings.setValue('psm_output_inputbox', self.psm_output_inputbox.text())

        settings.setValue('raw_radiobutton', self.raw_radiobutton.isChecked())
        settings.setValue('raw_inputbox', self.raw_inputbox.text())
        settings.setValue('raw_fasta_inputbox', self.raw_fasta_inputbox.text())
        settings.setValue('raw_param_inputbox', self.raw_param_inputbox.text())
        settings.setValue('raw_output_inputbox', self.raw_output_inputbox.text())

        mhc_I_params = []
        for checkbox in self.checkboxes_mhc_I:
            if checkbox.isChecked():
                mhc_I_params.append(checkbox.text())
        settings.setValue('checkboxes_mhc_I', mhc_I_params)
        mhc_II_params = []
        for checkbox in self.checkboxes_mhc_II:
            if checkbox.isChecked():
                mhc_II_params.append(checkbox.text())
        settings.setValue('checkboxes_mhc_II', mhc_II_params)
        settings.setValue('allele_inputbox', self.allele_inputbox.text())

        settings.setValue('ap_checkbox', self.ap_checkbox.isChecked())
        rt_params = []
        for checkbox in self.checkboxes_rt:
            if checkbox.isChecked():
                rt_params.append(checkbox.text())
        settings.setValue('checkboxes_rt', rt_params)
        ms2_params = []
        for checkbox in self.checkboxes_ms2:
            if not isinstance(checkbox, QCheckBox):
                continue
            if checkbox.isChecked():
                ms2_params.append(checkbox.text())
        settings.setValue('checkboxes_ms2', ms2_params)
        ccs_params = []
        for checkbox in self.checkboxes_ccs:
            if checkbox.isChecked():
                ccs_params.append(checkbox.text())
        settings.setValue('checkboxes_ccs', ccs_params)

        settings.setValue('pe_checkbox', self.pe_checkbox.isChecked())
        settings.setValue('ft_checkbox', self.ft_checkbox.isChecked())
        settings.setValue('pi_checkbox', self.pi_checkbox.isChecked())
        settings.setValue('rd_checkbox', self.rd_checkbox.isChecked())
        settings.setValue('rc_checkbox', self.rc_checkbox.isChecked())
        settings.setValue('pl_checkbox', self.pl_checkbox.isChecked())
        settings.setValue('pl_min_spinbox', self.pl_min_spinbox.value())
        settings.setValue('pl_max_spinbox', self.pl_max_spinbox.value())
        settings.setValue('psm_fdr_spinbox', self.psm_fdr_spinbox.value())
        settings.setValue('pep_fdr_spinbox', self.pep_fdr_spinbox.value())
        settings.setValue('seq_fdr_spinbox', self.seq_fdr_spinbox.value())
        settings.setValue('cfdr_checkbox', self.cfdr_checkbox.isChecked())
        settings.setValue('koina_inputbox', self.koina_inputbox.text())
        settings.setValue('thread_spinbox', self.thread_spinbox.value())
                
    def restore_settings(self):
        settings = QSettings('MHCBooster')
        # return
        if settings.value('psm_radiobutton') is None:
            return
        self.psm_radiobutton.setChecked(settings.value('psm_radiobutton') == 'true')
        self.psm_inputbox.setText(settings.value('psm_inputbox'))
        self.psm_fasta_inputbox.setText(settings.value('psm_fasta_inputbox'))
        self.psm_mzml_inputbox.setText(settings.value('psm_mzml_inputbox'))
        self.psm_output_inputbox.setText(settings.value('psm_output_inputbox'))

        self.raw_radiobutton.setChecked(settings.value('raw_radiobutton') == 'true')
        self.raw_inputbox.setText(settings.value('raw_inputbox'))
        self.raw_fasta_inputbox.setText(settings.value('raw_fasta_inputbox'))
        self.raw_param_inputbox.setText(settings.value('raw_param_inputbox'))
        self.raw_output_inputbox.setText(settings.value('raw_output_inputbox'))

        if self.psm_radiobutton.isChecked():
            self.input_stacked_widget.setCurrentIndex(0)
        else:
            self.input_stacked_widget.setCurrentIndex(1)

        if settings.value('checkboxes_mhc_I') is not None:
            for checkbox in self.checkboxes_mhc_I:
                if checkbox.text() in settings.value('checkboxes_mhc_I'):
                    checkbox.setChecked(True)
        if settings.value('checkboxes_mhc_II') is not None:
            for checkbox in self.checkboxes_mhc_II:
                if checkbox.text() in settings.value('checkboxes_mhc_II'):
                    checkbox.setChecked(True)
        self.allele_inputbox.setText(settings.value('allele_inputbox'))

        self.ap_checkbox.setChecked(settings.value('ap_checkbox') == 'true')
        self.on_autopred_checkbox_toggled(self.ap_checkbox.isChecked())
        if settings.value('checkboxes_rt') is not None:
            for checkbox in self.checkboxes_rt:
                if checkbox.text() in settings.value('checkboxes_rt'):
                    checkbox.setChecked(True)
        if settings.value('checkboxes_ms2') is not None:
            for checkbox in self.checkboxes_ms2:
                if isinstance(checkbox, QCheckBox) and checkbox.text() in settings.value('checkboxes_ms2'):
                    checkbox.setChecked(True)
        if settings.value('checkboxes_ccs') is not None:
            for checkbox in self.checkboxes_ccs:
                if checkbox.text() in settings.value('checkboxes_ccs'):
                    checkbox.setChecked(True)

        self.pe_checkbox.setChecked(settings.value('pe_checkbox') == 'true')
        self.ft_checkbox.setChecked(settings.value('ft_checkbox') == 'true')
        self.pi_checkbox.setChecked(settings.value('pi_checkbox') == 'true')
        self.rd_checkbox.setChecked(settings.value('rd_checkbox') == 'true')
        self.rc_checkbox.setChecked(settings.value('rc_checkbox') == 'true')
        self.pl_checkbox.setChecked(settings.value('pl_checkbox') == 'true')
        self.pl_min_spinbox.setValue(int(settings.value('pl_min_spinbox')))
        self.pl_max_spinbox.setValue(int(settings.value('pl_max_spinbox')))
        self.psm_fdr_spinbox.setValue(float(settings.value('psm_fdr_spinbox')))
        self.pep_fdr_spinbox.setValue(float(settings.value('pep_fdr_spinbox')))
        self.seq_fdr_spinbox.setValue(float(settings.value('seq_fdr_spinbox')))
        self.cfdr_checkbox.setChecked(settings.value('cfdr_checkbox') == 'true')
        self.koina_inputbox.setText(settings.value('koina_inputbox'))
        self.thread_spinbox.setValue(int(settings.value('thread_spinbox')))


class MhcBoosterWorker(QThread):
    message = Signal(str)
    finished = Signal()
    def __init__(self, commands):
        super().__init__()
        self.commands = commands
        self.process = None
        self._stop_flag = False

    def run(self):
        self._stop_flag = False
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        for command in self.commands:
            if self._stop_flag:
                break
            self.process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, env=env)
            while True:
                if self._stop_flag:
                    parent_process = psutil.Process(self.process.pid)
                    child_process = parent_process.children(recursive=True)
                    self.message.emit(f'Terminating {len(child_process)} child processes...')
                    for child in child_process:
                        child.terminate()
                    self.message.emit(f'Terminating MHCBooster main process...')
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1)
                        self.message.emit("Terminated gracefully.")
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.message.emit("Process didn't terminate in time, forcibly killed.")
                    break

                rlist, _, _ = select.select([self.process.stdout], [], [], 0.1)
                if rlist:
                    line = self.process.stdout.readline()
                    if line:
                        self.message.emit(line.strip())
                    else:
                        # EOF reached; check process return code.
                        ret_code = self.process.poll()
                        if ret_code is not None:
                            self.message.emit(f"Process finished with return code: {ret_code}")
                            break
                else:
                    # No data available; check if the process has finished.
                    ret_code = self.process.poll()
                    if ret_code is not None:
                        self.message.emit(f"Process finished with return code: {ret_code}")
                        break
                time.sleep(0.01)  # Add a small delay to avoid high CPU usage
        self.finished.emit()

    def stop(self):
        self._stop_flag = True
        self.quit()  # Quit the QThread event loop
        self.wait()  # Wait for the thread to finish

def run():
    app = QApplication(sys.argv)
    font = QFont('Arial')
    font.setPointSizeF(8.8)
    app.setFont(font)
    gui = MhcBoosterGUI()
    gui.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    run()