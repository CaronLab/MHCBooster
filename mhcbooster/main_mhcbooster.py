import os
import re
import time

from mhcbooster.report.combined_reporter import CombinedReporter
from mhcbooster.report.run_reporter import RunReporter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from typing import Union, List
from os import PathLike
from pathlib import Path
from mhcbooster.utils.data_loaders import load_file
from mhcbooster.utils.mzml_parser import get_rt_ccs_ms2_from_mzml, get_rt_ccs_ms2_from_msfragger_mzml, get_rt_ccs_ms2_from_timsconvert_mzml
from mhcbooster.utils.features import prepare_features
from mhcbooster.utils.log import append_log
from mhcbooster.predictors.netmhcpan_helper import NetMHCpanHelper
from mhcbooster.predictors.mhcflurry_helper import MhcFlurryHelper
from mhcbooster.predictors.bigmhc_helper import BigMhcHelper
from mhcbooster.predictors.mixmhc2pred_helper import MixMhc2PredHelper
from mhcbooster.predictors.peptdeep_helper import PeptDeepHelper
from mhcbooster.predictors.autort_helper import AutortHelper
from mhcbooster.predictors.deeplc_helper import DeepLCHelper
from mhcbooster.predictors.im2deep_helper import IM2DeepHelper
from mhcbooster.predictors.koina_helper import KoinaHelper, general_koina_predictors
from mhcbooster.predictors.auto_model_predictor import predict_best_combination
from mhcbooster.utils.fdr import calculate_qs, calculate_peptide_level_qs
from mhcflurry.encodable_sequences import EncodableSequences
from mhcbooster.model.models import get_model_without_peptide_encoding, get_model_with_peptide_encoding, focal_loss
from mhcbooster.utils.peptide import remove_previous_and_next_aa, get_previous_and_next_aa, remove_charge, \
    remove_modifications
from mhcbooster.model.nd_standard_scalar import NDStandardScaler
from copy import deepcopy
from datetime import datetime
import tempfile
from collections import Counter
import tensorflow.python.util.deprecation as deprecation
from inspect import signature
from mhcbooster.utils.dataset import k_fold_split

deprecation._PRINT_DEPRECATION_WARNINGS = False

# This can be uncommented to prevent the GPU from getting used.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DEFAULT_TEMP_MODEL_DIR = str(Path(tempfile.gettempdir()) / 'validator_models')


class MHCBooster:
    def __init__(self,
                 random_seed: int = 0,
                 model_dir: Union[str, PathLike] = DEFAULT_TEMP_MODEL_DIR,
                 max_threads: int = -1):
        self.filename: Union[str, None] = None
        self.filepath: Union[Path, None] = None
        self.decoy_tag: Union[str, None]  = None
        self.model: keras.Model = None
        self.raw_data: Union[pd.DataFrame, None] = None
        self.feature_matrix: Union[pd.DataFrame, None] = None
        self.labels: Union[List[int], None] = None
        self.peptides: Union[List[str], None] = None
        self.peptides_with_mods: Union[List[str], None] = None
        self.prev_aas: Union[np.ndarray, None] = None
        self.next_aas: Union[np.ndarray, None] = None
        self.charges: Union[List[int], None] = None
        self.high_prob_indices: Union[np.ndarray, None] = None
        self.exp_rts: Union[np.ndarray, None] = None
        self.exp_ims: Union[np.ndarray, None] = None
        self.spec_names: Union[np.ndarray, None] = None
        self.spec_indices: Union[np.ndarray, None] = None
        self.exp_ms2s: Union[pd.DataFrame, None] = None
        self.encoded_peptides = None
        self.loaded_filetype: Union[str, None] = None
        self.random_seed: int = random_seed
        self.predictions: np.array = None
        self.qs: np.array = None
        self.percolator_qs = None
        self.obs_rts: np.array = None
        self.pred_rts: np.array = None
        self.mhc_class: Union[str, None] = None
        self.alleles: List[str] = None
        self.min_len: int = 5
        self.max_len: int = 100
        self.model_dir = Path(model_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.fine_tune: bool = False
        self.koina_predictors: List[str] = []
        self.mhcflurry_predictions: pd.DataFrame = None
        self.netmhcpan_predictions: pd.DataFrame = None
        self.peptdeep_predictions: pd.DataFrame = None
        self.annotated_data: pd.DataFrame = None
        if max_threads < 1:
            self.max_threads: int = os.cpu_count()
        else:
            self.max_threads: int = max_threads

    def set_mhc_params(self,
                       alleles: Union[str, List[str]] = None,
                       mhc_class: str = 'I',
                       max_pep_len: int = None,
                       min_pep_len: int = None) -> None:
        """
        Set the MHC-specific parameters.

        :param alleles: The alleles to be used by MhcFlurry or NetMHCpan.
        :param mhc_class: The MHC class of the peptides. Must be one of {'I', 'II'}
        :param min_pep_len: Maximum length of peptides allowed. Will default to 16 for class I and 30 for class II. Note
        that MhcFlurry does not accept peptide lengths greater than 16. There is no length restriction for NetMHCpan.
        :param max_pep_len: Minimum length of peptides allowed. Will default to 8 for class I and 9 for class II. Note
        that NetMHC(II)pan does not accept peptide lengths less than 8 for class I or 9 for class I. NetMHCpan predictions
        take much longer for longer peptides.
        :return: None
        """

        if isinstance(alleles, str):
            alleles = [alleles]
        self.alleles = list(set(alleles))

        if min_pep_len is not None:
            self.min_len = min_pep_len
        if max_pep_len is not None:
            self.max_len = max_pep_len

        if mhc_class is not None:
            if mhc_class not in ['I', 'II']:
                raise ValueError("mhc_class must be one of {'I', 'II'}")
            self.mhc_class = mhc_class

        if mhc_class == 'I' and self.min_len < 8:
            self.min_len = 8
            print('The minimum peptide length is reset to 8, due to limitations in MHC-I predictors.')
        if mhc_class == 'I' and self.max_len > 15:
            self.max_len = 15
            print('The maximum peptide length is reset to 15, due to limitations in MHC-I predictors.')
        if mhc_class == 'II' and self.min_len < 9:
            self.min_len = 9
            print('The minimum peptide length is reset to 9, due to limitations in MHC-II predictors.')
        if mhc_class == 'II' and self.max_len > 30:
            self.max_len = 30
            print('The maximum peptide length is reset to 30, due to limitations in MHC-II predictors.')


    def _check_peptide_lengths(self):
        max_len = self.max_len
        longest_peptide = np.max(np.vectorize(len)(self.peptides))
        if max_len > longest_peptide:
            print(f'Longest peptide ({longest_peptide} mer) is shorter than set maximum length ({max_len} mer). '
                  f'Changing max_len to {longest_peptide}.')
            self.max_len = longest_peptide

    def load_data(self,
                  filepath: Union[str, PathLike],
                  filetype='auto',
                  decoy_tag='rev_',
                  peptide_column: str = None,
                  protein_column: str = None,
                  tag_is_prefix: bool = True,
                  file_delimiter: str = '\t',
                  use_features: Union[List[str], None] = None):
        """
        Generic tabular files must contain a column titled 'Peptide' or 'peptide' which contains the peptide sequences.

        :param filepath: The path to the file you want to load. Can be absolute or relative.
        :param filetype: The type of file. Must be one of {'auto', 'pin', 'pepxml', 'tabular', 'mzid', 'tandem',
            'spectromine'}. If you choose 'auto', the file type will be inferred from the file extension. Be
            cautious when loading pepXML and X! Tandem files, as the extensions are similar. It is best to be explicit
            in these cases.
        :param decoy_tag: The decoy tag used in the upstream FASTA to differentiate decoy proteins from target proteins.
        :param protein_column: The header of the column containing the protein IDs. Required for tabular data of an
            unspecified format.
        :param tag_is_prefix: Whether or not the decoy tag is a prefix. If False, it is assumed the tag is a suffix.
        :param file_delimiter: The delimiter used if the file is tabular.
        :param use_features: A list of column headers to be used as training features. Not required  If your tabular data
            contains a column indicating the target/decoy label of each PSM, DO NOT INCLUDE THIS COLUMN! The label will
            be determined from the protein IDs.
        :return: None
        """

        if filetype == 'auto':
            if str(filepath).lower().endswith('pin') or str(filepath).lower().endswith('mhcv'):
                filetype = 'pin'
            elif str(filepath).lower().endswith('pepxml'):
                filetype = 'tandem'
            elif str(filepath).lower().endswith('pep.xml'):
                filetype = 'pepxml'
            else:
                raise ValueError('File type could not be inferred from filename. You must explicitly specify the '
                                 'filetype.')
        else:
            if filetype not in ['auto', 'pin', 'pepxml', 'tabular', 'mhcv']:
                raise ValueError("filetype must be one of "
                                 "{'auto', 'pin', 'pepxml', 'tabular', 'mhcv'}")

        print(f'MHC class: {self.mhc_class if self.mhc_class else "not specified"}')
        print(f'Alleles: {self.alleles if self.alleles else "not specified"}')
        print(f'Minimum peptide length: {self.min_len}')
        print(f'Maximum peptide length: {self.max_len}')

        print('Loading PSM file...')
        self.raw_data = load_file(filename=filepath, filetype=filetype, decoy_tag=decoy_tag,
                                  protein_column=protein_column, file_sep=file_delimiter,
                                  tag_is_prefix=tag_is_prefix, min_len=self.min_len, max_len=self.max_len)
        self.labels = self.raw_data['Label'].to_numpy()
        self.decoy_tag = decoy_tag

        # Peptide
        if peptide_column is not None:
            self.peptides_with_mods = list(self.raw_data[peptide_column])
        elif filetype == 'pin':
            self.peptides_with_mods = list(self.raw_data['Peptide'])
            peptide_column = 'Peptide'
        #elif filetype == 'mzid':
        #    self.peptides = list(self.raw_data['PeptideSequence'])
        #elif filetype == 'spectromine':
        #    self.peptides = list(self.raw_data['PEP.StrippedSequence'])
        else:
            if 'peptide' in self.raw_data.columns:
                self.peptides_with_mods = list(self.raw_data['peptide'])
                peptide_column = 'peptide'
            elif 'Peptide' in self.raw_data.columns:
                self.peptides_with_mods = list(self.raw_data['Peptide'])
                peptide_column = 'Peptide'
            else:
                raise IndexError('Peptide field could not be automatically found. Please indicate the column '
                                 'containing the peptide sequences')
        self.prev_aas, self.next_aas = get_previous_and_next_aa(self.peptides_with_mods)
        self.peptides_with_mods = np.array(remove_charge(remove_previous_and_next_aa(self.peptides_with_mods)))
        self.peptides = np.array(remove_modifications(self.peptides_with_mods))
        self._check_peptide_lengths()

        # Charge
        self.charges = np.ones(len(self.peptides_with_mods), dtype=int)
        for col in self.raw_data.columns:
            if 'z=other' in col:
                continue
            if 'charge' == col.lower():
                self.charges = self.raw_data[col].to_numpy(dtype=int)
                continue
            if 'charge' in col.lower() or 'z=' in col:
                self.charges[self.raw_data[col] == '1'] =  int(re.findall(r'\d+', col)[0])

        # High prob indices
        self.high_prob_indices = None
        qs_threshold = 0.0001
        max_qs_threshold = 0.01
        min_points = 100

        score_candidates = [('lnExpect', False), ('log10_evalue', False), ('ln(hyperscore)', True),
                            ('hyperscore', True), ('Final_Score', False), ('qvalue', False)]
        score_column, higher_better = None, None
        for col, flag in score_candidates:
            if col in self.raw_data.columns:
                score_column = col
                higher_better = flag
                break

        if score_column is not None:
            qs = calculate_qs(self.raw_data[score_column].astype(float), self.labels, higher_better=higher_better)
        else:
            qs = None
            print('lnExpect or log10_evalue score cannot be found from input files. Processing without calibration!')

        mask = self.raw_data[score_column] == self.raw_data.groupby(peptide_column)[score_column].transform('min')
        if qs is not None and len(qs) > min_points:
            high_prob_indices = (qs <= qs_threshold) & mask
            if np.sum(high_prob_indices) >= min_points:
                self.high_prob_indices = high_prob_indices
            else:
                sorted_indices = np.argsort(qs)
                true_indices = np.flatnonzero(mask.iloc[sorted_indices])
                if len(true_indices) < min_points:
                    print('Not enough PSMs for calibration. Processing without calibration!')
                else:
                    qs_threshold = qs[sorted_indices][true_indices[min_points - 1]]
                    if qs_threshold < max_qs_threshold:
                        self.high_prob_indices = (qs <= qs_threshold) & mask
                        print(f'Not enough PSMs for calibration. Relaxed the high confidence q-value threshold to {qs_threshold}')
                    else:
                        print('Not enough PSMs for calibration. Processing without calibration!')
        else:
            print('Not enough PSMs for calibration. Processing without calibration!')

        print(f'Loaded {len(self.peptides)} PSMs, including {np.sum(self.high_prob_indices)} high confidence PSMs.')

        self.loaded_filetype = filetype
        self.filename = Path(filepath).name
        self.filepath = Path(filepath).expanduser().resolve()

        print('Preparing training features')
        self.feature_matrix = prepare_features(self.raw_data, filetype=self.loaded_filetype, use_features=use_features)


    def encode_peptide_sequences(self):
        """
        Use a BLOSUM62 substitution matrix to numerically encode each peptide sequence. Uses the EncodableSequences
        class from MhcFlurry. Encoded peptides are saved in self.encoded_peptides.
        :return:
        """

        encoder = EncodableSequences(list(self.peptides))
        padding = 'pad_middle' if self.mhc_class == 'I' else 'left_pad_right_pad'
        encoded_peps = encoder.variable_length_to_fixed_length_vector_encoding('BLOSUM62',
                                                                               max_length=self.max_len,
                                                                               alignment_method=padding)
        self.encoded_peptides = deepcopy(encoded_peps)

    def load_psm_coordinates(self, predictor_types=['RT','MS2','CCS']):
        if predictor_types is None or len(predictor_types) == 0:
            return
        if ('RT' not in predictor_types or self.exp_rts is not None) and \
            ('MS2' not in predictor_types or self.exp_ms2s is not None) and \
            ('CCS' not in predictor_types or self.exp_ims is not None):
            return
        if 'retentiontime' in self.raw_data.columns:
            self.exp_rts = self.raw_data['retentiontime'].astype(float)
        if 'RT' in self.raw_data.columns:
            self.exp_rts = self.raw_data['RT'].astype(float)
        if 'IM' in self.raw_data.columns:
            self.exp_ims = self.raw_data['IM'].astype(float)
        if ('RT' in predictor_types and self.exp_rts is None) or \
                ('MS2' in predictor_types and self.exp_ms2s is None) or \
                ('CCS' in predictor_types and self.exp_ims is None):
            assert self.mzml_path is not None, f'mzML folder must be provided for {predictor_types} scores'

    def add_mhcflurry_predictions(self):
        """
        Run MhcFlurry and add presentation predictions to the training feature matrix.

        :return: None
        """
        if self.mhc_class == 'II':
            raise RuntimeError('MhcFlurry is only compatible with MHC class I')

        mhcflurry_helper = MhcFlurryHelper(self.peptides, self.alleles, self.report_directory)
        mhcflurry_helper.predict_df()
        mhcflurry_helper.pred_df.to_csv(Path(self.report_directory) / f'app_prediction.mhcflurry.tsv', sep='\t')

        predictions = mhcflurry_helper.score_df()
        self.feature_matrix = self.feature_matrix.join(predictions)
        mhcflurry_helper.draw_prediction_distributions(predictions, self.labels)


    def add_netmhcpan_predictions(self):
        """
        Run NetMHCpan and add presentation predictions to the training feature matrix.

        :return: None
        """
        if self.alleles is None or self.mhc_class is None:
            raise RuntimeError('You must first set the MHC parameters using set_mhc_params')

        print(f'Running NetMHC{"II" if self.mhc_class == "II" else ""}pan')
        netmhcpan_helper = NetMHCpanHelper(peptides=self.peptides, alleles=self.alleles,
                                           mhc_class=self.mhc_class, n_threads=self.max_threads,
                                           report_directory=self.report_directory)
        netmhcpan_helper.predict_df()
        netmhcpan_helper.pred_df.to_csv(Path(self.report_directory) /
                                        f"app_prediction.netmhc{'ii' if self.mhc_class == 'II' else ''}pan.tsv", sep='\t')

        predictions = netmhcpan_helper.score_df()
        self.feature_matrix = self.feature_matrix.join(predictions)
        netmhcpan_helper.draw_prediction_distributions(predictions, self.labels)


    def add_bigmhc_predictions(self):
        if self.alleles is None or self.mhc_class is None:
            raise RuntimeError('You must first set the MHC parameters using set_mhc_params')

        bigmhc_helper = BigMhcHelper(peptides=self.peptides, alleles=self.alleles, report_directory=self.report_directory)
        bigmhc_helper.predict_df()
        bigmhc_helper.pred_df.to_csv(Path(self.report_directory) / 'app_prediction.bigmhc.tsv', sep='\t')
        predictions = bigmhc_helper.score_df()
        self.feature_matrix = self.feature_matrix.join(predictions)
        bigmhc_helper.draw_prediction_distributions(predictions, self.labels)


    def add_mixmhc2pred_predictions(self):
        if self.alleles is None:
            raise RuntimeError('You must first set the MHC parameters using set_mhc_params')

        mixmhc2pred_helper = MixMhc2PredHelper(peptides=self.peptides, alleles=self.alleles, report_directory=self.report_directory)
        mixmhc2pred_helper.predict_df()
        mixmhc2pred_helper.pred_df.to_csv(Path(self.report_directory) / f'app_prediction.mixmhc2pred.tsv', sep='\t')
        predictions = mixmhc2pred_helper.score_df()
        self.feature_matrix = self.feature_matrix.join(predictions)
        mixmhc2pred_helper.draw_prediction_distributions(predictions, self.labels)


    def add_autort_predictions(self):
        """
        Run AutoRT and add predicted RT scores to the training feature matrix.

        :return: None
        """
        autort_helper = AutortHelper(self.peptides_with_mods, self.exp_rts, self.high_prob_indices, fine_tune=self.fine_tune, verbose=False, report_directory=self.report_directory)
        autort_helper.predict_df()
        predictions = autort_helper.score_df()
        self.feature_matrix = self.feature_matrix.join(predictions)
        autort_helper.draw_prediction_distributions(predictions, self.labels)


    def add_deeplc_predictions(self):
        """
        Run DeepLC and add predicted RT scores to the training feature matrix.

        :return: None
        """
        deeplc_helper = DeepLCHelper(self.peptides, self.peptides_with_mods, self.exp_rts, self.high_prob_indices, fine_tune=self.fine_tune, verbose=False, report_directory=self.report_directory)
        deeplc_helper.predict_df()
        predictions = deeplc_helper.score_df()
        self.feature_matrix = self.feature_matrix.join(predictions)
        deeplc_helper.draw_prediction_distributions(predictions, self.labels)


    def add_im2deep_predictions(self):
        """
        Run IM2Deep and add predicted CCS scores to the training feature matrix.

        :return: None
        """
        if np.max(self.exp_ims) == 0:
            print('Cannot read ion mobility from experimental data. Skipping IM2Deep predictions...')
            return
        im2deep_helper = IM2DeepHelper(self.peptides, self.peptides_with_mods, self.charges, self.exp_ims,
                                       self.high_prob_indices, report_directory=self.report_directory,
                                       fine_tune=self.fine_tune, verbose=False)
        im2deep_helper.predict_df()
        predictions = im2deep_helper.score_df()
        self.feature_matrix = self.feature_matrix.join(predictions)
        im2deep_helper.draw_prediction_distributions(predictions, self.labels)


    def add_peptdeep_predictions(self):
        """
        Run AlphaPeptDeep and add presentation predictions to the training feature matrix.

        :return: None
        """
        peptdeep_helper = PeptDeepHelper(self.peptides, self.raw_data, self.report_directory)
        peptdeep_helper.predict_df()
        predictions = peptdeep_helper.score_df()
        self.feature_matrix = self.feature_matrix.join(predictions)
        peptdeep_helper.draw_prediction_distributions(predictions, self.labels)


    def add_koina_predictions(self):
        attempt = 0
        max_retries = 5  # Maximum number of retries

        while attempt < max_retries:
            try:
                koina_helper = KoinaHelper(self.peptides, self.peptides_with_mods, self.charges,
                                           predictor_names=self.koina_predictors,
                                           exp_rts=self.exp_rts, exp_ims=self.exp_ims, exp_ms2s=self.exp_ms2s,
                                           high_prob_indices=self.high_prob_indices,
                                           instrument_type='QE',
                                           fragmentation_type='HCD',
                                           koina_server_url=self.koina_server_url,
                                           report_directory=self.report_directory
                                           )
                koina_helper.predict_df()
                predictions = koina_helper.score_df()
                self.feature_matrix = self.feature_matrix.join(predictions)
                koina_helper.draw_prediction_distributions(predictions, self.labels)
                break
            except Exception as e:
                print(f"Warning: {e}")
                attempt += 1
                if attempt < max_retries:
                    print(f"Retrying in 1 minute... (Attempt {attempt}/{max_retries})")
                    time.sleep(60)
                else:
                    print("Max retries reached. Operation failed.")
                    raise

    def add_all_predictors(self, use_ccs=False):

        # RT, MS2, CCS
        if self.fine_tune:
            self.add_autort_predictions()
            self.add_deeplc_predictions()
            self.koina_predictors = general_koina_predictors.copy()
            self.koina_predictors.remove('Deeplc_hela_hf')
            self.koina_predictors.remove('Chronologer_RT')
            if not use_ccs:
                self.koina_predictors.remove('IM2Deep')
                self.koina_predictors.remove('AlphaPeptDeep_ccs_generic')
            else:
                self.add_im2deep_predictions()
                self.koina_predictors.remove('IM2Deep')
            self.add_koina_predictions()
        else:
            self.add_autort_predictions()
            self.koina_predictors = general_koina_predictors.copy()
            self.koina_predictors.remove('Chronologer_RT')
            if not use_ccs:
                self.koina_predictors.remove('IM2Deep')
                self.koina_predictors.remove('AlphaPeptDeep_ccs_generic')
            self.add_koina_predictions()



    def _set_seed(self, random_seed: int = None):
        if random_seed is None:
            random_seed = self.random_seed
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)


    def get_nn_model(self,
                     learning_rate: float = 0.001,
                     dropout: float = 0.5,
                     hidden_layers: int = 2,
                     width_ratio: float = 5.0,
                     alpha: float = 0.5):
        """
        Return a compiled multilayer perceptron neural network with the indicated architecture.

        :param learning_rate: Learning rate used by the optimizer (adam).
        :param dropout: Dropout between each layer.
        :param hidden_layers: Number of hidden layers.
        :param width_ratio: Ratio of width of hidden layers to width of input layer.
        :param alpha: Ratio of decoy to target
        :return: A compiled keras.Model
        """

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model = get_model_without_peptide_encoding(self.feature_matrix.shape[1],
                                                   dropout=dropout,
                                                   hidden_layers=hidden_layers,
                                                   max_pep_length=self.max_len,
                                                   width_ratio=width_ratio)
        model.compile(loss=focal_loss(gamma=0, alpha=alpha), optimizer=optimizer)

        return model

    def get_nn_model_with_sequence_encoding(self,
                                            learning_rate: float = 0.001,
                                            dropout: float = 0.5,
                                            hidden_layers: int = 2,
                                            width_ratio: float = 5.0,
                                            convolutional_layers: int = 1,
                                            filter_size: int = 4,
                                            n_filters: int = 12,
                                            filter_stride: int = 3,
                                            n_encoded_sequence_features: int = 6,
                                            alpha: float = 0.5):
        """
        Return a compiled neural network, similar to get_nn_model but also includes a convolutional network for
        encoding peptide sequences which feeds into the multilayer perceptron.

        :param learning_rate: Learning rate used by the optimizer (adam).
        :param dropout:  Dropout between each layer.
        :param hidden_layers: Number of hidden layers.
        :param width_ratio: Ratio of width of hidden layers to width of input layer.
        :param convolutional_layers: Number of convolutional layers.
        :param filter_size: Convolution filter size.
        :param n_filters: Number of filters.
        :param filter_stride: Filter stride.
        :param n_encoded_sequence_features: Number of nodes in the output of the convolutional network.
        :param alpha: Ratio of decoy to target
        :return: A compiled keras.Model
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        max_len = self.max_len if self.mhc_class == 'I' else self.max_len * 2
        model = get_model_with_peptide_encoding(ms_feature_length=self.feature_matrix.shape[1],
                                                dropout=dropout,
                                                hidden_layers_after_convolutions=hidden_layers,
                                                after_convolutions_width_ratio=width_ratio,
                                                convolutional_layers=convolutional_layers,
                                                filter_size=filter_size,
                                                n_filters=n_filters,
                                                filter_stride=filter_stride,
                                                n_encoded_sequence_features=n_encoded_sequence_features,
                                                max_pep_length=max_len
                                                )
        model.compile(optimizer=optimizer, loss=focal_loss(gamma=0, alpha=alpha))
        return model


    def run(self,
            additional_training_data=None,
            n_splits: int = 3,
            early_stopping_patience: int = 10,
            weight_by_inverse_peptide_counts: bool = False,
            random_seed: int = None,
            clear_session: bool = True,
            fit_model: bool = True,
            report_directory: Union[str, PathLike] = None,
            rt_predictors: List[str] = None,
            ms2_predictors: List[str] = None,
            ccs_predictors: List[str] = None,
            app_predictors: List[str] = None,
            auto_predict_predictor = False,
            koina_server_url: str = 'koina.wilhelmlab.org:443',
            fine_tune: bool = False,
            sequence_encoding: bool = False,
            mzml_folder: PathLike = None,
            fasta_path: PathLike = None,
            psm_fdr: float = 1,
            pep_fdr: float = 1,
            seq_fdr: float = 1,
            remove_decoy: bool = False,
            force_rerun=True,
            **kwargs):


        self.report_directory = str(report_directory)
        report_directory = Path(report_directory)
        report_directory.mkdir(parents=True, exist_ok=True)

        # Skip processing if results already exist
        if not force_rerun and (report_directory / 'peptide.tsv').exists():
            print(f'Results already exist in {report_directory}. Skipping reprocessing.')
            return

        self.input_log_path = report_directory / 'input_files.txt'
        if self.input_log_path.exists():
            self.input_log_path.unlink()
        append_log(f'psm={self.filepath}', self.input_log_path, False)
        append_log(f'fasta={fasta_path}', self.input_log_path, False)

        self.mzml_path = None
        if mzml_folder is not None:
            mzml_paths = Path(mzml_folder).rglob('*.mzML')
            mzml_map = {path.stem.replace('_uncalibrated', ''): str(path.expanduser().resolve()) for path in mzml_paths}
            mzml_name = self.filepath.name.replace('_edited.pin', '').replace('.pin', '')
            assert mzml_name in mzml_map.keys(), f'mzML file not found: {mzml_folder}/{mzml_name}.mzML '
            self.mzml_path = mzml_map[mzml_name]
            append_log(f'mzml={self.mzml_path}', self.input_log_path, False)
            if '_uncalibrated.' in self.mzml_path:
                self.spec_names, self.spec_indices, self.exp_rts, self.exp_ims, self.exp_ms2s = \
                    get_rt_ccs_ms2_from_msfragger_mzml(self.mzml_path, self.raw_data['ScanNr'],
                                                       self.raw_data['ExpMass'].astype(float), self.charges)
            else:
                self.spec_names, self.spec_indices, self.exp_rts, self.exp_ims, self.exp_ms2s = \
                    get_rt_ccs_ms2_from_mzml(self.mzml_path, self.raw_data['ScanNr'],
                                             self.raw_data['ExpMass'].astype(float), self.charges)

        if clear_session:
            K.clear_session()

        if random_seed is None:
            random_seed = self.random_seed
        self._set_seed(random_seed)

        self.rt_predictors = rt_predictors if rt_predictors is not None else []
        self.ms2_predictors = ms2_predictors if ms2_predictors is not None else []
        self.ccs_predictors = ccs_predictors if ccs_predictors is not None else []
        self.app_predictors = app_predictors if app_predictors is not None else []
        self.auto_predict_predictor = auto_predict_predictor
        self.fine_tune = fine_tune
        self.koina_server_url = koina_server_url

        if len(self.app_predictors) > 0:
            self.app_predictors = [p.lower() for p in self.app_predictors]
            if 'netmhcpan' in self.app_predictors or 'netmhciipan' in self.app_predictors:
                self.add_netmhcpan_predictions()
            if 'mhcflurry' in self.app_predictors and self.mhc_class == 'I':
                self.add_mhcflurry_predictions()
            if 'bigmhc' in self.app_predictors and self.mhc_class == 'I':
                self.add_bigmhc_predictions()
            if 'mixmhc2pred' in self.app_predictors and self.mhc_class == 'II':
                self.add_mixmhc2pred_predictions()

        if self.auto_predict_predictor:
            self.load_psm_coordinates()
            self.add_all_predictors()
            self.rt_predictors, self.ms2_predictors = predict_best_combination(self.feature_matrix, report_directory)
            self.ccs_predictors = ['im2deep', 'alphapeptdeep_ccs_generic']

            features_all = self.raw_data.join(self.feature_matrix, how='left', rsuffix='_right')
            features_all = features_all[[col for col in features_all.columns if '_right' not in col]]
            features_all.to_csv(report_directory / 'all_features.tsv', index=False, sep='\t')

            drop_columns = []
            for column in self.feature_matrix.columns:
                if column.endswith('_log_rt_error'):
                    predictor = column[:-13]
                    if predictor not in self.rt_predictors:
                        drop_columns.append(predictor + '_rt_error')
                        drop_columns.append(predictor + '_log_rt_error')
                        drop_columns.append(predictor + '_rt_rel_error')
                        drop_columns.append(predictor + '_log_rt_rel_error')
                if column.endswith('_entropy_score'):
                    predictor = column[:-14]
                    if predictor not in self.ms2_predictors:
                        drop_columns.append(predictor + '_entropy_score')
                        drop_columns.append(predictor + '_cosine_score')
                        drop_columns.append(predictor + '_forward_score')
                        drop_columns.append(predictor + '_reverse_score')
                        drop_columns.append(predictor + '_entropy_b_score')
                        drop_columns.append(predictor + '_entropy_y_score')
            self.feature_matrix.drop(drop_columns, axis=1, inplace=True)
        else:
            predictor_types = []
            if len(self.rt_predictors) > 0:
                self.rt_predictors = [p.lower() for p in self.rt_predictors]
                predictor_types.append('RT')
            if len(self.ms2_predictors) > 0:
                predictor_types.append('MS2')
                self.ms2_predictors = [p.lower() for p in self.ms2_predictors]
            if len(self.ccs_predictors) > 0:
                predictor_types.append('CCS')
                self.ccs_predictors = [p.lower() for p in self.ccs_predictors]

            self.load_psm_coordinates(predictor_types=predictor_types)
            if self.exp_ims is None or np.any(self.exp_ims == None) or np.max(self.exp_ims) <= 0:
                self.ccs_predictors = []

            if 'autort' in self.rt_predictors:
                self.add_autort_predictions()
            if 'deeplc' in self.rt_predictors:
                self.add_deeplc_predictions()
            if 'im2deep' in self.ccs_predictors and self.fine_tune:
                self.add_im2deep_predictions()

            self.koina_predictors = []
            for predictor in self.rt_predictors:
                if predictor not in ['autort', 'deeplc']:
                    self.koina_predictors.append(predictor)
            self.koina_predictors += self.ms2_predictors
            for predictor in self.ccs_predictors:
                if predictor == 'im2deep' and self.fine_tune:
                    continue
                self.koina_predictors.append(predictor)
            if len(self.koina_predictors) > 0:
                self.add_koina_predictions()

        features_all = self.raw_data.join(self.feature_matrix, how='left', rsuffix='_right')
        features_all = features_all[[col for col in features_all.columns if '_right' not in col]]
        features_all.to_csv(report_directory / 'features.tsv', index=False, sep='\t')

        # Initialize training model
        alpha = np.sum(self.labels != 1) / np.sum(self.labels == 1)
        alpha = max(0.1, min(alpha, 0.9))
        if not sequence_encoding:
            model_args = {key: arg for key, arg in kwargs.items() if key in signature(self.get_nn_model).parameters}
            kwargs = {key: arg for key, arg in kwargs.items() if key not in model_args}
            model_args['alpha'] = alpha
            model = self.get_nn_model(**model_args)
        else:
            model_args = {key: arg for key, arg in kwargs.items() if key in
                          signature(self.get_nn_model_with_sequence_encoding).parameters}
            kwargs = {key: arg for key, arg in kwargs.items() if key not in model_args}
            model_args['alpha'] = alpha
            model = self.get_nn_model_with_sequence_encoding(**model_args)
            if self.encoded_peptides is None:
                self.encode_peptide_sequences()
            additional_training_data = self.encoded_peptides

        if 'epochs' not in kwargs.keys():
            print('`epochs` was not passed as a keyword argument. Setting it to default value of 30')
            kwargs['epochs'] = 30
        if 'batch_size' not in kwargs.keys():
            print('`batch_size` was not passed as a keyword argument. Setting it to default value of 512')
            kwargs['batch_size'] = 512

        # prepare data for training
        all_data = self.feature_matrix.copy(deep=True).values
        labels = deepcopy(self.labels)
        peptides = self.peptides

        skf = k_fold_split(s=self.raw_data['ExpMass'].to_numpy(dtype=float), peptides=peptides, k_folds=n_splits, random_state=random_seed)

        predictions = np.zeros_like(labels, dtype=float)
        k_splits = np.zeros_like(labels, dtype=int)

        history = []

        now = str(datetime.now()).replace(' ', '_').replace(':', '-')
        initial_model_weights = str(self.model_dir / f'mhcvalidator_initial_weights_{now}.keras')
        model.save(initial_model_weights)

        for k_fold, (train_index, predict_index) in enumerate(skf):
            print('-----------------------------------')
            print(f'Training on split {k_fold+1}')
            self._set_seed(random_seed)

            if isinstance(model, keras.Model):
                model.load_weights(initial_model_weights)
            feature_matrix = deepcopy(all_data)

            mask = np.ones_like(labels[train_index], dtype=bool)  # just in case we implement the q-value subset again

            x_train = deepcopy(feature_matrix[train_index, :][mask])
            rnd_idx = RandomState(random_seed).choice(len(x_train), len(x_train), replace=False)
            x_train = x_train[rnd_idx]
            x_predict = deepcopy(feature_matrix[predict_index, :])
            input_scalar = NDStandardScaler()
            input_scalar = input_scalar.fit(x_train)
            x_train = input_scalar.transform(x_train)
            x_predict = input_scalar.transform(x_predict)
            feature_matrix = input_scalar.transform(feature_matrix)

            x = deepcopy(feature_matrix)
            x_train = deepcopy(x_train)
            x_predict = deepcopy(x_predict)
            train_labels = labels[train_index][mask][rnd_idx]
            predict_labels = labels[predict_index]
            print(f' Training split - {np.sum(train_labels == 1)} targets | {np.sum(train_labels == 0)} decoys')
            print(f' Prediction split - {np.sum(predict_labels == 1)} targets | {np.sum(predict_labels == 0)} decoys')

            if weight_by_inverse_peptide_counts:
                pep_counts = Counter(peptides[train_index][mask])
                weights = np.array([np.sqrt(1 / pep_counts[p]) for p in peptides[train_index][mask][rnd_idx]])
            else:
                weights = np.ones_like(labels[train_index][mask][rnd_idx])

            if additional_training_data is not None:
                additional_training_data = deepcopy(additional_training_data)
                x2_train = additional_training_data[train_index][mask][rnd_idx]
                x2_test = additional_training_data[predict_index]
                input_scalar2 = NDStandardScaler()
                input_scalar2 = input_scalar2.fit(x2_train)

                x2_train = input_scalar2.transform(x2_train)
                x2_test = input_scalar2.transform(x2_test)

                x_train = (x_train, x2_train)
                x_predict = (x_predict, x2_test)

            model_fit_parameters = eval(f'signature(model.fit)').parameters
            if 'validation_data' in model_fit_parameters.keys():
                val_str = 'validation_data=(x_predict, predict_labels),'
            else:
                val_str = ''

            if 'sample_weight' in model_fit_parameters.keys():
                weight_str = 'sample_weight=weights,'
            else:
                weight_str = ''

            if isinstance(model, keras.Model):
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    verbose=1,
                    mode="auto",
                    restore_best_weights=False)
                now = str(datetime.now()).replace(' ', '_').replace(':', '-')
                model_name = str(self.model_dir / f'mhcvalidator_k={k_fold+1}_{now}.keras')
                checkpoint = keras.callbacks.ModelCheckpoint(model_name,
                                                             monitor='val_loss', verbose=0,
                                                             save_best_only=True, mode='min')
                callbacks_str = 'callbacks=[early_stopping, checkpoint],'
            else:
                callbacks_str = ''
                model_name = ''

            # Train the model
            if fit_model:
                fit_history = eval(f"model.fit(x_train, train_labels, {val_str} {weight_str} {callbacks_str} **kwargs)")
                if model_name != '':
                    model.load_weights(model_name)
                    if report_directory is not None:
                        model_directory = report_directory / 'model'
                        model_directory.mkdir(parents=True, exist_ok=True)
                        model.save(model_directory / f'mhcvalidator_model_k={k_fold+1}.keras')
            else:
                fit_history = None

            if fit_history is not None and hasattr(fit_history, 'history'):
                history.append(fit_history)

            predict_preds = eval(f"model.predict(x_predict)").flatten()  # all these predictions are assumed to be arrays. we flatten them because sometimes the have an extra dimension of size 1
            predict_qs = calculate_qs(predict_preds.flatten(), predict_labels)
            predictions[predict_index] = predict_preds
            k_splits[predict_index] = k_fold + 1
            assert np.all(predict_labels == self.labels[predict_index])

            pep_qvalue, _, pep_labels, _, _ = calculate_peptide_level_qs(predict_preds, predict_labels, self.peptides_with_mods[predict_index])
            seq_qvalue, _, seq_labels, _, _ = calculate_peptide_level_qs(predict_preds, predict_labels, self.peptides[predict_index])

            print(f' | PSMs in this split validated at 1% FDR: {np.sum((predict_qs <= 0.01) & (predict_labels == 1))}')
            print(f' | Peptides in this split validated at 1% FDR: {np.sum((pep_qvalue <= 0.01) & (pep_labels == 1))}')
            print(f' | Sequences in this split validated at 1% FDR: {np.sum((seq_qvalue <= 0.01) & (seq_labels == 1))}')
            print('-----------------------------------')

        self.predictions = np.empty(len(labels), dtype=float)
        self.qs = np.empty(len(labels), dtype=float)
        self.predictions = predictions

        psm_qvalue = calculate_qs(predictions, labels)
        pep_qvalue, _, pep_labels, _, _ = calculate_peptide_level_qs(self.predictions, self.labels, self.peptides_with_mods)
        seq_qvalue, _, seq_labels, _, _ = calculate_peptide_level_qs(self.predictions, self.labels, self.peptides)
        print('===================================')
        print('Validation results')
        print(f' | PSMs validated at 1% FDR: {np.sum((psm_qvalue <= 0.01) & (self.labels == 1))}')
        print(f' | Peptides validated at 1% FDR: {np.sum((pep_qvalue <= 0.01) & (pep_labels == 1))}')
        print(f' | Sequences validated at 1% FDR: {np.sum((seq_qvalue <= 0.01) & (seq_labels == 1))}')
        print('===================================')

        run_reporter = RunReporter(report_directory, self.decoy_tag)
        run_reporter.add_run_result(spec_names=self.spec_names, spec_indices=self.spec_indices,
                                    rts=self.exp_rts, ims=self.exp_ims, masses=self.raw_data['ExpMass'].astype(float),
                                    charges=self.charges,
                                    peptides=self.peptides_with_mods, sequences=self.peptides,
                                    prev_aas=self.prev_aas, next_aas=self.next_aas,
                                    labels=self.labels, scores=self.predictions,
                                    proteins=self.raw_data['Proteins'].str.replace('@', '', regex=False))
        run_reporter.add_app_score()
        run_reporter.generate_pep_xml(fasta_path=fasta_path)
        run_reporter.generate_psm_report(psm_fdr=psm_fdr, remove_decoy=remove_decoy)
        run_reporter.generate_peptide_report(pep_fdr=pep_fdr, remove_decoy=remove_decoy, sequential=True, psm_fdr=psm_fdr)
        run_reporter.generate_sequence_report(seq_fdr=seq_fdr, remove_decoy=remove_decoy, sequential=True, psm_fdr=psm_fdr)
        run_reporter.draw_result_figure([h.history['val_loss'] for h in history])


def run_mhcbooster(pin_files, sequence_encoding, alleles, mhc_class, app_predictors, auto_predict_predictor,
                   rt_predictors, ms2_predictors, ccs_predictors, fine_tune, fasta_path, mzml_folder, output_folder):
    output_folder = Path(output_folder)
    for pin in pin_files:
        pin = Path(pin)
        file_name = pin.stem
        print(file_name)
        mhcb = MHCBooster(max_threads=max(1, os.cpu_count() - 2))
        mhcb.set_mhc_params(alleles=alleles, mhc_class=mhc_class)
        mhcb.load_data(pin, filetype='pin')
        mhcb.run(sequence_encoding=sequence_encoding,
                 app_predictors=app_predictors,
                 auto_predict_predictor=auto_predict_predictor,
                 rt_predictors=rt_predictors,
                 ms2_predictors=ms2_predictors,
                 ccs_predictors=ccs_predictors,
                 fine_tune=fine_tune,
                 n_splits=3,
                 mzml_folder=mzml_folder,
                 fasta_path=fasta_path,
                 report_directory=output_folder / f'{file_name}')

        if auto_predict_predictor:
            rt_predictors = mhcb.rt_predictors
            ms2_predictors = mhcb.ms2_predictors
            ccs_predictors = mhcb.ccs_predictors
            auto_predict_predictor = False

    combined_reporter = CombinedReporter(result_folder=output_folder, fasta_path=fasta_path)
    combined_reporter.run()