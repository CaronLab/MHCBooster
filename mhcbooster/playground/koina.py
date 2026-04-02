from koinapy import Koina
import numpy as np
import pandas as pd
# koinapy only takes the input it requires for the current model.
# if you want to compare multiple models you can use a dataframe wit all columns at the same time.
inputs = pd.DataFrame()
inputs['peptide_sequences'] = np.array(["AAAAAKAKM[UNIMOD:35]", "n[-17.0265]QQPSAPQHQGTL"])
inputs['precursor_charges'] = np.array([2, 2])
inputs['collision_energies'] = np.array([25, 25])
inputs['instrument_types'] = np.array(["QE", "QE"])


# If you are unsure what inputs your model requires run `model.model_inputs`
model = Koina("AlphaPeptDeep_ms2_generic", "koina.wilhelmlab.org:443")
predictions = model.predict(inputs)


print(predictions)