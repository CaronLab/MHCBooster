import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text


if __name__ == '__main__':
    feature_matrix = pd.read_csv('/mnt/d/workspace/mhc-booster/experiment/JY_1_10_25M/best/JY_Class1_25M_DDA_60min_Slot1-12_1_552_MHCBooster/all_features.tsv', sep='\t')

    # Load example data
    feature_matrix = feature_matrix.iloc[:, 44:]
    # feature_matrix = feature_matrix[[col for col in feature_matrix.columns if 'log_rt_error' in col and 'Chronologer' not in col]]
    # feature_matrix = feature_matrix[[col for col in feature_matrix.columns if 'entropy_score' in col]]
    X = feature_matrix.T
    y = list(range(len(feature_matrix.columns)))

    # Run UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)

    # Plot
    plt.figure(figsize=[12,12])
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=50)
    texts = []
    for xi, yi, label in zip(X_umap[:, 0], X_umap[:, 1], feature_matrix.columns):
        texts.append(plt.text(xi, yi, label, fontsize=9, ha='left', va='top'))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
    plt.title("UMAP Projection of the Iris Dataset")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar()
    plt.show()