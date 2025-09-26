import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use Nature-style fonts and theme
sns.set_theme(style="whitegrid", font="Arial", rc={
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 300
})

def draw_length_distribution(path, col_name, title=None):
    # Read the TSV file
    df = pd.read_csv(path, sep="\t")

    # Set figure size in inches (Nature figure panels: ~3.5 x 2.5 for single column)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Draw histogram
    sns.histplot(
        data=df,
        x=col_name,
        bins=range(df[col_name].min(), df[col_name].max() + 1),
        edgecolor="black",
        linewidth=0.5,
        color="black",
        ax=ax
    )

    # Axis formatting
    ax.set_xlabel("Peptide length (mer)", labelpad=5)
    ax.set_ylabel("Count", labelpad=5)
    ax.set_title(title)  # No title for journal figures
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tight layout for clean export
    plt.tight_layout()

    # Save as vector (e.g., PDF or SVG) for journal submission
    # plt.savefig("histogram_peptide_length.pdf", dpi=300)
    plt.show()

    # # Plot the histogram using Seaborn
    # plt.figure(figsize=(6, 4))
    # sns.histplot(data=df, x=col_name, bins=range(df[col_name].min(), df[col_name].max() + 1))
    #
    # # Labeling
    # plt.xlabel("AA Length")
    # plt.ylabel("Frequency")
    # plt.title(title)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    draw_length_distribution("/mnt/e/data/JY_HLA-II/mhcbooster_comb/combined_sequence.tsv", "seq_len", "Peptide IDs")