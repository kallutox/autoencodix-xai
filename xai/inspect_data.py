import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_cf = pd.read_parquet("../data/raw/cf_clinical_data_formatted.parquet")
data_tcga = pd.read_parquet("../data/raw/data_clinical_formatted.parquet")

sex_counts_tcga = data_tcga['SEX'].value_counts()
sex_counts_cf = data_cf['sex'].value_counts()
cf_perc = data_cf['sex'].value_counts(normalize=True) * 100
tcga_perc = data_tcga['SEX'].value_counts(normalize=True) * 100

sample_split = pd.read_parquet("../data/processed/synth_data_10features_09signal/sample_split.parquet")
#print(sample_split.head())


def example_plot():
    # Example data format (replace with your actual data)
    data = pd.DataFrame({
        "latent_dim": np.repeat([f"DIM_{i}" for i in range(1, 9)], 50),
        "latent_intensity": np.random.gamma(2, 2, 400),  # Non-zero variance
        "sex": np.tile(["Female", "Male", "Unknown"], 400 // 3 + 1)[:400]
    })

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    # Define a modern color palette
    palette = sns.color_palette("Set2")

    # Create the ridge plot
    g = sns.FacetGrid(
        data,
        row="latent_dim",
        hue="sex",
        aspect=3,
        height=1,
        palette=palette,
        sharex=False,
        sharey=False,
        margin_titles=False
    )
    g.map(sns.kdeplot, "latent_intensity", alpha=0.7, fill=True, warn_singular=False)

    g.set(xlim=(-3, 12))
    for ax in g.axes.flat:
        ax.set_title("")
    g.set_titles("{row_name}", loc="left", size=11)
    g.set_axis_labels("Latent Intensity", "")
    g.set(yticks=[], ylabel="")
    g.despine(left=True)

    # Adjust spacing and position of the legend
    g.fig.subplots_adjust(hspace=0.5)
    g.add_legend(title="Latent Dimension - sex", loc="upper right", bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    plt.show()


example_plot()

