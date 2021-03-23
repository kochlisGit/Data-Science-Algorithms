import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


# Computes the ideal number of bins for histogram plot, according to Freedman-Diaconis rule.
def freedman_diaconis_rule(data):
    data = np.array(data)
    IQR = stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy='omit')
    N = data.size
    bin_width = (2*IQR)/np.power(N, 1/3)
    return int((data.max() - data.min() / bin_width)+1)


# Computes the ideal number of bins for histogram plot.
def sturge_law(data):
    data = np.array(data)
    N = data.size
    return int(1 + np.log2(N))


# Plots the distribution of data, Mean, Median, STD
def plot_distribution(plot_data_df, bins_generator, title, x_label, y_label):
    n_bins = bins_generator(plot_data_df.values.T)
    ax = plot_data_df.plot.hist(title=title, bins=n_bins, edgecolor='black')

    # Plotting mean of histogram.
    mean = plot_data_df.mean()
    plt.axvline(mean, color='red', linewidth=2, label='Mean')

    # Plotting median of histogram.
    median = plot_data_df.median()
    plt.axvline(median, color='orange', linewidth=2, label='Median')

    # Plotting standard deviation of histogram
    std = plot_data_df.std()
    plt.axvline(mean-std, color='cyan', linestyle='dashed', linewidth=2, label='Standard Deviation')
    plt.axvline(mean+std, color='cyan', linestyle='dashed', linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='best')
    plt.show()

    # Constructing QQ-Plot
    stats.probplot(plot_data_df, plot=plt)
    plt.show()


# Reading data from csv.
data_df = pd.read_csv('../movies_data.csv')

# Extracting ratings.
rating_df = data_df['rating']
plot_distribution(rating_df, freedman_diaconis_rule, 'Ratings Distribution', 'Ratings', 'Counts')

# Extracting budget
budget_df = data_df['budget']
plot_distribution(budget_df, sturge_law, 'Budget Distribution', 'Budget', 'Counts')