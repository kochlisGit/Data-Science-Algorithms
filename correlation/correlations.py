import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


# Constructs data plot of a 2 column variable.
def plot_variable(x, y, title, x_label, y_label):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.scatter(x=x, y=y)
    plt.show()


# Computes the cosine similarity of 2 vectors.
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def correlation(x, y, xlabel, ylabel):
    print('\n', xlabel, '-', ylabel)
    print('Covariance:\n', np.cov(x, y))
    print('Pearson Correlation\n', stats.pearsonr(x, y))
    print('Spearman Correlation\n', stats.spearmanr(x, y))
    print('Fisher-Z Transformation\n', np.arctan(stats.pearsonr(x, y)))
    print('Kendall Correlation\n', stats.kendalltau(x, y))
    print('Weighted Kendall\n', stats.weightedtau(x, y))
    print('Cosine Similarity\n', cosine_similarity(x, y))

# Loading data from csv file.
data_df = pd.read_csv('../movies_data.csv')

# Plotting Rating - Budget relationship.
rating = data_df['rating']
budget = data_df['budget']
plot_variable(rating, budget, 'Rating - Budget Relationship', 'Rating', 'Budget')
correlation(rating, budget, 'Rating', 'Budget')

# Plotting Budget - Revenue relationship.
revenue = data_df['revenue']
plot_variable(budget, revenue, 'Budget - Revenue Relationship', 'Budget', 'Revenue')
correlation(budget, revenue, 'Budget', 'Revenue')

# Plotting Rating - Revenue relationship.
plot_variable(rating, revenue, 'Rating - Revenue Relationship', 'Rating', 'Revenue')
correlation(rating, revenue, 'Rating', 'Revenue')
