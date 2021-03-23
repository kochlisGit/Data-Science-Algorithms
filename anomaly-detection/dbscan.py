import pandas as pd
import anomaly_detection

# Reading data from csv.
data_df = pd.read_csv('../movies_data.csv')
rating_budget_df = data_df[['rating', 'budget']]

# Plotting original data.
anomaly_detection.plot_original_data(rating_budget_df['rating'],
                                     rating_budget_df['budget'],
                                     'Original Plot',
                                     'Rating',
                                     'Budget')

# Detect outliers with Envelope.
epsilon = 0.3
min_samples = 20
outliers_data_indices = anomaly_detection.dbscan(
    rating_budget_df,
    epsilon,
    min_samples,
    True
)

normal_data = rating_budget_df[outliers_data_indices != -1]
outliers_data = rating_budget_df[outliers_data_indices == -1]

# Plotting Normal data + Outliers.
anomaly_detection.plot_normal_outliers_data(normal_data['rating'],
                                            normal_data['budget'],
                                            outliers_data['rating'],
                                            outliers_data['budget'],
                                            'DBSCAN Plot',
                                            'Rating',
                                            'Budget')
