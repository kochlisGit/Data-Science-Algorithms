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

# Detect outliers with Isolation Forest.
n_estimators = 200
contamination = 0.03
outliers_data_indices = anomaly_detection.isolation_forest(
    rating_budget_df,
    n_estimators,
    contamination,
    False
)

normal_data = rating_budget_df[outliers_data_indices == 1]
outliers_data = rating_budget_df[outliers_data_indices == -1]

# Plotting Normal data + Outliers.
anomaly_detection.plot_normal_outliers_data(normal_data['rating'],
                                            normal_data['budget'],
                                            outliers_data['rating'],
                                            outliers_data['budget'],
                                            'Isolation Forest Plot',
                                            'Rating',
                                            'Budget')
