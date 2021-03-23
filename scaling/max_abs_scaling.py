import matplotlib.pyplot as plt
import pandas as pd
import scaling

# Reading data from csv file.
data_df = pd.read_csv('../movies_data.csv')
rating_df = data_df['rating']

# Scaling data according to z values.
max_abs_scaled_data = scaling.transform('MaxAbs', rating_df.to_numpy().reshape((-1, 1))).reshape(1, -1)
max_abs_scaled_data_df = pd.DataFrame({
    'rating': max_abs_scaled_data[0]
})

# Plotting Rating's scaled histogram.
ax = max_abs_scaled_data_df.plot.hist(title='Max Abs Scaled Rating Data - Histogram', bins=14, edgecolor='black')
ax.set_xlabel('Scaled Ratings')
plt.show()
