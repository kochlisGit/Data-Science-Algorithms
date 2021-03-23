import matplotlib.pyplot as plt
import pandas as pd
import scaling

# Reading data from csv file.
data_df = pd.read_csv('../movies_data.csv')
rating_df = data_df['rating']

# Scaling data according to z values.
z_scaled_data = scaling.transform('Standard', rating_df.to_numpy().reshape((-1, 1))).reshape(1, -1)
z_scaled_data_df = pd.DataFrame({
    'rating': z_scaled_data[0]
})

# Plotting Rating's scaled histogram.
ax = z_scaled_data_df.plot.hist(title='Standard Scaled Rating Data - Histogram', bins=14, edgecolor='black')
ax.set_xlabel('Rating in STD')
plt.show()
