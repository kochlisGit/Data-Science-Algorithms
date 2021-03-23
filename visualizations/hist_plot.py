import trends_data
import matplotlib.pyplot as plt

region = 'US'
n_hits = 2
n_bins = 12

keywords = trends_data.get_trending_keywords(n_hits)
hist_df_list = [trends_data.get_historical_data(keyword, region) for keyword in keywords]
for df, keyword in zip(hist_df_list, keywords):
    ax = df[keyword].plot.hist(title='Histogram', bins=n_bins, alpha=0.5)
ax.set_xlabel('Scores')
ax.set_ylabel('Counts')
plt.show()
