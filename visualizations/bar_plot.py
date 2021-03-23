import trends_data
import matplotlib.pyplot as plt

region = 'US'
n_hits = 2

keywords = trends_data.get_trending_keywords(n_hits)
hist_df_list = [trends_data.get_historical_data(keyword, region) for keyword in keywords]
means_df = trends_data.compute_means(hist_df_list, keywords)
ax = means_df.plot.bar(title='Bar Plot')
ax.set_xlabel('Hours')
ax.set_ylabel('Score')
plt.show()
