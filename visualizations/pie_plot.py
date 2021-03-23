import trends_data
import matplotlib.pyplot as plt

region = 'US'
n_hits = 4

keywords = trends_data.get_trending_keywords(n_hits)
hist_df_list = [trends_data.get_historical_data(keyword, region) for keyword in keywords]
scores_df = trends_data.compute_total_score(hist_df_list, keywords)
ax = scores_df.plot.pie(title='Pie Plot', y='score', autopct='%1.1f%%')
ax.get_legend().remove()
plt.show()
