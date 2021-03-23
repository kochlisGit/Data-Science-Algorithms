from pytrends.request import TrendReq
import datetime
import pandas as pd

# Defining engine's configurations.
__trends = TrendReq(hl='en-US', timeout=60, backoff_factor=0.1)

# Defining time history.
__year_start = datetime.datetime.now().year
__month_start = datetime.datetime.now().month-1
__day_start = datetime.datetime.now().day
__hour_start = 0
__year_end = datetime.datetime.now().year
__month_end = datetime.datetime.now().month
__day_end = datetime.datetime.now().day
__hour_end = datetime.datetime.now().hour

# Retrieving trending keywords.
__trends_df = __trends.trending_searches()


# Gets trending search keywords.
def get_trending_keywords(n_hits):
    return __trends_df[0].values[0:n_hits]


# Gets historical data for specified keyword.
def get_historical_data(keyword, region):
    return __trends.get_historical_interest(
        [keyword], geo=region,
        year_start=__year_start, month_start=__month_start, day_start=__day_start, hour_start=__hour_start,
        year_end=__year_end, month_end=__month_end, day_end=__day_end, hour_end=__hour_end,
    )


# Computes mean values from dataframes.
def compute_means(df_list, keywords):
    return pd.DataFrame({
        keyword: [df.loc[df.index.hour == i].mean()[keyword] for i in range(24)]
        for df, keyword in zip(df_list, keywords)
    })


# Computes standard deviation values from dataframes.
def compute_stds(df_list, keywords):
    return pd.DataFrame({
        keyword: [df.loc[df.index.hour == i].std()[keyword] for i in range(24)]
        for df, keyword in zip(df_list, keywords)
    })


# Extracts the sum of values for each dataframe.
def compute_total_score(df_list, keywords):
    return pd.DataFrame({
        'score': [df[keyword].sum() for df, keyword in zip(df_list, keywords)]
    }, index=keywords)
