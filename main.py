import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/amazon_review.csv")
df.head()
df.shape

df["overall"].mean()


df.head()

a = df["day_diff"].quantile(0.25)
b = df["day_diff"].quantile(0.50)
c = df["day_diff"].quantile(0.75)

df["overall"].mean()


df.loc[df["day_diff"] <= a, "overall"].mean()
df.loc[(df["day_diff"] > a) & (df["day_diff"] <= b), "overall"].mean()
df.loc[(df["day_diff"] > b) & (df["day_diff"] <= c), "overall"].mean()
df.loc[(df["day_diff"] > c), "overall"].mean()


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= a, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > a) & (dataframe["day_diff"] <= b), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > b) & (dataframe["day_diff"] <= c), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > c), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

df.head()


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

def score_up_down_diff(up, down):
    return up - down

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("score_pos_neg_diff", ascending=False).head(20)



def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("score_average_rating", ascending=False).head(20)



def wilson_lower_bound(up, down, confidence=0.95):

    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)

df.sort_values("wilson_lower_bound", ascending=False).head(20)


