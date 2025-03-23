import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

## Data Preparation

df = pd.read_csv("Projects/CRM/Spring/marketing_campaign.csv", sep = "\t")

df.head()

df.info()

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])

df.isnull().sum()

df.dropna(inplace=True)

df.head()

## Calculating RFM Metrics

df["Dt_Customer"].max()

date = pd.to_datetime("2014-12-08")

df["Monetary"] = df.loc[:,df.columns.str.contains("Mnt")].sum(axis=1)
df["Frequency"] = df.loc[:, df.columns.str.contains("Purchases")].sum(axis=1)
df["Recency"] = (date - df["Dt_Customer"]).dt.days

rfm = df[["Recency", "Frequency", "Monetary","ID"]]
rfm.head()

## Calculating RFM Scores

rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], 5, labels = [1,2,3,4,5])
rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"].rank(method = "first"), 5 , labels = [1,2,3,4,5])
rfm["Recency_Score"] = pd.qcut(rfm["Recency"], 5, labels = [5,4,3,2,1])

rfm.head()

rfm["RF_Score"] = (rfm["Recency_Score"].astype(str) +
                   rfm["Frequency_Score"].astype(str))

rfm.head()

## Creating & Analysing RFM Segments

seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
}

rfm["Segment"] = rfm["RF_Score"].replace(seg_map, regex = True)
rfm.head()

##########

## CLTV Calculating

def outlier_threshold(dataframe, variable):
    q1 = dataframe[variable].quantile(0.1)
    q3 = dataframe[variable].quantile(0.99)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    return lower, upper

def replace_with_threshold(dataframe,variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


data = df.copy()
data["Monetary"] = data.loc[:, data.columns.str.contains("Mnt")].sum(axis = 1)
data["Frequency"] = data.loc[:, data.columns.str.contains("Purchases")].sum(axis = 1)
data["Recency"] = (date - data["Dt_Customer"]).dt.days
data["last_order_date"] = dt.datetime(2014, 12, 8) - pd.to_timedelta(data["Recency"])

cltv = pd.DataFrame()

cltv["Recency_Weekly"] = (data["last_order_date"] - data["Dt_Customer"]).dt.days // 7
cltv["T_Weekly"] = (data["last_order_date"] - data["Dt_Customer"]).dt.days // 7
cltv["Frequency"] = data["Frequency"]
cltv["Frequency"] = round(cltv["Frequency"])
cltv["Monetary"] = data["Monetary"]
cltv.index = data["ID"]
cltv = cltv[cltv["Frequency"]>1]
cltv.head()

## BG-NBD Modelling

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv["Frequency"],
        cltv["Recency_Weekly"],
        cltv["T_Weekly"])

cltv["expected_purchase_6_month"] = bgf.predict(4 * 6,cltv['Frequency'],
                                            cltv['Recency_Weekly'],
                                            cltv['T_Weekly']).sum()

cltv.head()

plot_period_transactions(bgf)
plt.show()

## Gamma-Gamma Modelling

ggf = GammaGammaFitter(penalizer_coef=0.001)

ggf.fit(cltv["Frequency"], cltv["Monetary"])

cltv["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv["Frequency"],
                                                                          cltv["Monetary"])

cltv.sort_values("expected_average_profit", ascending = False).head(15)

## BG/NBG & Gamma-Gamma Modelling

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv["Frequency"],
                                   cltv["Recency_Weekly"],
                                   cltv["T_Weekly"],
                                   cltv["Monetary"],
                                   time=3,
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index()

cltv.head()

cltv.sort_values("clv", ascending = False).head(15)

cltv["Segment"] = pd.qcut(cltv["clv"], 4 ,labels = ["D","C","B","A"])
cltv.groupby("Segment").agg({"clv" : ["sum","count","mean"]})
