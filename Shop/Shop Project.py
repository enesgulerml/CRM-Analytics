## Imports and Data Preparation

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

orders = pd.read_excel("Projects/CRM/Autumn/Raw Data.xlsx", sheet_name="orders")
customers = pd.read_excel("Projects/CRM/Autumn/Raw Data.xlsx", sheet_name="customers")
products = pd.read_excel("Projects/CRM/Autumn/Raw Data.xlsx", sheet_name="products")

df = pd.merge(orders, customers, on="Customer ID")

df = pd.merge(df, products, on="Product ID")

df.head()

df.info()

df.isnull().sum()

df["Order Date"].max()

today_date = pd.to_datetime("2022-08-21")

df["Total Price"] = df["Quantity"] * df["Unit Price_y"]

df["Email_x"].fillna(0, inplace=True)
df["Email_y"].fillna(0, inplace=True)
df["Phone Number"].fillna(0, inplace=True)

## Creating RFM Metrics

rfm = df.groupby("Customer ID").agg({"Order Date" : lambda x: (today_date - x.max()).days,
                               "Order ID" : lambda x: x.nunique(),
                               "Total Price" : lambda x: x.sum()})

rfm.columns = ["Recency", "Frequency", "Monetary"]

## Creating RFM Score

rfm["Recency Score"] = pd.qcut(rfm["Recency"], 5, labels = [5,4,3,2,1])
rfm["Frequency Score"] = pd.qcut(rfm["Frequency"].rank(method = "first"),5, labels = [1,2,3,4,5])
rfm["Monetary Score"] = pd.qcut(rfm["Monetary"], 5, labels = [1,2,3,4,5])

rfm["RF Score"] = (rfm["Recency Score"].astype(str) +
                   rfm["Frequency Score"].astype(str))

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["Segment"] = rfm["RF Score"].replace(seg_map, regex = True)

sgm= rfm["Segment"].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=sgm.index,y=sgm.values, color = "cyan")
plt.xticks(rotation=45)
plt.title('Customer Segments',color = 'crimson',fontsize=15)
plt.show()

data = df.copy()

## CLTV

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


replace_with_threshold(df, "Unit Price_y")
replace_with_threshold(df, "Price per 100g")
replace_with_threshold(df,"Profit")

df["Total Price"] = df["Quantity"] * df["Unit Price_y"]

date = pd.to_datetime("2022-08-21")

cltv_data = df.groupby("Customer ID").agg({"Order Date": [lambda x: (x.max() - x.min()).days,
                                           lambda y: (date - y.min()).days],
                                           "Order ID" : lambda x: x.nunique(),
                                           "Total Price" : lambda x: x.sum()})

cltv_data.columns = cltv_data.columns.droplevel(0)

cltv_data.columns=['R', 'T', 'F', 'M']

cltv_data["M"] = cltv_data["M"] / cltv_data["F"]
cltv_data = cltv_data[(cltv_data["F"] > 1)]
cltv_data["R_Weekly"] = cltv_data["R"] / 7
cltv_data["T_Weekly"] = cltv_data["T"] / 7

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_data["F"],
        cltv_data["R_Weekly"],
        cltv_data["T_Weekly"])

cltv_data["expected_purchase_1_month"] = bgf.predict(4,
                                              cltv_data['F'],
                                              cltv_data['R_Weekly'],
                                           cltv_data['T_Weekly'])

cltv_data["expected_purchase_1_month"].sort_values(ascending=False).head(10)

## BG-NBD and Gamma-Gamma Prediction

## 3 Month
bgf.predict(4 * 3,
            cltv_data['F'],
            cltv_data['R_Weekly'],
            cltv_data['T_Weekly']).sort_values(ascending=False).head(10)


ggf = GammaGammaFitter(penalizer_coef=0.001)

ggf.fit(cltv_data["F"], cltv_data["M"])

ggf.conditional_expected_average_profit(cltv_data["F"], cltv_data["M"]).sort_values(ascending=False).head(10)

cltv_data["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_data['F'],cltv_data['M'])

cltv_data.sort_values("expected_average_profit", ascending = False).head(15)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_data['F'],
                                   cltv_data['R_Weekly'],
                                   cltv_data['T_Weekly'],
                                   cltv_data['M'],
                                   time=4,
                                   freq="W",
                                   discount_rate=0.01)
cltv.head()

cltv.reset_index()

cltv_final = cltv_data.merge(cltv, on="Customer ID", how="left")
cltv_final.head()

cltv_final.sort_values("clv", ascending = False).head(15)

cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4 ,labels = ["D","C","B","A"])
cltv_final.groupby("Segment").agg({"clv" : ["sum","count","mean"]})