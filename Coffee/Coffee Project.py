# Import Libraries

import numpy as np
import seaborn as sns
import datetime as dt
import pandas as pd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', None)

sales = pd.read_csv("Projects/CRM/Coffee Project/201904 sales reciepts.csv")
customers = pd.read_csv("Projects/CRM/Coffee Project/customer.csv")

# Data Understanding

def check_df(dataframe):
    print(dataframe.shape)
    print("#" * 15)
    print(dataframe.isnull().sum())
    print("#" * 15)
    print(dataframe.nunique())
    print("#" * 15)
    print(dataframe.dtypes)
    print("#" * 15)

check_df(customers)
check_df(sales)

sales["transaction_date"] = pd.to_datetime(sales["transaction_date"])

sales.groupby("customer_id").agg({"transaction_id": "count"}).sort_values("transaction_id", ascending=False).head(10)

sales = sales[sales["customer_id"] != 0]

# Calculating RFM Metrics

sales["transaction_date"].max()

analyze_date = dt.datetime(2019,5,1)

rfm = sales.groupby("customer_id").agg({"transaction_date": lambda x: (analyze_date - x.max()).days,
                                        "transaction_id" : lambda x: x.nunique(),
                                        "line_item_amount" : lambda x: x.sum()})

rfm.columns = ["R","F","M"]

rfm = rfm[rfm["M"] > 0]

rfm.head()

# Calculating RFM Scores

rfm["R_Score"] = pd.qcut(rfm["R"].rank(method = "first"), 5, labels = [5,4,3,2,1])
rfm["F_Score"] = pd.qcut(rfm["F"].rank(method = "first"), 5, labels = [1,2,3,4,5])
rfm["M_Score"] = pd.qcut(rfm["M"],5, labels = [1,2,3,4,5])

rfm["RF_Score"] = (rfm["R_Score"].astype(str) +
                   rfm["F_Score"].astype(str))

rfm.head()

# Creating RFM Segments

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}

rfm["Segment"] = rfm["RF_Score"].replace(seg_map, regex=True)

rfm[["Segment", "R", "F", "M"]].groupby("Segment").agg(["mean", "count"])

# CLTV

cltv_df = sales.groupby("customer_id").agg({"transaction_id": lambda x: x.nunique(),
                                        "quantity": lambda x: x.sum(),
                                        "line_item_amount": lambda x: x.sum()})

cltv_df.columns = ["total_transaction", "total_unit", "total_price"]
cltv_df.head()

##  Average Order Value

cltv_df["average_order_value"] = cltv_df["total_price"] / cltv_df["total_transaction"]

## Purchase Frequency

cltv_df["purchase_frequency"] = cltv_df["total_transaction"] / cltv_df.shape[0]

## Repeat Rate & Churn Rate

repeat_rate = cltv_df[cltv_df["total_transaction"] > 1].shape[0] / cltv_df.shape[0]

churn_rate = 1 - repeat_rate

## Profit Margin

cltv_df["profit_margin"] = cltv_df["total_price"] * 0.10

## Customer Value

cltv_df["customer_value"] = cltv_df["average_order_value"] * cltv_df["purchase_frequency"]

## Customer Lifetime Value

cltv_df["Cltv"] = (cltv_df["customer_value"] / churn_rate) * cltv_df["profit_margin"]

cltv_df.head()

# Creating Segments (for CLTV)

cltv_df["Segment"] = pd.qcut(cltv_df["Cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.sort_values(by="Cltv", ascending=False).head()

cltv_df.groupby("Segment").agg({"count", "mean", "sum"})

# CLTV Prediction

def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(sales, "quantity")
replace_with_thresholds(sales, "unit_price")

sales["line_item_amount"] = sales["quantity"] * sales["unit_price"]

cltv_df2 = sales.groupby("customer_id").agg({"transaction_date" : [lambda date: (date.max() - date.min()).days,
                                                                      lambda date: (analyze_date - date.min()).days],
                                               "transaction_id" : lambda x: x.nunique(),
                                               "line_item_amount" : lambda x: x.sum()})


cltv_df2.columns = cltv_df2.columns.droplevel(0)
cltv_df2.columns = ["R", "T", "F", "M"]

cltv_df2["M"] = cltv_df2["M"] / cltv_df2["F"]

cltv_df2 = cltv_df2[(cltv_df2['F'] > 1)]

cltv_df2["R_Weekly"] = cltv_df2["R"] / 7

cltv_df2["T_Weekly"] = cltv_df2["T"] / 7

# BG-NBD Modelling

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df2["F"],
        cltv_df2["R_Weekly"],
        cltv_df2["T_Weekly"])

# Who are the customers we expect to shop the most in a week?

bgf.predict(1,
            cltv_df2["F"],
            cltv_df2["R_Weekly"],
            cltv_df2["T_Weekly"]).sort_values(ascending=False).head(10)

cltv_df["expected_purchase_1_week"] = bgf.predict(1,
                                              cltv_df2["F"],
                                              cltv_df2["R_Weekly"],
                                              cltv_df2["T_Weekly"])

# Who are the customers we expect to shop the most in a month?

bgf.predict(4,
            cltv_df2["F"],
            cltv_df2["R_Weekly"],
            cltv_df2["T_Weekly"]).sort_values(ascending=False).head(10)

cltv_df["expected_purchase_1_month"] = bgf.predict(4,
                                               cltv_df2["F"],
                                               cltv_df2["R_Weekly"],
                                               cltv_df2["T_Weekly"])


# Who are the customers we expect to shop the most in 3 months?

bgf.predict(4 * 3,
            cltv_df2["F"],
            cltv_df2["R_Weekly"],
            cltv_df2["T_Weekly"]).sum()

cltv_df["expected_purchase_3_month"] = bgf.predict(4 * 3,
                                               cltv_df2["F"],
                                               cltv_df2["R_Weekly"],
                                               cltv_df2["T_Weekly"])

# Let's see some graphics

plot_period_transactions(bgf)
plt.show()

# Gamma-Gamma Modelling

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df2["F"], cltv_df2["M"])

ggf.conditional_expected_average_profit(cltv_df2["F"],
                                        cltv_df2["M"]).head(10)

cltv_df2["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df2["F"],
                                                                             cltv_df2["M"])

ggf.conditional_expected_average_profit(cltv_df2["F"],
                                        cltv_df2["M"]).sort_values(ascending=False).head(10)

cltv_df2.sort_values("expected_average_profit", ascending=False).head(10)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df2["F"],
                                   cltv_df2["R"],
                                   cltv_df2["T"],
                                   cltv_df2["M"],
                                   time=3,
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index()
cltv.head()

cltv_final = cltv_df2.merge(cltv, on="customer_id", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Creation of Segments (for CLTV_FINAL)

cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(10)

cltv_final.groupby("Segment").agg({"count", "mean", "sum"})

