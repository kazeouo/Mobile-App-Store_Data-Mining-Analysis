# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:03:48 2022

@author: sly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

app_data = pd.read_csv("C:/Users/sly/Desktop/UoA/Semester Two/INFOSYS 722 Data Mining and Big Data/Mobile App Store/Iteration 3 OSAS/AppleStore.csv")

#-----------------------------------------------------------------------------------------------------
# Step 2 Data Understanding
print(app_data.shape)

print(app_data.head(5))
print(app_data.tail(5))

print(app_data.dtypes)

print(app_data.iloc[0])
print(app_data.iloc[7196])

print(app_data[["track_name","ver"]])
print(app_data[["currency","cont_rating","prime_genre","ipadSc_urls.num","vpp_lic"]])

print("currency", app_data["currency"].unique())
print("cont_rating", app_data["cont_rating"].unique())
print("prime_genre", app_data["prime_genre"].unique())
print("ipadSc_urls.num", app_data["ipadSc_urls.num"].unique())
print("vpp_lic", app_data["vpp_lic"].unique())

print(app_data.isnull().sum())

print(app_data["size_bytes"].describe())
print(app_data["rating_count_tot"].describe())
print(app_data["rating_count_ver"].describe())
print(app_data["sup_devices.num"].describe())
print(app_data["lang.num"].describe())

#-----------------------------------------------------------------------------------------------------
# Step 3 Data Preparation
print(app_data.columns)
app_data = app_data.rename(columns={"Unnamed: 0":"No.",
                         "id":"ID",
                         "track_name":"AppName",
                         "size_bytes":"AppSize(Bytes)",
                         "currency":"Currency",
                         "price":"Price",
                         "rating_count_tot":"TotalRatingCount",
                         "rating_count_ver":"TotalRatingCount(CurrnetVersion)",
                         "user_rating":"UserRating",
                         "user_rating_ver":"UserRating(CurrentVersion)",
                         "ver":"CurrentVersionNo.",
                         "cont_rating":"AppRestriction",
                         "prime_genre":"AppType",
                         "sup_devices.num":"SupportDeviceNum",
                         "ipadSc_urls.num":"DisplayedScreenshotNum",
                         "lang.num":"SupportLanguageNum",
                         "vpp_lic":"VppLicense"})
print(app_data.columns)

print(app_data[["ID","TotalRatingCount","UserRating"]][app_data["UserRating"] == 0])

null_rating = app_data["UserRating"].replace(0,np.nan)
app_data3 = app_data
app_data3["UserRating"] = null_rating
print(app_data3.isnull().sum())
print(app_data3.isnull().any(axis=1).sum()/app_data3.shape[0])

app_data3 = app_data3.dropna(axis=0, how="any")
print(app_data3.shape)

rating_level = app_data3[["UserRating"]]
rating_level = rating_level.replace([1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],
                           ["Low Rating","Low Rating","Low Rating",
                            "Low Rating","Low Rating","Low Rating",
                            "High Rating","High Rating","High Rating"])
print(app_data3["UserRating"].head(20))
print(rating_level.head(20))

plt.hist(rating_level)
plt.show()

app_data3["RatingLevel"] = rating_level
app_data3 = app_data3[["ID","AppSize(Bytes)","Price","TotalRatingCount",
                     "TotalRatingCount(CurrnetVersion)","AppRestriction","AppType",
                     "SupportDeviceNum","DisplayedScreenshotNum","SupportLanguageNum",
                     "VppLicense","RatingLevel"]]
print(app_data3.columns)

plt.scatter(x=app_data3["ID"],y=app_data3["AppSize(Bytes)"])
plt.show()
plt.scatter(x=app_data3["ID"],y=app_data3["TotalRatingCount"])
plt.show()
plt.scatter(x=app_data3["ID"],y=app_data3["TotalRatingCount(CurrnetVersion)"])
plt.show()
plt.scatter(x=app_data3["ID"],y=app_data3["SupportDeviceNum"])
plt.show()
plt.scatter(x=app_data3["ID"],y=app_data3["SupportLanguageNum"])
plt.show()

treat_outlier = app_data3[["ID","AppSize(Bytes)","TotalRatingCount",
                     "TotalRatingCount(CurrnetVersion)",
                     "SupportDeviceNum","SupportLanguageNum"]]

print(treat_outlier["AppSize(Bytes)"].quantile(0.01)) #3903508.48
print(treat_outlier["AppSize(Bytes)"].quantile(0.99)) #1861763440.6399994
treat_outlier["AppSize(Bytes)"] = np.where(treat_outlier["AppSize(Bytes)"] < 3903508.48, 3903508.48,treat_outlier["AppSize(Bytes)"])
treat_outlier["AppSize(Bytes)"] = np.where(treat_outlier["AppSize(Bytes)"] > 1861763440.6399994, 1861763440.6399994,treat_outlier["AppSize(Bytes)"])

print(treat_outlier["TotalRatingCount"].quantile(0.01)) #1.0
print(treat_outlier["TotalRatingCount"].quantile(0.99)) #288713.30999999976
treat_outlier["TotalRatingCount"] = np.where(treat_outlier["TotalRatingCount"] < 1.0, 1.0,treat_outlier["TotalRatingCount"])
treat_outlier["TotalRatingCount"] = np.where(treat_outlier["TotalRatingCount"] > 288713.30999999976, 288713.30999999976,treat_outlier["TotalRatingCount"])

print(treat_outlier["TotalRatingCount(CurrnetVersion)"].quantile(0.01)) #0.0
print(treat_outlier["TotalRatingCount(CurrnetVersion)"].quantile(0.99)) #7072.53999999999
treat_outlier["TotalRatingCount(CurrnetVersion)"] = np.where(treat_outlier["TotalRatingCount(CurrnetVersion)"] < 0.0, 0.0,treat_outlier["TotalRatingCount(CurrnetVersion)"])
treat_outlier["TotalRatingCount(CurrnetVersion)"] = np.where(treat_outlier["TotalRatingCount(CurrnetVersion)"] > 7072.53999999999, 7072.53999999999,treat_outlier["TotalRatingCount(CurrnetVersion)"])

print(treat_outlier["SupportDeviceNum"].quantile(0.01)) #24.0
print(treat_outlier["SupportDeviceNum"].quantile(0.99)) #43.0
treat_outlier["SupportDeviceNum"] = np.where(treat_outlier["SupportDeviceNum"] < 24.0, 24.0,treat_outlier["SupportDeviceNum"])
treat_outlier["SupportDeviceNum"] = np.where(treat_outlier["SupportDeviceNum"] > 43.0, 43.0,treat_outlier["SupportDeviceNum"])

print(treat_outlier["SupportLanguageNum"].quantile(0.01)) #1.0
print(treat_outlier["SupportLanguageNum"].quantile(0.99)) #34.0
treat_outlier["SupportLanguageNum"] = np.where(treat_outlier["SupportLanguageNum"] < 1.0, 1.0,treat_outlier["SupportLanguageNum"])
treat_outlier["SupportLanguageNum"] = np.where(treat_outlier["SupportLanguageNum"] > 34.0, 34.0,treat_outlier["SupportLanguageNum"])

print(app_data3["AppSize(Bytes)"].skew())
print(treat_outlier["AppSize(Bytes)"].skew())
print(app_data3["TotalRatingCount"].skew())
print(treat_outlier["TotalRatingCount"].skew())
print(app_data3["TotalRatingCount(CurrnetVersion)"].skew())
print(treat_outlier["TotalRatingCount(CurrnetVersion)"].skew())
print(app_data3["SupportDeviceNum"].skew())
print(treat_outlier["SupportDeviceNum"].skew())
print(app_data3["SupportLanguageNum"].skew())
print(treat_outlier["SupportLanguageNum"].skew())

print(app_data3["AppSize(Bytes)"].describe())
print(app_data3["TotalRatingCount"].describe())
print(app_data3["TotalRatingCount(CurrnetVersion)"].describe())
print(app_data3["SupportDeviceNum"].describe())
print(app_data3["SupportLanguageNum"].describe())

print(treat_outlier["AppSize(Bytes)"].describe())
print(treat_outlier["TotalRatingCount"].describe())
print(treat_outlier["TotalRatingCount(CurrnetVersion)"].describe())
print(treat_outlier["SupportDeviceNum"].describe())
print(treat_outlier["SupportLanguageNum"].describe())

plt.scatter(x=treat_outlier["ID"],y=treat_outlier["AppSize(Bytes)"])
plt.show()
plt.scatter(x=treat_outlier["ID"],y=treat_outlier["TotalRatingCount"])
plt.show()
plt.scatter(x=treat_outlier["ID"],y=treat_outlier["TotalRatingCount(CurrnetVersion)"])
plt.show()
plt.scatter(x=treat_outlier["ID"],y=treat_outlier["SupportDeviceNum"])
plt.show()
plt.scatter(x=treat_outlier["ID"],y=treat_outlier["SupportLanguageNum"])
plt.show()


data_divide1 = app_data3[["ID","AppSize(Bytes)","Price","TotalRatingCount",
                     "TotalRatingCount(CurrnetVersion)","AppRestriction","AppType"]]
data_divide2 = app_data3[["ID","SupportDeviceNum","DisplayedScreenshotNum",
                          "SupportLanguageNum","VppLicense","RatingLevel"]]
print(data_divide1.columns)
print(data_divide2.columns)

data_merge = pd.merge(data_divide1,data_divide2,on='ID', how='outer')
print(data_merge.shape)

app_data4 = app_data3
app_data4[["ID","AppSize(Bytes)","TotalRatingCount","TotalRatingCount(CurrnetVersion)",
           "SupportDeviceNum","SupportLanguageNum"]] = treat_outlier

#-----------------------------------------------------------------------------------------------------
# Step 4 Data Transformation
app_data4["RatingLevel"].value_counts()

boost_sample = app_data4
ratinglevel_go = boost_sample["RatingLevel"].replace(["High Rating","Low Rating"],[1,0])
boost_sample["RatingLevel"] = ratinglevel_go
print(boost_sample.head(20))

count_boost_X, count_boost_y = boost_sample["RatingLevel"].value_counts()
boost_X = boost_sample[boost_sample["RatingLevel"]==1]
boost_y = boost_sample[boost_sample["RatingLevel"]==0]
oversample_y = boost_y.sample(count_boost_X, replace = True)
boost_data = pd.concat([boost_X,oversample_y],axis=0)
print(boost_data["RatingLevel"].value_counts())

ratinglevel_back = boost_data
ratinglevel_back["RatingLevel"] = boost_data["RatingLevel"].replace([1,0],["High Rating","Low Rating"])
app_data4 = ratinglevel_back.sort_values(["ID"])
plt.hist(app_data4["RatingLevel"])
plt.show()

price_level_before = app_data3[["Price"]]
price_level_before = price_level_before.replace([0,3.99,0.99,9.99,4.99,7.99,2.99,1.99,5.99,12.99,
                                                 21.99,249.99,6.99,74.99,19.99,8.99,24.99,13.99,
                                                 14.99,16.99,11.99,59.99,15.99,27.99,17.99,299.99,
                                                 49.99,23.99,39.99,99.99,29.99,34.99,20.99,18.99,22.99],
                                                ["Free","Paid","Paid","Paid","Paid","Paid","Paid","Paid",
                                                 "Paid","Paid","Paid","Paid","Paid","Paid","Paid","Paid",
                                                 "Paid","Paid","Paid","Paid","Paid","Paid","Paid","Paid",
                                                 "Paid","Paid","Paid","Paid","Paid","Paid","Paid","Paid",
                                                 "Paid","Paid","Paid",])
plt.hist(price_level_before)
plt.show()

price_level_after = app_data4[["Price"]]
price_level_after = price_level_after.replace([0,3.99,0.99,9.99,4.99,7.99,2.99,1.99,5.99,12.99,
                                               21.99,249.99,6.99,74.99,19.99,8.99,24.99,13.99,
                                               14.99,16.99,11.99,59.99,15.99,27.99,17.99,299.99,
                                               49.99,23.99,39.99,99.99,29.99,34.99,20.99,18.99,22.99],
                                              ["Free","Paid","Paid","Paid","Paid","Paid","Paid","Paid",
                                               "Paid","Paid","Paid","Paid","Paid","Paid","Paid","Paid",
                                               "Paid","Paid","Paid","Paid","Paid","Paid","Paid","Paid",
                                               "Paid","Paid","Paid","Paid","Paid","Paid","Paid","Paid",
                                               "Paid","Paid","Paid",])
plt.hist(price_level_after)
plt.show()

print(price_level_before[price_level_before["Price"]=="Paid"].value_counts()
      /price_level_before.shape[0])
print(price_level_after[price_level_after["Price"]=="Paid"].value_counts()
      /price_level_after.shape[0])

feature_data = app_data4[["AppSize(Bytes)","Price","TotalRatingCount",
                     "TotalRatingCount(CurrnetVersion)","AppRestriction","AppType",
                     "SupportDeviceNum","DisplayedScreenshotNum","SupportLanguageNum",
                     "VppLicense","RatingLevel"]]
feature_data["AppRestriction"] = feature_data["AppRestriction"].replace(["4+","12+","9+","17+"],
                                                                        [1,2,3,4])
feature_data["AppType"] = feature_data["AppType"].replace(["Games","Entertainment","Education",
                                                    "Photo & Video","Utilities","Social Networking",
                                                    "Sports","Productivity","Health & Fitness",
                                                    "Lifestyle","Music","Shopping","Finance",
                                                    "Weather","News","Travel","Food & Drink",
                                                    "Book","Business","Reference","Navigation",
                                                    "Medical","Catalogs"
                                                    ],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                                                       16,17,18,19,20,21,22,23])
feature_data["RatingLevel"] = feature_data["RatingLevel"].replace(["High Rating","Low Rating"],
                                                                  [1,0])
print(feature_data.dtypes)

feature_X = feature_data.iloc[:,0:10]
feature_y = feature_data.iloc[:,-1]
feature= SelectKBest(score_func=chi2, k=10)
feature_score = pd.DataFrame(feature.fit(feature_X,feature_y).scores_)
feature_column = pd.DataFrame(feature_X.columns)
feature_selection = pd.concat([feature_column,feature_score],axis=1)
feature_selection.columns = ["Column","Score"]
print(feature_selection.nlargest(10,"Score"))
      
app_data4 = app_data4[["AppSize(Bytes)","Price","TotalRatingCount",
                     "TotalRatingCount(CurrnetVersion)","AppType","DisplayedScreenshotNum",
                     "SupportLanguageNum","RatingLevel"]]
print(app_data4.columns)
print(app_data4.shape)

#-----------------------------------------------------------------------------------------------------
# Step 7 Data Mining
app_data7 = app_data4
#app_data7.to_csv("C:/Users/sly/Desktop/UoA/Semester Two/INFOSYS 722 Data Mining and Big Data/Mobile App Store/Iteration 3 OSAS/AppleStore_DataMining.csv")

#-----------------------------------------------------------------------------------------------------
# Step 8 Data Mining
app_data8 = ratinglevel_back.sort_values(["ID"])
app_data8["PriceLevel"] = price_level_after
app_data8 = app_data8[["AppSize(Bytes)","AppRestriction","AppType","SupportDeviceNum",
                       "DisplayedScreenshotNum","SupportLanguageNum","RatingLevel","PriceLevel"]]
print(app_data8.columns)
print(app_data8.shape)

newfilter_1 = app_data8[app_data8["AppType"] == "Games"]
newfilter_2 = app_data8[app_data8["AppType"] == "Music"]
newfilter_3 = app_data8[app_data8["AppType"] == "Photo & Video"]
newfilter_4 = app_data8[app_data8["AppType"] == "Reference"]
app_data8 = pd.concat([newfilter_1, newfilter_2, newfilter_3, newfilter_4])
app_data8 = app_data8[app_data8["RatingLevel"] == "High Rating"]
print(app_data8.head(20))
print(app_data8.shape)

feature_new = app_data8[["AppSize(Bytes)","AppRestriction","SupportDeviceNum",
                       "DisplayedScreenshotNum","SupportLanguageNum","PriceLevel","AppType"]]
feature_new["AppRestriction"] = feature_new["AppRestriction"].replace(["4+","12+","9+","17+"],
                                                                        [1,2,3,4])
feature_new["AppType"] = feature_new["AppType"].replace(["Games","Photo & Video",
                                                         "Music","Reference"],
                                                         [1,2,3,4])
feature_new["PriceLevel"] = feature_new["PriceLevel"].replace(["Free","Paid"],
                                                                  [0,1])

feature_newX = feature_new.iloc[:,0:6]
feature_newy = feature_new.iloc[:,-1]
feature2= SelectKBest(score_func=chi2, k=6)
feature_newscore = pd.DataFrame(feature2.fit(feature_newX,feature_newy).scores_)
feature_newcolumn = pd.DataFrame(feature_newX.columns)
feature_newselection = pd.concat([feature_newcolumn,feature_newscore],axis=1)
feature_newselection.columns = ["Column","Score"]
print(feature_newselection.nlargest(6,"Score"))

app_data8 = app_data8[["AppSize(Bytes)","AppRestriction","AppType","SupportDeviceNum",
                       "DisplayedScreenshotNum","SupportLanguageNum"]]
print(app_data8.columns)
#app_data8.to_csv("C:/Users/sly/Desktop/UoA/Semester Two/INFOSYS 722 Data Mining and Big Data/Mobile App Store/Iteration 3 OSAS/AppleStore_DataMining_New.csv")
