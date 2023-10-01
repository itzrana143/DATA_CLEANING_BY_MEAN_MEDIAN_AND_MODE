# DATA_CLEANING_BY_MEAN_MEDIAN_AND_MODE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\HP\Desktop\iplData.csv')

df

df.shape

df.head()

df.tail(6)

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

df

df.info()

df.isnull()

df.isnull().sum()

plt.figure(figsize=(16,9))
sns.heatmap(df.isnull())

null_var=df.isnull().sum()/df.shape[0]*100
null_var

drop_columns=null_var[null_var>90].keys()
drop_columns

df2_drop_clm=df.drop(columns=drop_columns)
df2_drop_clm

df2_drop_clm.shape

df3_num=df2_drop_clm.select_dtypes(include=['int64','float64'])
df3_num
#df3_num.head()

plt.figure(figsize=(16,9))
sns.heatmap(df3_num.isnull())

df3_num[df3_num.isnull().any(axis=1)]

df3_num.isnull().sum()

missing_num_var=[var for var in df3_num.columns if df3_num[var].isnull().sum()>0]
missing_num_var

plt.figure(figsize=(10,10))
sns.set()
for i,var in enumerate(missing_num_var):
    plt.subplot(2,2,i+1)
    sns.distplot(df3_num[var],bins=20,kde_kws={'linewidth':5,'color':'red'})

# when missing Data Is Randomly we use mean and median

df4_num_mean=df3_num.fillna(df3_num.mean())
df4_num_mean.isnull().sum().sum()

plt.figure(figsize=(10,10))
sns.set()
for i,var in enumerate(missing_num_var):
    plt.subplot(2,2,i+1)
    sns.distplot(df3_num[var],bins=20,kde_kws={'linewidth':5,'color':'red'},label="original")
    sns.distplot(df4_num_mean[var],bins=20,kde_kws={'linewidth':5,'color':'green'},label="Mean")
    plt.legend()

df5_num_median=df3_num.fillna(df3_num.median())
df5_num_median.isnull().sum().sum()

plt.figure(figsize=(10,10))
sns.set()
for i,var in enumerate(missing_num_var):
    plt.subplot(2,2,i+1)
    sns.distplot(df3_num[var],bins=20,hist=False,kde_kws={'linewidth':8,'color':'red'},label="original")
    sns.distplot(df4_num_mean[var],bins=20,hist=False,kde_kws={'linewidth':5,'color':'green'},label="Mean")
    sns.distplot(df5_num_median[var],bins=20,hist=False,kde_kws={'linewidth':3,'color':'k'},label="Median")
    plt.legend()

#for outler we use boxplot 

for i,var in enumerate(missing_num_var):
    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    sns.boxplot(df[var])
    plt.subplot(3,1,2)
    sns.boxplot(df4_num_mean[var])
    plt.subplot(3,1,3)
    sns.boxplot(df5_num_median[var])

df_concat=pd.concat([df3_num[missing_num_var],df4_num_mean[missing_num_var],df5_num_median[missing_num_var]],axis=1)

df_concat[df_concat.isnull().any(axis=1)]

