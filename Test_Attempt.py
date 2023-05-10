#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import HTML

InteractiveShell.ast_node_interactivity = "all"
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler 

import statsmodels.api as sm
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[5]:


df = pd.read_csv("s3://bcg-rise-prod-sagemaker-input-data/wave-2/SalesData.csv")


# ## ***Understanding the data***

# In[6]:


df.head(2)


# In[7]:


df.info()


# ## ***Data Cleansing***
# - dropping of columns, specific rows, and conversion of day to datetime
# - Note: data cleansing of this portion is saved as df2

# #### Dropping columns

# In[8]:


#drop columns 'product_type', 'taxes', 'product_vendor', 
df2 = df.drop(['product_type', 'taxes', 'product_vendor' ], axis=1)
df2.head(5)


# In[9]:


#checking if net_sales column vales are the same in total_sales
df2.net_sales.equals(df2.total_sales)


# In[10]:


#net_sales and total_sales are the same, dropping column net_sales
df2 = df.drop(['product_type', 'taxes', 'product_vendor', 'net_sales' ], axis=1)
df2.head(5)


# #### Dropping rows

# In[11]:


#only including rows where net_quantity is greater than 0
df2 = df2[(df2['net_quantity'] > 0)]
df2.info()


# In[12]:


#only including rows where customer ID is greater than 0 (eliminating the 0 IDS)
df2 = df2[(df2['customer_id'] > 0)]
df2.info()


# In[13]:


#only including rows where gross_sales is greater than $0
df2 = df2[(df2['gross_sales'] > 0)]
df2.info()


# In[14]:


#only including rows where total_sales is greater than $0 - because some may have been given discounts
df2 = df2[(df2['total_sales'] > 0)]
df2.info()


# In[15]:


# Change dtype of 'day' column to datetime format 
df2['day'] = pd.to_datetime(df2['day'])


# In[16]:


df2.describe()


# In[17]:


df2.info()


# ## ***Data Manipulation***
# - Addition of columns 

# #### Adding Month, Day of the week, Month and year to dataframe df2

# In[19]:


#add a month and a day column and save new data frame as df3

#add month column
df2['month'] = df2['day'].dt.month
#add day of the week column
df2['weekday'] = df2['day'].dt.weekday
#add year and month column 
df2['year_month'] = df2['day'].dt.to_period('M')
#dropping a mistake i made earlier - can remove later. remb to change it to df3

df2.head(3)


# ### Note: Also added columns recency, frequency and monetary in RFM analysis below to df3

# ## ***Data quering - Exploritory Data Analysis***

# ### Company's total sales (revenue) from 2021 - 2022

# In[16]:


# group data by 'year_month' and calculate total sales
monthly_sales = df2.groupby('year_month')['total_sales'].sum()

# convert 'year_month' column to string
monthly_sales.index = monthly_sales.index.astype(str)

plt.figure(figsize=(10, 6))

# plot the data
plt.plot(monthly_sales.index, monthly_sales.values)
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales')
plt.show()


# ### Company's total sales (revenue) in 2021

# In[17]:


# filter rows for year 2021
df2_2021 = df2[df2['day'].dt.year == 2021]

# group data by 'year_month' and calculate total sales
monthly_sales_2021 = df2_2021.groupby('year_month')['total_sales'].sum()

# convert 'year_month' column to string
monthly_sales_2021.index = monthly_sales_2021.index.astype(str)

# set the figure size
plt.figure(figsize=(10, 6))

# plot the data
plt.plot(monthly_sales_2021.index, monthly_sales_2021.values)
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales in 2021')
plt.show()


# ### Company's total sales (revenue) in 2022

# In[18]:


# filter rows for year 2022
df2_2022 = df2[df2['day'].dt.year == 2022]

# group data by 'year_month' and calculate total sales
monthly_sales_2022 = df2_2022.groupby('year_month')['total_sales'].sum()

# convert 'year_month' column to string
monthly_sales_2022.index = monthly_sales_2022.index.astype(str)

# set the figure size
plt.figure(figsize=(10, 6))

# plot the data
plt.plot(monthly_sales_2022.index, monthly_sales_2022.values)
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales in 2022')
plt.show()


# ### ***EDA on product sales based on revenue and quantity***
# EDA for trip-driving products

# #### What products are sold in the highest quantity? 

# In[23]:


#what products are sold in highest quantity and show their revenue

# group the data by product_title and calculate total quantity and total sales
product_sales_qty = df2.groupby('product_title')['net_quantity'].sum().reset_index()
product_sales_value = df2.groupby('product_title')['total_sales'].sum().reset_index()

# merge the two dataframes
product_sales = pd.merge(product_sales_qty, product_sales_value, on='product_title')

# sort the data in descending order by total sales
sorted_sales = product_sales.sort_values('net_quantity', ascending=False)

pd.set_option("display.max_rows", 10)
sorted_sales


# #### What products contribute to the greatest revenue?

# In[24]:


#### what products contribute to the greatest revenue?

# group the data by product_title and calculate total quantity and total sales
product_sales_qty = df2.groupby('product_title')['net_quantity'].sum().reset_index()
product_sales_value = df2.groupby('product_title')['total_sales'].sum().reset_index()

# merge the two dataframes
product_sales = pd.merge(product_sales_qty, product_sales_value, on='product_title')

# sort the data in descending order by total sales
sorted_sales = product_sales.sort_values('total_sales', ascending=False)

sorted_sales


# In[21]:


sorted_sales.describe()


# ## ***Determining Trip Driving Products***
# 
# The concept to determine the items that drives the greatest sales and transactions for CTF is based on a calculation of these factors: 
# 
# 1. N_baskets: how many baskets does a product appear
# 
# 2. N_unique: sum of total unique items in baskets where item is present
# 
# 3. % share of basket: n_baskets / n_unique
# 
# 4. Flag/Flagship: item with the largest share of basket 
# 
# 5. Flag/Flagship %: Flag/Flagship / N_baskets

# In[20]:


df2.head(2)


# ### Finding n_baskets 

# In[21]:


# Find the 'n_basket' by finding the total count of 'order_name' for each 'product title'

df_n_baskets_for_each_product_title = df2.groupby("product_title")['order_name'].count().reset_index()
df_n_baskets_for_each_product_title = df_n_baskets_for_each_product_title .sort_values(by='order_name', ascending=False)
df_n_baskets_for_each_product_title.head(3)


# In[22]:


df_n_baskets_for_each_product_title.columns = ["product_title", "n_baskets"]
df_n_baskets_for_each_product_title.head(3) 


# In[23]:


#checking to make sure it's 218 items 
df_n_baskets_for_each_product_title.describe()


# In[24]:


#merge with original data frame so that we can work out the % share of basket later on
df2_add_n_baskets = pd.merge(df2, df_n_baskets_for_each_product_title, on='product_title' )
df2_add_n_baskets.head(1)


# ### Finding n_unique 

# #### Creating a dataframe of the order name and getting the number of unique items in each order name 

# In[25]:


df_n_unique_products_at_transactional_level = df2.groupby("order_name")['product_title'].nunique().reset_index()
df_n_unique_products_at_transactional_level.columns = ['order_name', 'n_unique_products_at_transactional_level']
df_n_unique_products_at_transactional_level.sort_values(by='n_unique_products_at_transactional_level', ascending=False)


# In[26]:


#merge with original data frame so that we can work out the % share of basket later on
df2_add_n_baskets_add_n_unique = pd.merge(df2_add_n_baskets, df_n_unique_products_at_transactional_level, on = 'order_name' )
df2_add_n_baskets_add_n_unique.head(1)


# In[27]:


df_n_unique = df2_add_n_baskets_add_n_unique.groupby('product_title')['n_unique_products_at_transactional_level'].sum().reset_index()
df_n_unique.columns = ['product_title', 'n_unique']
df_n_unique.sort_values(by='n_unique', ascending = False)


# #### At this point i have 3 dataframes:
# 1. df2_add_n_baskets_add_n_unique (already put back into main dataframe for n_baskets and transaction level n_unique)
# 2. df_n_unique (total n_unique not yet in dataframe

# ### Finding % share of basket : n_basket / n_unique 

# In[28]:


df2_n_basket_n_unique_final = pd.merge(df2_add_n_baskets_add_n_unique, df_n_unique, on = 'product_title' )
df2_n_basket_n_unique_final.head(1)


# In[29]:


df2_percent_share = df2_n_basket_n_unique_final[['product_title', 'n_baskets', 'n_unique']]
df2_percent_share


# In[30]:


df2_percent_share_clean = df2_percent_share.drop_duplicates()
df2_percent_share_clean


# In[31]:


df2_percent_share_clean['%_share_of_basket'] = df2_percent_share_clean['n_baskets'] / df2_percent_share_clean['n_unique']
df2_percent_share_clean


# In[33]:


df2_n_basket_n_unique_percent_share = pd.merge(df2_n_basket_n_unique_final, df2_percent_share_clean, on = 'product_title' )
df2_n_basket_n_unique_percent_share.head(1)


# In[36]:


df2_n_basket_n_unique_percent_share_filtered = df2_n_basket_n_unique_percent_share.filter(['product_title', 'order_name', 'n_baskets_x', 'n_unique_x', '%_share_of_basket'])
df2_n_basket_n_unique_percent_share_filtered.sort_values(by= 'order_name' , ascending=True).head()


# ### Flag
# - Each time a product has the highest % share in the transaction(order_name), it is a flag
# - Add up the total number of flags

# In[37]:


df2_n_basket_n_unique_percent_share_filtered.sort_values(by= 'order_name' , ascending=True).head(3)


# I think that there are 2 ways i can do this, the most straightforward way would be to arrange it in descending order and count the first item in the row as 1

# #### Method 1: Get the first value of the highest % share of basket in each row

# In[41]:


#method 1

df2_n_basket_n_unique_percent_share_filtered = df2_n_basket_n_unique_percent_share_filtered.sort_values(by = ['order_name', '%_share_of_basket'] , ascending=[True, False])
df2_n_basket_n_unique_percent_share_filtered


# In[45]:


df2_n_basket_n_unique_percent_share_filtered.groupby(['order_name']).first()['product_title'].value_counts().reset_index()


# In[68]:


df2_flag = df2_n_basket_n_unique_percent_share_filtered.groupby(['order_name']).first()['product_title'].value_counts().reset_index()
df2_flag.columns = ['product_title', 'flag']
df2_flag.head(3)


# In[65]:


df2_n_basket_n_unique_percent_share_flag = pd.merge(df2_n_basket_n_unique_percent_share_filtered, df2_flag, on = 'product_title')
df2_n_basket_n_unique_percent_share_flag.head(2)


# #### Method 2 : Encoding with 1 then sum up 

# In[53]:


df_grouped_1 = df2_n_basket_n_unique_percent_share_filtered.groupby(['order_name', 'product_title'])['%_share_of_basket'].sum().reset_index()
df_grouped_1.head(1)


# In[56]:


df_max = df_grouped_1.loc[df_grouped_1.groupby('order_name')['%_share_of_basket'].idxmax()]
df_max


# In[60]:


# Create a new column indicating if a product_title has the highest % share in its corresponding order_name
df_max['has_highest_share'] = 1
df_max.head(2)


# In[64]:


df_final = df_max.groupby('product_title')['has_highest_share'].sum().reset_index()
df_final.sort_values(by = 'has_highest_share', ascending = False)


# ### % Flagship
# - % Flag / n_baskets

# In[72]:


df2_n_basket_n_unique_percent_share_flag_filtered = df2_n_basket_n_unique_percent_share_flag.drop_duplicates(subset = ['product_title'])
df2_n_basket_n_unique_percent_share_flag_filtered


# In[74]:


df2_n_basket_n_unique_percent_share_flag_filtered['% Flagship']  =  (df2_n_basket_n_unique_percent_share_flag_filtered['flag'] / df2_n_basket_n_unique_percent_share_flag_filtered['n_baskets_x']) * 100
df2_n_basket_n_unique_percent_share_flag_filtered


# In[75]:


df2_n_basket_n_unique_percent_share_flag_filtered.sort_values(by = '% Flagship', ascending = False)


# In[ ]:





# In[ ]:





# ## RFM Analysis

# ### Recency
# When was a customers' most recent purchase?

# In[76]:


# 'date of last purchase' calculation and creation of new dataframe (df2_recency)

df2_recency = df2.groupby(by='customer_id',
                        as_index=False)['day'].max()

df2_recency.columns = ['customer_id', 'day']

recent_date = df2_recency['day'].max()

df2_recency['Recency'] = df2_recency['day'].apply(
                                                            lambda x: (recent_date - x).days)

df2_recency.head()


# In[77]:


#drop the irrelavant column from the 'df2_recency' 

df2_recency_final = df2_recency.drop(["day"], axis=1)
df2_recency_final.head(5)


# In[78]:


# merge the 'df2_recency' dataframe and df2 dataframe to give df3

df3 = pd.merge(df2, df2_recency_final, on='customer_id')
df3.head(3)


# In[79]:


#double check to make sure everything is in order
df3[df3['customer_id'] == 2.986860e+12]


# ### Frequency
# How often do they purchase?

# In[80]:


df3_frequency = df3.groupby("customer_id", as_index=False)["order_name"].nunique()

df3_frequency.columns = ['customer_id','Frequency']
df3_frequency.head()


# In[38]:


df3_frequency.describe()


# In[81]:


df3 = pd.merge(df3, df3_frequency, on='customer_id')
df3.head(2)


# In[ ]:


#double check to make sure everything is in order
df3[df3['customer_id'] == 2.986860e+12]


# ### Monetary
# How much revenue we get from their visit or how much do they spend when they purchase?
# 
# ***Monetary can be calculated as the sum of the Amount of all orders by each customer.***

# In[82]:


df3_monetary = df3.groupby(by="customer_id", as_index=False )["total_sales"].sum()
df3_monetary


# In[83]:


df3 = pd.merge(df3, df3_monetary, on='customer_id')
df3.head(2)


# In[84]:


df3 = df3.rename(columns={'total_sales_y': 'Monetary'})
df3.head(1)


# ### Creating a new df3_rfm that only includes a dataframe with the values customer_id, recency, frequency, monetary

# In[85]:


df3_rfm = df3[["customer_id", "Recency", "Frequency", "Monetary"]]
df3_rfm.info()


# In[86]:


df3_rfm["customer_id"].nunique()


# In[87]:


df3_rfm_cus = df3_rfm.groupby("customer_id").first().reset_index()
df3_rfm_cus.info()


# In[88]:


df3_rfm_cus = df3_rfm_cus[["Recency", "Frequency", "Monetary"]]
df3_rfm_cus.describe()


# ### Creating a new df4_rfm that only includes a dataframe with the values customer_id, recency, frequency, monetary, with the top 3 highest monetary removed

# In[89]:


#locating the top 5 highest spend as they are outliers as derived from the monetary vs frequency scatterplot above
df3_rfm_cus_top5 = df3_rfm_cus.nlargest(5, "Monetary")
df3_rfm_cus_top5


# In[90]:


df4_rfm = df3_rfm_cus.drop(df3_rfm_cus_top5.index)
df4_rfm.describe()


# ### Plotting RFM analysis for df3_rfm

# #### Heatmap to show correlation of RFM

# In[91]:


sns.heatmap(df3_rfm_cus.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[92]:


plt.scatter(df3_rfm_cus['Recency'], df3_rfm_cus['Frequency'])
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.show()


# In[93]:


plt.scatter(df3_rfm_cus['Monetary'], df3_rfm_cus['Frequency'])
plt.xlabel('Monetary')
plt.ylabel('Frequency')
plt.show()


# #### Distribution of Recency, Frequency, Monetary

# In[94]:


df3_rfm_cus.head()


# In[95]:


sns.histplot(df3_rfm_cus[df3_rfm_cus.Recency>20].Recency)


# In[96]:


sns.histplot(df3_rfm_cus.Frequency)


# In[97]:


sns.histplot(df3_rfm_cus.Monetary)


# In[98]:


df3_rfm_cus.describe()


# ### Plotting RFM analysis for df4_rfm

# In[99]:


plt.scatter(df4_rfm['Monetary'], df4_rfm['Frequency'])
plt.xlabel('Monetary')
plt.ylabel('Frequency')
plt.show()


# In[100]:


sns.histplot(df4_rfm.Monetary)


# ## ***K-means clustering*** on df4_rfm
# 

# In[101]:


df4_rfm.head(2)


# In[105]:


# scaling RFM for df4

from sklearn.preprocessing import StandardScaler

# select the columns to scale
rfm_cols = ['Recency', 'Frequency', 'Monetary']

# create a scaler object
scaler = StandardScaler()

# fit and transform the data
scaled_rfm_df4 = scaler.fit_transform(df4_rfm[rfm_cols])

scaled_rfm_df4


# In[106]:


#elbow on standard scaler 

from sklearn.cluster import KMeans

# determine the optimal number of clusters using the elbow method
rfm_elbow_df4 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_rfm_df4)
    rfm_elbow_df4.append(kmeans.inertia_)
    
# plot the elbow curve
plt.plot(range(1, 11), rfm_elbow_df4)
plt.title('Elbow Method')
plt.show()


# In[107]:


from sklearn.metrics import silhouette_score

# calculate silhouette scores for clusters 2 to 7
for n_clusters in range(2, 8):
    kmeans_df4 = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_df4.fit(scaled_rfm_df4)
    cluster_labels_df4 = kmeans_df4.labels_
    silhouette_avg = silhouette_score(scaled_rfm_df4, cluster_labels_df4)
    print("For n_clusters =", n_clusters,
          "the average silhouette_score is :", silhouette_avg)


# ## ***K-means clustering*** on df3_rfm_cus
# 
# Actions performed
# 1. Creating new dataframe called df_k1 to only include recency, frequency, monetary
# 2. Scale data & K means elbow to determine the clusters to use
# 3. Checking silhoutte score to determine which is the better cluster 
# 4. Getting the mean for clusters 6, 4, 3

# #### df3_rfm_cus : only including: R, F, M

# In[116]:


df3_rfm_cus.head(2)


# In[144]:


# scaling RFM

scaler = StandardScaler()
scaled_rfm = scaler.fit_transform(df3_rfm_cus)
scaled_rfm


# #### Elbow

# In[110]:


from sklearn.cluster import KMeans

# determine the optimal number of clusters using the elbow method
rfm_elbow = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_rfm)
    rfm_elbow.append(kmeans.inertia_)
    
# plot the elbow curve
plt.plot(range(1, 11), rfm_elbow)
plt.title('Elbow Method')
plt.show()


# #### Determining results of culsters - silhouette score

# In[111]:


from sklearn.metrics import silhouette_score

# calculate silhouette scores for clusters 2 to 7
for n_clusters in range(2, 8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_rfm)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(scaled_rfm, cluster_labels)
    print("For n_clusters =", n_clusters,
          "the average silhouette_score is :", silhouette_avg)


# #### 5 clusters

# In[135]:


df3_rfm_cus


# In[145]:


kmeans_5 = KMeans(n_clusters=5, random_state=42)
kmeans_5.fit(scaled_rfm)

df3_rfm_cus['cluster'] = kmeans_5.labels_
df3_rfm_cus


# In[152]:


df3_rfm_cus.groupby('cluster')['Recency', 'Frequency', 'Monetary'].describe()


# #### 6 clusters

# In[154]:


kmeans_6 = KMeans(n_clusters=6, random_state=42)
kmeans_6.fit(scaled_rfm)

df3_rfm_cus['cluster'] = kmeans_6.labels_
df3_rfm_cus.groupby('cluster')['Recency', 'Frequency', 'Monetary'].describe()


# In[ ]:





# ## Market Basket Analysis

# In[155]:


df3.head(3)


# In[156]:


df3 = df3.drop("total_sales_x", axis = 1)


# In[157]:


df3["product_title"].nunique()


# In[158]:


basket_df3_qty = df3.groupby(["order_name", "product_title"])["net_quantity"].sum().unstack().reset_index().fillna(0).set_index("order_name")
basket_df3_qty.head(3)


# In[159]:


basket_df3_qty.describe()


# ### One hot encoding of products

# In[160]:


def encode(x):
    if x >=1:
        return(1)
    else: 
        return(0)


# In[161]:


basket_df3 = basket_df3_qty.applymap(encode)
basket_df3.head()


# In[162]:


basket_df3.describe()


# In[163]:


get_ipython().system('pip install mlxtend')


# In[164]:


from mlxtend.frequent_patterns import association_rules,apriori


# In[165]:


#create freq itemset

freqitemset = apriori(basket_df3, min_support=0.01, use_colnames=True)
freqitemset.sort_values(by='support', ascending = False )


# In[166]:


rules = association_rules(freqitemset, metric='lift', min_threshold = 1 )
rules


# In[167]:


rules.sort_values('lift', ascending = False)


# In[168]:


rules.sort_values('support', ascending = False)


# In[33]:


# filter the DataFrame to only include "Matcha Latte" products
item = df2[df2['product_title'] == "Full Length Matcha Whisk"]

# group the DataFrame by year_month and sum the net_quantity for each group
monthly_sales_item = item.groupby('year_month')['total_sales'].sum()

# plot a bar chart of the monthly sales
monthly_sales_item.plot(kind='bar')
plt.title('Monthly Sales of Selected item')
plt.xlabel('Year-Month')
plt.ylabel('Sales')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# To discard: 
# - Matcha & Hojicha Latte Bundle	(declining sales)
# - Subscription Refill Packs (100g) Auto Renew (declining sales)
# - Matcha Latte
# - Hojicha Latte
# - Matcha Home Cafe Kit (sales stopped in 2021-11. Trend of declining sales from 2021-05 to 2021-11)
# - Matcha Whisk Set (Essential Brewing Kit) (sales stopped in 2021-07 but likely can explore again because there wasn't enough data to support that it was declining)
# - Chawan | Matcha Bowl with Spout (Declining sales MoM)
# - Barista Uji Matcha Powder (20g) (Declining sales and sales stopped in 2022-05)
# - Awakening Matcha Whisk Set (w/ Subscription) Auto renew (Ships every 4 Weeks) (declining sales and sales stopped in 2022-04)
# - $1.99 Islandwide Delivery Latte Bundle (declining sales and sales stopped in 2021-09)
# - Barista Uji Hojicha Powder (20g) (sales stopped in 2022-06 and declining sales)
# - Matcha Starter Kit (No Subscription) (sales is good but only ran from 2022-05 to 2022-06 most likely merged with the regular Matcha Starter Kit)
# - Ceremonial Uji Matcha Powder (20g) (sales stopped 2022-05, delining sales last month it stopped was only 200)
# - Ryokan Escape: Wagashi Tasting Box(actually inconclusive data because one month it did over 2k sales, but sales stopped in 2022-10. sales only ran from 2022-08 to 2022-10)
# - Matcha Home Cafe Kit (with free Barista Uji Matcha) (sales only in 2021-04 and 2021-05). Same as matcha home cafe kit
# - Matcha Latte Quickie	(data was increasing though from 2022-04 to 2022-12 but 2022-12 which was supposed to be good season sale is not as good as 2022-04)
# - Ceremonial Uji Matcha 100g - Subscription (2 Mths Supply) (declining sales rapidly from 2022-06 to 2022-11. sales only good in 2022-06)
# - Hojicha Latte (Subscription) Auto renew (declining sales and stopped at 2022-11)
# - Subscription Refill Packs (100g) (declining sales stopped at 2022-11)
# - Matcha Subscription (Free Chawan Bundle) Auto renew (Ships every 8 Weeks) (declining sales and ended in 2022-04)
# - Matcha Latte (Subscription) Auto renew
# - Matcha Subscription Refill Packs (100g) Auto renew
# - Dirty Matcha Home Cafe Kit (bad sales)
# - Hojicha Latte Quickie	(bad and declining sales)
# - Matcha Subscription Refill Packs (100g)'
# - Mixed Lattes (Pack of 8) (bit conflicted cos only sold during 2022-11 and 2022-12 with sales near 800 each month)
# - Matcha Subscription Refill Packs (100g) Auto Renew',
# - 'Ceremonial Uji Matcha 100g - Subscription',
# - Barista Uji Hojicha Powder (100g) 15.00% Off Auto renew' (bad sales)
# 
# Non-obvious things we shouldn't drop: 
# - Subscription Refill Packs (100g / 3.53oz)
# - Ceremonial Uji Matcha Powder (30g) - sells well in december 2022
# - Ryokan Escape: Japanese Snack Box (only 3 months data from 2022-10 to 2022-12 and looks like good sales)
# - Barista Uji Matcha Powder (30g): good xmas sales 
# - Barista Uji Hojicha Powder (30g) (good xmas sales end of year sales)
# - Mother's Day Pastry Set (only ran for 2022-04 and 2022-05. Can repeat again next year for mother's day)
# - Matcha Sieve (sales picked up towards end of the year and may be able to bundle this with something)
# - Matcha Bamboo Scoop (sales picked up towards end of the year and may be able to bundle this with something)
# - Crystal Chawan | Matcha Bowl with Spout (Double Walled) - good sales for christmas
# 

# In[ ]:




