from __future__ import division
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from chart_studio import plotly
'''import plotly.offline as pyoff
import plotly.graph_objs as go'''
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# Load the data
df = pd.read_csv("customer_segmentation_.csv")

# Convert 'InvoiceDate' to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')


# Filter data for United Kingdom
df_uk = df.query("Country=='United Kingdom'").reset_index(drop=True)

# Filter data for 3 months and 6 months periods
df_3m = df_uk[(df_uk.InvoiceDate >= datetime(2011, 6, 1)) & (df_uk.InvoiceDate < datetime(2011, 9, 1))].reset_index(drop=True)
df_6m = df_uk[(df_uk.InvoiceDate >= datetime(2011, 6, 1)) & (df_uk.InvoiceDate < datetime(2011, 12, 1))].reset_index(drop=True)

# Create a DataFrame with unique CustomerIDs
df_user = pd.DataFrame(df_3m['CustomerID'].unique(), columns=['CustomerID'])

# Define a function to order clusters
def order_cluster(cluster_field_name, target_field_name, x, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    x_new = x.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    x_new = x_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    x_new['index'] = x_new.index
    x_final = pd.merge(x, x_new[[cluster_field_name,'index']], on=cluster_field_name)
    x_final = x_final.drop([cluster_field_name], axis=1)
    x_final = x_final.rename(columns={"index": cluster_field_name})
    return x_final

# Recency calculation
df_max_purchase = df_3m.groupby('CustomerID').InvoiceDate.max().reset_index()
df_max_purchase.columns = ['CustomerID', 'MaxPurchaseDate']
df_max_purchase['Recency'] = (df_max_purchase['MaxPurchaseDate'].max() - df_max_purchase['MaxPurchaseDate']).dt.days
df_user = pd.merge(df_user, df_max_purchase[['CustomerID','Recency']], on='CustomerID')
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_user[['Recency']])
df_user['RecencyCluster'] = kmeans.predict(df_user[['Recency']])
df_user = order_cluster('RecencyCluster', 'Recency', df_user, False)

# Frequency calculation
df_frequency = df_3m.groupby('CustomerID').InvoiceDate.count().reset_index()
df_frequency.columns = ['CustomerID', 'Frequency']
df_user = pd.merge(df_user, df_frequency, on='CustomerID')
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_user[['Frequency']])
df_user['FrequencyCluster'] = kmeans.predict(df_user[['Frequency']])
df_user = order_cluster('FrequencyCluster', 'Frequency', df_user, True)

# Monetary calculation
df_3m['Revenue'] = df_3m['UnitPrice'] * df_3m['Quantity']
df_revenue = df_3m.groupby('CustomerID').Revenue.sum().reset_index()
df_user = pd.merge(df_user, df_revenue, on='CustomerID')
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_user[['Revenue']])
df_user['RevenueCluster'] = kmeans.predict(df_user[['Revenue']])
df_user = order_cluster('RevenueCluster', 'Revenue', df_user, True)

# Calculate OverallScore and Segment
df_user['OverallScore'] = df_user['RecencyCluster'] + df_user['FrequencyCluster'] + df_user['RevenueCluster']
df_user['Segment'] = 'Low-Value'
df_user.loc[df_user['OverallScore'] > 2, 'Segment'] = 'Mid-Value' 
df_user.loc[df_user['OverallScore'] > 4, 'Segment'] = 'High-Value'

# 6 months revenue analysis
df_6m['Revenue'] = df_6m['UnitPrice'] * df_6m['Quantity']
df_user_6m = df_6m.groupby('CustomerID')['Revenue'].sum().reset_index()
df_user_6m.columns = ['CustomerID', 'm6_Revenue']

# Plot 6 months revenue
'''plot_data = [go.Histogram(x=df_user_6m.query('m6_Revenue < 10000')['m6_Revenue'])]
plot_layout = go.Layout(title='6m Revenue')
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)'''

# Merge data for clustering
df_merge = pd.merge(df_user, df_user_6m, on='CustomerID', how='left')
df_merge = df_merge.fillna(0)

# Plot LTV vs OverallScore
df_graph = df_merge.query("m6_Revenue < 30000")
'''plot_data = [
    go.Scatter(
        x=df_graph.query("Segment == 'Low-Value'")['OverallScore'],
        y=df_graph.query("Segment == 'Low-Value'")['m6_Revenue'],
        mode='markers',
        name='Low',
        marker=dict(size=7, line=dict(width=1), color='blue', opacity=0.8)
    ),
    go.Scatter(
        x=df_graph.query("Segment == 'Mid-Value'")['OverallScore'],
        y=df_graph.query("Segment == 'Mid-Value'")['m6_Revenue'],
        mode='markers',
        name='Mid',
        marker=dict(size=9, line=dict(width=1), color='green', opacity=0.5)
    ),
    go.Scatter(
        x=df_graph.query("Segment == 'High-Value'")['OverallScore'],
        y=df_graph.query("Segment == 'High-Value'")['m6_Revenue'],
        mode='markers',
        name='High',
        marker=dict(size=11, line=dict(width=1), color='red', opacity=0.9)
    ),
]

plot_layout = go.Layout(
    yaxis={'title': "6m LTV"},
    xaxis={'title': "RFM Score"},
    title='LTV'
)
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)'''

# Remove outliers
df_merge = df_merge[df_merge['m6_Revenue'] < df_merge['m6_Revenue'].quantile(0.99)]

# Create 3 LTV clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_merge[['m6_Revenue']])
df_merge['LTVCluster'] = kmeans.predict(df_merge[['m6_Revenue']])

# Order LTV clusters
df_merge = order_cluster('LTVCluster', 'm6_Revenue', df_merge, True)

# Convert categorical columns to numerical
df_class = pd.get_dummies(df_merge)

# Feature set and label
X = df_class.drop(['LTVCluster', 'm6_Revenue'], axis=1)
y = df_class['LTVCluster']

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

# XGB Classifier
ltv_xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, objective='multi:softprob', n_jobs=-1).fit(X_train, y_train)
'''print('Accuracy of XGB classifier on training set: {:.2f}'.format(ltv_xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'.format(ltv_xgb_model.score(X_test[X_train.columns], y_test)))
'''
# Classification report
y_pred = ltv_xgb_model.predict(X_test)
#print(classification_report(y_test, y_pred))

# Save the result
df_class.to_csv('output.csv', index=False)
