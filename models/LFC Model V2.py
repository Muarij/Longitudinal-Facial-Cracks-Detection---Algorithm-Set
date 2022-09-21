#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization
import matplotlib.pyplot as plt


# In[53]:


df = pd.read_csv('D:\\Caster\\Year 2022\\LFC Project\\FY 22 23\\LFC Model\\Caster 1 Bad Case\\15th Feb\\Good Case\\14 feb Good Case C11.csv')
#data.sample(n=5)
df


# In[54]:


# Function that calculates the percentage of missing values
def calc_percent_NAs(df):
    nans = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df), columns=['percent']) 
    idx = nans['percent'] > 0
    return nans[idx]
# Let's use above function to look at top ten columns with NaNs
calc_percent_NAs(df).head(10)


# In[55]:


df


# In[56]:


# Extract the readings from the BROKEN state of the pump
broken = df[df['machine_status']=='BROKEN']
# Extract the names of the numerical columns
df2 = df.drop(['machine_status'], axis=1)
names=df2.columns
# Plot time series for each sensor with BROKEN state marked with X in red color
for name in names:
    _ = plt.figure(figsize=(18,3))
    _ = plt.plot(broken[name], linestyle='none', marker='X', color='red', markersize=12)
    _ = plt.plot(df[name], color='blue')
    _ = plt.title(name)
    plt.show()


# In[57]:


# Resample the entire dataset by daily average
#rollmean = df.groupby(df['TIME_STAMP'].min)[['MLF']].mean()
#rollstd = ddf.groupby(df['TIME_STAMP'].min)[['MLF']].std()
#df.groupby(df['Generated On'].hour)[['CB_P']].mean()


# In[58]:


# Standardize/scale the dataset and apply PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
# Extract the names of the numerical columns
df2 = df.drop(['machine_status','TIME_STAMP'], axis=1)
#df2 = df.drop([''], axis=1)
names=df2.columns
x = df[names]
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(x)
# Plot the principal components against their inertia
features = range(pca.n_components_)
_ = plt.figure(figsize=(15, 5))
_ = plt.bar(features, pca.explained_variance_)
_ = plt.xlabel('PCA feature')
_ = plt.ylabel('Variance')
_ = plt.xticks(features)
_ = plt.title("Importance of the Principal Components based on inertia")
plt.show()


# In[59]:


# Calculate PCA with 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])


# In[60]:


principalDf.values


# In[61]:


# Plot ACF
from statsmodels.graphics.tsaplots import plot_acf
#from matplotlib.mlab import PCA
#plot_acf(pca1.dropna(), lags=20, alpha=0.05)


# In[62]:


principalDf.values


# In[63]:


# Import IsolationForest
from sklearn.ensemble import IsolationForest
# Assume that 13% of the entire data set are anomalies
 
outliers_fraction = 0.03
model =  IsolationForest(contamination=outliers_fraction,n_estimators=1000)
model.fit(principalDf.values) 
principalDf['anomaly2'] = pd.Series(model.predict(principalDf.values))
# visualization
df['anomaly2'] = pd.Series(principalDf['anomaly2'].values, index=df.index)
a = df.loc[df['anomaly2'] == -1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(df['BPS25'], color='blue', label='Normal')
_ = plt.plot(a['BPS25'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('[Isolation Forest + PCA Anamoly detection Algorithm] Thermocouple 5')
_ = plt.legend(loc='best')
plt.show();


# In[64]:


df['anomaly2'] = pd.Series(principalDf['anomaly2'].values, index=df.index)
a = df.loc[df['anomaly2'] == -1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(df['BPS5'], color='blue', label='Normal')
_ = plt.plot(a['BPS5'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('[Isolation Forest + PCA Anamoly detection Algorithm] Thermocouple 5')
_ = plt.legend(loc='best')
plt.show();


# In[65]:


df['anomaly2'] = pd.Series(principalDf['anomaly2'].values, index=df.index)
a = df.loc[df['anomaly2'] == -1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(df['BPS40'], color='blue', label='Normal')
_ = plt.plot(a['BPS40'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('[Isolation Forest + PCA Anamoly detection Algorithm] Thermocouple 5')
_ = plt.legend(loc='best')
plt.show();


# In[66]:


df['anomaly2'].value_counts()


# In[67]:


import numpy as np 
import pandas as pd


# In[106]:


df = pd.read_csv('D:\\Caster\\Year 2022\\LFC Project\\FY 22 23\\LFC Model\\Caster 1 Bad Case\\15th Feb\\Bad case\\15TH Feb C1 Bad Case 8am.csv')
#data.sample(n=5)
df


# In[107]:


plt.plot(df['BPS3'],label='BPS 3', color='green')
plt.plot(df['BPS23'],label='BPS 23', color='steelblue', linewidth=4)
plt.plot(df['BPS43'],label='BPS 43', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[108]:


plt.plot(df['BPS4'],label='BPS 4', color='green')
plt.plot(df['BPS24'],label='BPS 24', color='steelblue', linewidth=4)
plt.plot(df['BPS44'],label='BPS 44', color='purple', linestyle='dashed')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[109]:


plt.plot(df['BPS5'],label='BPS 5', color='green')
plt.plot(df['BPS25'],label='BPS 25', color='steelblue', linewidth=4)
plt.plot(df['BPS45'],label='BPS 45', color='purple', linestyle='dashed')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[110]:


plt.plot(df['BPS6'],label='BPS 6', color='green')
plt.plot(df['BPS26'],label='BPS 26', color='steelblue', linewidth=4)
plt.plot(df['BPS46'],label='BPS 46', color='purple', linestyle='dashed')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[111]:


plt.plot(df['BPS7'],label='BPS 7', color='green')
plt.plot(df['BPS27'],label='BPS 27', color='steelblue', linewidth=4)
plt.plot(df['BPS47'],label='BPS 47', color='purple', linestyle='dashed')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[112]:


plt.plot(df['BPS8'],label='BPS 8', color='green')
plt.plot(df['BPS28'],label='BPS 28', color='steelblue', linewidth=4)
plt.plot(df['BPS48'],label='BPS 48', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[113]:


plt.plot(df['BPS4'],label='BPS 9', color='green')
plt.plot(df['BPS24'],label='BPS 29', color='steelblue', linewidth=4)
plt.plot(df['BPS44'],label='BPS 49', color='purple', linestyle='dashed')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[114]:


plt.plot(df['BPS10'],label='BPS 10', color='green')
plt.plot(df['BPS30'],label='BPS 30', color='steelblue', linewidth=4)
plt.plot(df['BPS50'],label='BPS 50', color='purple', linestyle='dashed')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[115]:


plt.plot(df['BPS11'],label='BPS 11', color='green')
plt.plot(df['BPS31'],label='BPS 31', color='steelblue', linewidth=4)
plt.plot(df['BPS51'],label='BPS 51', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[116]:


plt.plot(df['BPS12'],label='BPS 12', color='green')
plt.plot(df['BPS32'],label='BPS 32', color='steelblue', linewidth=4)
plt.plot(df['BPS52'],label='BPS 52', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[117]:


plt.plot(df['BPS15'],label='BPS 15', color='green')
plt.plot(df['BPS39'],label='BPS 35', color='steelblue', linewidth=4)
plt.plot(df['BPS55'],label='BPS 55', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[118]:


plt.plot(df['BPS4'],label='BPS 4', color='green')
plt.plot(df['BPS24'],label='BPS 24', color='steelblue', linewidth=4)
plt.plot(df['BPS44'],label='BPS 44', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[119]:


plt.plot(df['BPS3'],label='BPS 3', color='green')
plt.plot(df['BPS23'],label='BPS 23', color='steelblue', linewidth=4)
plt.plot(df['BPS43'],label='BPS 43', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[120]:


plt.plot(df['BPS2'],label='BPS 2', color='green')
plt.plot(df['BPS22'],label='BPS 22', color='steelblue', linewidth=4)
plt.plot(df['BPS42'],label='BPS 42', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[121]:


plt.plot(df['BPS1'],label='BPS 1', color='green')
plt.plot(df['BPS21'],label='BPS 21', color='steelblue', linewidth=4)
plt.plot(df['BPS41'],label='BPS 41', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[122]:


df = pd.read_csv('D:\\Caster\\Year 2022\\LFC Project\\FY 22 23\\LFC Model\\Caster 1 Bad Case\\15th Feb\\Bad case\\Good Case.csv')
#data.sample(n=5)
df


# In[123]:


plt.plot(df['BPS3'],label='BPS 3', color='green')
plt.plot(df['BPS23'],label='BPS 23', color='steelblue', linewidth=4)
plt.plot(df['BPS43'],label='BPS 43', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Good Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Good Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[124]:


plt.plot(df['BPS4'],label='BPS 4', color='green')
plt.plot(df['BPS24'],label='BPS 24', color='steelblue', linewidth=4)
plt.plot(df['BPS44'],label='BPS 44', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Good Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Good Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[126]:


plt.plot(df['BPS5'],label='BPS 5', color='green')
plt.plot(df['BPS25'],label='BPS 25', color='steelblue', linewidth=4)
plt.plot(df['BPS45'],label='BPS 45', color='purple', linestyle='solid')



plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Good Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Good Case slabs', fontsize=16)
#fig = plt.figure(figsize=(80, 60))
#plt.plot(kind='line', figsize=(10, 5))
plt.show()


# In[127]:


df = pd.read_csv('D:\\Caster\\Year 2022\\LFC Project\\FY 22 23\\LFC Model\\Caster 2 Bad case\\LFC Bad Case\\LFC C2 15 Feb Bad Case.csv')
#data.sample(n=5)
df


# In[136]:


plt.plot(df['BPS3'],label='BPS 3', color='green')
plt.plot(df['BPS23'],label='BPS 23', color='steelblue', linewidth=4)
plt.plot(df['BPS43'],label='BPS 43', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs C2', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs C2', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[135]:


plt.plot(df['BPS4'],label='BPS 4', color='green')
plt.plot(df['BPS24'],label='BPS 24', color='steelblue', linewidth=4)
plt.plot(df['BPS44'],label='BPS 44', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs C2', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs C2', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[130]:


plt.plot(df['BPS5'],label='BPS 5', color='green')
plt.plot(df['BPS25'],label='BPS 25', color='steelblue', linewidth=4)
plt.plot(df['BPS45'],label='BPS 45', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[131]:


plt.plot(df['BPS6'],label='BPS 6', color='green')
plt.plot(df['BPS26'],label='BPS 26', color='steelblue', linewidth=4)
plt.plot(df['BPS46'],label='BPS 46', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[137]:


plt.plot(df['BPS7'],label='BPS 7', color='green')
plt.plot(df['BPS27'],label='BPS 27', color='steelblue', linewidth=4)
plt.plot(df['BPS47'],label='BPS 47', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs C2', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[133]:


plt.plot(df['BPS8'],label='BPS 8', color='green')
plt.plot(df['BPS28'],label='BPS 28', color='steelblue', linewidth=4)
plt.plot(df['BPS48'],label='BPS 48', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[134]:


plt.plot(df['BPS9'],label='BPS 9', color='green')
plt.plot(df['BPS29'],label='BPS 29', color='steelblue', linewidth=4)
plt.plot(df['BPS49'],label='BPS 49', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs', fontsize=14)
plt.title('Crack Thermo couple Signature for Bad Case slabs', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[141]:


df = pd.read_csv('D:\\Caster\\Year 2022\\LFC Project\\FY 22 23\\LFC Model\\Caster 2 Bad case\\LFC Good Case\\15TH Good Case C2.csv')
#data.sample(n=5)
df


# In[143]:


plt.plot(df['BPS6'],label='BPS 6', color='green')
plt.plot(df['BPS26'],label='BPS 26', color='steelblue', linewidth=4)
plt.plot(df['BPS46'],label='BPS 46', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Bad Slabs C2', fontsize=14)
plt.title('Crack Thermo couple Signature for Good Case slabs C2', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[145]:


plt.plot(df['BPS3'],label='BPS 3', color='green')
plt.plot(df['BPS23'],label='BPS 23', color='steelblue', linewidth=4)
plt.plot(df['BPS43'],label='BPS 43', color='purple', linestyle='dashed')
plt.legend()
#display plot
plt.ylabel('Thermo Couple Reading', fontsize=14)
plt.xlabel('Time instance of Good Slabs C2', fontsize=14)
plt.title('Crack Thermo couple Signature for Good Case slabs C2', fontsize=16)
fig = plt.figure(figsize=(80, 60))
plt.show()


# In[ ]:




