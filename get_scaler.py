
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder


rmse = 25

df_list = pd.read_csv("Resources/cleandata/clean_listings.csv")

df_cal = pd.read_csv("Resources/cleandata/clean_calendar.csv")

df_cal_g = df_cal.groupby(['listing_id','date']).mean()

df_cal_g = df_cal_g.reset_index(level=['date'])

df = df_list.merge(df_cal_g, how='inner', on='listing_id')



df.rename(columns={"price_y":'price','date':'month'}, inplace=True)



df.drop(columns=['state','listing_id','price_x'], inplace=True)



df['bedrooms']=df['bedrooms'].apply(np.sqrt)


df['bathrooms'] = df['bathrooms'].apply(lambda x: pow(x,1/3))




df['accommodates']=df['accommodates'].apply(np.log10)

enc = OneHotEncoder(sparse=False)
encode_df = pd.DataFrame(enc.fit_transform(df[['city','zipcode','month']]))
encode_df.columns = enc.get_feature_names(['city','zipcode','month'])


df = df.merge(encode_df, left_index=True, right_index=True)
df.drop(columns=['city','zipcode','month'], inplace=True)

df.head()

X = df.drop(columns=['price'])
y = df['price']


print(X.shape)
print(y.shape)


X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)





