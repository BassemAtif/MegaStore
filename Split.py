import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv("megastore-regression-dataset.csv")
#Extract datetime
data['Order Date']=pd.to_datetime(data['Order Date'])
data['Order year']=data['Order Date'].dt.year
data['Order quarter']=data['Order Date'].dt.quarter
data = data.drop('Order Date', axis=1)
data['Ship Date']=pd.to_datetime(data['Ship Date'])
data['Ship year']=data['Ship Date'].dt.year
data['Ship quarter']=data['Ship Date'].dt.quarter
data = data.drop('Ship Date', axis=1)
#Extract Categories
data[["Main Category","Sub Category"]]=data["CategoryTree"].str.extract("'MainCategory': '(?P<Main_Category>[^']*)', 'SubCategory': '(?P<Sub_Category>[^']*)'")
data.drop(columns=['CategoryTree'],inplace=True)
#Drop Features
data.drop(columns=["Row ID","Order ID","Customer ID","City","Postal Code","Ship year","Ship quarter","Product Name","Country","Segment"],inplace=True)

# Split into features and target
X = data.drop(columns=['Profit'])
y = data['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Combine X_train and y_train into a single dataframe
train_df = pd.concat([X_train, y_train], axis=1)

# Combine X_test and y_test into a single dataframe
test_df = pd.concat([X_test, y_test], axis=1)
