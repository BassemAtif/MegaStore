import copy
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder

def anova_test(df, categorical_columns, target_column):
    
    p_values = {}  
    
    for categorical_column in categorical_columns:
        categories = df[categorical_column].unique()
        
        data = []
        
        for category in categories:
            data.append(df[df[categorical_column] == category][target_column])
        
        statistic, p_value = f_oneway(*data)
        
        p_values[categorical_column] = p_value
    
    return p_values

def filter_categorial(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    categorical_df = df[categorical_columns]

    return categorical_df

def filter_numerical(df):
    numerical_columns = df.select_dtypes(exclude=['object', 'category']).columns
    numerical_df = df[numerical_columns]

    return numerical_df

def transform_ordinal_means(df,column,target):  
    return df.groupby(column)[target].mean().to_dict()


def transform_ordinal(df,column,target):
    sub_dict=df.groupby(column)[target].mean().sort_values()
    sub_dict={k:v for v,k in enumerate(sub_dict.keys())}
    return sub_dict     #df[new_col]=df[column].map(transform_ordinal(df,column,target_col"Profit")) To apply mapping on the dataframe in a new column

def feature_select_numerical(df,target_col):
    
    corr=filter_numerical(df).corr().round(2)
    final_df=pd.DataFrame()
    for column in corr.columns:
        if(abs(corr[column][target_col])>=0.05):
            final_df[column]=df[column]
    return final_df

def one_hot_encode_columns(columns, df):
    encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
    encoded_columns = encoder.fit_transform(df[columns])
    feature_names = encoder.get_feature_names_out(columns)
    df_encoded = pd.DataFrame(encoded_columns, columns=feature_names, index=df.index)
    return df_encoded,encoder

def normalize_feature(df):
    copy_df=copy.deepcopy(df)
    scaler=MinMaxScaler()
    scaler.fit(copy_df)
    return pd.DataFrame(scaler.transform(copy_df),columns=df.columns,index=df.index),scaler

def remove_outliers(df, columns, threshold=3):
    
    df_no_outliers = df.copy()
    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > threshold]
        df_no_outliers = df_no_outliers.drop(outliers.index)
    
    return df_no_outliers
    
def fill_missing(df,arr):
    x = 0 # index for column cuz 'col' is String
    for col in range(len(arr)): # for all columns
        for index, row in df.iterrows(): # and all rows
            if pd.isnull(row[col]) or row[col] == '': # if value is null
                # print(x)
                # print("missing value is ", arr[x])
                df.loc[index, col] = arr[x]
        x = x+1
    return df
