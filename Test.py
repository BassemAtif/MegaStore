# %%
import pickle
from joblib import load
from UTILS import *
from Split import test_df as data
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures



# %%
with open('preprocessing_option1.pkl', 'rb') as f:
    preprocess_dict = pickle.load(f)

# Use the loaded dictionary to transform names in new data
data["state_mean"] = data["State"].map(preprocess_dict["state_dict"])
data["sub_cat_encoded"] = data["Sub Category"].map(preprocess_dict["sub_cat_dict"])

# %%
final_cat_df=one_hot_encode_columns(["Main Category","Region"],data)  #Categorical features


# %%
final_df_num=data[preprocess_dict["num_columns"]]


# %%
"""
<h2>Loading Model</h2>
"""

# %%
model=load("rf_option1.joblib")

# %%
final_df=pd.concat([final_df_num,final_cat_df],axis=1)
final_df_norm=pd.DataFrame(model["scaler"].transform(final_df),columns=final_df.columns)

# %%
X=final_df_norm.drop(columns=["Profit"])
y=final_df_norm["Profit"]

# %%
"""
<h3>Random Forest</h2>
"""

# %%
y_pred = model["model"]["rf"].predict(X)
r2_op1 = r2_score(y, y_pred)
mse_op1 = mean_squared_error(y, y_pred)

print("R^2 Score: ", r2_op1)
print("Mean Squared Error (MSE): ", mse_op1)


# %%
"""
<h3>Polynomial Regression</h3>
"""

# %%
x_poly=PolynomialFeatures(degree=model["model"]["plreg"][1]).fit_transform(X)
y_pred = model["model"]["plreg"][0].predict(x_poly)
r2_op1 = r2_score(y, y_pred)
mse_op1 = mean_squared_error(y, y_pred)

print("R^2 Score: ", r2_op1)
print("Mean Squared Error (MSE): ", mse_op1)


# %%
"""
<h3>Gradient Boosting Regressor</h3>
"""

# %%
y_pred = model["model"]["gbr"].predict(X)
r2_op1 = r2_score(y, y_pred)
mse_op1 = mean_squared_error(y, y_pred)

print("R^2 Score: ", r2_op1)
print("Mean Squared Error (MSE): ", mse_op1)
