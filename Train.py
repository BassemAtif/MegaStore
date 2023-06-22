# %%
from Split import train_df as data
from UTILS import *
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor 
import seaborn as sns
import pickle
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# %%
mapping_dict=transform_ordinal_means(data,"State","Profit")
data['state_mean'] = data['State'].map(mapping_dict)

# %%
sub_cat_dict=transform_ordinal(data,"Sub Category","Profit")
data["sub_cat_encoded"]=data["Sub Category"].map(sub_cat_dict)

# %%
anova_test(data,["State","Customer Name","Region","Order quarter","Main Category","Sub Category","Product ID"],"Profit")

# %%
sns.heatmap(filter_numerical(data).corr().round(2), annot=True, cmap='crest', center=0)

# %%
final_df_num=feature_select_numerical(data,"Profit")

# %%
final_cat_df=one_hot_encode_columns(["Main Category","Region"],data)  #Categorical features

# %%
final_df=pd.concat([final_df_num,final_cat_df],axis=1)

# %%
final_df_norm,scaler=normalize_feature(final_df)

# %%
"""
<h2>Model Training</h2>

"""

# %%
X=final_df_norm.drop(columns=["Profit"])
y=final_df_norm["Profit"]

# %%
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,y,test_size=0.2,random_state=42)

# %%
from sklearn.metrics import r2_score, mean_squared_error
rf1 = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training data
rf1.fit(X_train1, Y_train1)

# Make predictions on the testing data
y_pred1 = rf1.predict(X_test1)

# Calculate R^2 score and Mean Squared Error (MSE) for the model
r2_op1 = r2_score(Y_test1, y_pred1)
mse_op1 = mean_squared_error(Y_test1, y_pred1)

print("R^2 Score: ", r2_op1)
print("Mean Squared Error (MSE): ", mse_op1)

# %%
"""
<h3>Random Forest</h2>
"""

# %%
rf=RandomForestRegressor(n_estimators=90,random_state=42)
# Train the model using the training data
rf.fit(X, y)

# %%
"""
<h3>Polynomial using Cross-Validation</h3>
"""

# %%
pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())

param_grid = {'polynomialfeatures__degree': [1, 2, 3, 4]}  # Try different polynomial degrees

# Use GridSearchCV to find the best polynomial order
grid = GridSearchCV(pipeline,scoring="r2" ,param_grid=param_grid, cv=5)  # 5-fold cross-validation
grid.fit(X, y)

best_polynomial_order = grid.best_params_['polynomialfeatures__degree']
best_r2_score_poly=grid.best_score_
# Print the best polynomial order and the R-squared score
print("Best Polynomial Order:", best_polynomial_order)
print("R-squared Score:", best_r2_score_poly)

# %%
poly = PolynomialFeatures(degree=best_polynomial_order)
x_poly = poly.fit_transform(X)

# Create a linear regression model and fit the data
plreg = LinearRegression()
plreg.fit(x_poly, y)

# %%
"""
<h3>Gradient Boosting Regressor</h3>
"""

# %%
param_grid = {'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 7]}
gbr = GradientBoostingRegressor()
grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)
print("Best hyperparameters: ", grid_search.best_params_)

# %%
gbr=GradientBoostingRegressor(**(grid_search.best_params_))
gbr.fit(X,y)

# %%
"""
<h2>Saving the Model</h2>
"""

# %%
model_data={"model":{"rf":rf,"plreg":(plreg,best_polynomial_order),"gbr":gbr},"scaler":scaler}
dump(model_data,"rf_option1.joblib")

# %%
preprocessing={"state_dict":mapping_dict,"sub_cat_dict":sub_cat_dict,"num_columns":list(final_df_num.columns)}
dump(preprocessing,"preprocessing_option1.pkl")