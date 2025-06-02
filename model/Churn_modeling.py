import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# Author: Maria Jose Teran

## Load dataset
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
df_val = pd.read_csv("data/validation.csv")

## Data Cleaning
df_train.loc[(df_train['Internet Service'] == 0) & (df_train['Internet Type'].isna()), 'Internet Type'] = 'None'
df_test.loc[(df_test['Internet Service'] == 0) & (df_test['Internet Type'].isna()), 'Internet Type'] = 'None'
df_val.loc[(df_val['Internet Service'] == 0) & (df_val['Internet Type'].isna()), 'Internet Type'] = 'None'
df_train['Offer'] = df_train['Offer'].fillna('None')
df_test['Offer'] = df_test['Offer'].fillna('None')
df_val['Offer'] = df_val['Offer'].fillna('None')

# Fill nan 'Churn Category', 'Churn Reason' with NA for EDA
df_train[['Churn Category', 'Churn Reason']] = df_train[['Churn Category', 'Churn Reason']].fillna('NA')
df_test[['Churn Category', 'Churn Reason']] = df_test[['Churn Category', 'Churn Reason']].fillna('NA')
df_val[['Churn Category', 'Churn Reason']] = df_val[['Churn Category', 'Churn Reason']].fillna('NA')
df_EDA = pd.concat([df_train, df_test, df_val], ignore_index=True)
df_EDA.to_csv("data/df_EDA.csv", index=False)

## Feature Engineering
df_train = df_train.set_index('Customer ID')
df_test = df_test.set_index('Customer ID')
df_val = df_val.set_index('Customer ID')

# Save means for predicting
refunds_by_offer = df_train.groupby('Offer')['Total Refunds'].mean().to_dict()
avg_gb_means = df_train.groupby(['Unlimited Data', 'Internet Type', pd.cut(df_train['Age'], bins=[18, 30, 45, 60, 100])])['Avg Monthly GB Download'].mean().to_dict()
extra_data_charges_means = df_train.groupby(['Unlimited Data', 'Internet Type', pd.cut(df_train['Age'], bins=[18, 30, 45, 60, 100])])['Total Extra Data Charges'].mean().to_dict()
long_dist_means = df_train.groupby(['Phone Service', 'Multiple Lines'])['Avg Monthly Long Distance Charges'].mean().to_dict()
total_long_dist_means = df_train.groupby(['Phone Service', 'Multiple Lines'])['Total Long Distance Charges'].mean().to_dict()

means_dict = {
    "mean_refunds_by_offer": refunds_by_offer,
    "mean_gb_download": avg_gb_means,
    "mean_extra_data": extra_data_charges_means,
    "mean_long_distance": long_dist_means,
    "mean_total_long_distance": total_long_dist_means}
joblib.dump(means_dict, "model/means_churn_inputs.pkl")

# Separate features and target
drop_cols = ['CLTV', 'Churn', 'Churn Score', 'Churn Reason', 'Churn Category', 'Customer Status', 'Quarter', 'State', 'Country', 'Zip Code', 'Lat Long', 'Partner']
target = 'Churn'
X_train = df_train.drop(columns=drop_cols)
Y_train = df_train[target]
X_test = df_test.drop(columns=drop_cols)
Y_test = df_test[target]
X_val = df_val.drop(columns=drop_cols)
Y_val = df_val[target]

# Encoding
# Frequency encoder for City
city_freq = X_train['City'].value_counts(normalize=True).to_dict()
default_freq = min(city_freq.values())
for df in [X_train, X_test, X_val]:
    df['City_FE'] = df['City'].map(city_freq).fillna(default_freq)
    df.drop(columns='City', inplace=True)
joblib.dump(city_freq, "model/city_freq_dict.pkl") # Save dict

# Ordinal encode
encoder = OrdinalEncoder(categories=[['Month-to-Month', 'One Year', 'Two Year']])
X_train['Contract'] = encoder.fit_transform(X_train[['Contract']])
X_test['Contract'] = encoder.transform(X_test[['Contract']])
X_val['Contract'] = encoder.transform(X_val[['Contract']])
X_train.drop(columns='Contract', inplace=True)
X_test.drop(columns='Contract', inplace=True)
X_val.drop(columns='Contract', inplace=True)

# Binary
gender_map = {'Male': 1, 'Female': 0}
X_train['Gender'] = X_train['Gender'].map(gender_map)
X_test['Gender'] = X_test['Gender'].map(gender_map)
X_val['Gender'] = X_val['Gender'].map(gender_map)

# One-hot encoding
one_hot_cols = ['Internet Type', 'Offer', 'Payment Method']
X_train = pd.get_dummies(X_train, columns=one_hot_cols, drop_first=False)
X_test = pd.get_dummies(X_test, columns=one_hot_cols, drop_first=False)
X_val = pd.get_dummies(X_val, columns=one_hot_cols, drop_first=False)
for df in [X_train, X_test, X_val]:
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

# Add GeoCluster with KMeans
geo_features = ['Latitude', 'Longitude', 'Population']
scaler_geo = StandardScaler()
X_geo_scaled = scaler_geo.fit_transform(X_train[geo_features])
kmeans = KMeans(n_clusters=10, random_state=42)
X_train['GeoCluster'] = kmeans.fit_predict(X_geo_scaled)
X_test['GeoCluster'] = kmeans.predict(scaler_geo.transform(X_test[geo_features]))
X_val['GeoCluster'] = kmeans.predict(scaler_geo.transform(X_val[geo_features]))
X_train.drop(columns=geo_features, inplace=True)
X_test.drop(columns=geo_features, inplace=True)
X_val.drop(columns=geo_features, inplace=True)

# Save GeoCluster dict
geo_dict = df_train[['City']].copy()
geo_dict['GeoCluster'] = X_train['GeoCluster'].values
city_to_cluster = geo_dict.groupby('City')['GeoCluster'].agg(lambda x: x.mode()[0]).to_dict()
joblib.dump(city_to_cluster, "model/city_to_cluster.pkl")

# Drop features with multi collinearity (VIF = inf)
X_train.drop(['Total Charges', 'Internet Service', 'Payment Method_Mailed Check'], axis=1, inplace=True)
X_test.drop(['Total Charges', 'Internet Service', 'Payment Method_Mailed Check'], axis=1, inplace=True)
X_val.drop(['Total Charges', 'Internet Service', 'Payment Method_Mailed Check'], axis=1, inplace=True)

# Feature Scaling
columns = X_train.columns
rob_scaler = RobustScaler()
X_train_scaled = rob_scaler.fit_transform(X_train)
X_test_scaled = rob_scaler.transform(X_test)
X_val_scaled = rob_scaler.transform(X_val)
X_train = pd.DataFrame(X_train_scaled, columns=columns)
X_test = pd.DataFrame(X_test_scaled, columns=columns)
X_val = pd.DataFrame(X_val_scaled, columns=columns)
joblib.dump(rob_scaler, "model/rob_scaler.pkl")

## Modeling - Logistic Regression
best_lr = LogisticRegression(C=100, max_iter=1500, penalty='l1', random_state=42, solver='liblinear')
best_lr.fit(X_train, Y_train)
Y_pred_lr = best_lr.predict(X_test)
diff_df = pd.DataFrame({'Actual' : Y_test, 'Predicted': Y_pred_lr})
acc_lr_test = accuracy_score(Y_test, Y_pred_lr)

# AUC-ROC curve:
Y_probs_lr = best_lr.predict_proba(X_test)
Y_probs_lr = Y_probs_lr[:, 1] # only considering positive class
roc_auc = roc_auc_score(Y_test, Y_probs_lr)
fpr_lr, tpr_lr, _ = roc_curve(Y_test, Y_probs_lr)
auc_lr = auc(fpr_lr, tpr_lr)


# Predict on Validation Set
Y_val_pred_lr = best_lr.predict(X_val)
acc_lr_val = accuracy_score(Y_val, Y_val_pred_lr)
print("Final accuracy:", round(acc_lr_val,4))

target_names = ['Stayed', 'Churned']
report = classification_report(Y_val, Y_val_pred_lr, target_names=target_names)
print(report)

# Save model
best_lr.feature_names_in_ = X_train.columns
joblib.dump(best_lr, "model/churn_prediction_lr.pkl")
