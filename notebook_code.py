# CELL 1
# %pip install sklearn
# %pip install pandas
# %pip install numpy
# %pip install matplotlib
# %pip install seaborn
%pip install cleanlab
%pip install xgboost
%pip install ipywidgets
%pip install imbalanced-learn
%pip install statsmodels
%pip install shap
%pip install prophet
%pip install plotly
%pip install tensorflow
%pip install scipy
%pip install statsforecast
%pip install utilsforecast


# CELL 3
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cleanlab.filter import find_label_issues
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import cm

from sklearn.cluster import KMeans
from sklearn.decomposition import (
    PCA,
    TruncatedSVD
)
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import (
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_predict,
    train_test_split
)
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
    Binarizer,
    FunctionTransformer,
    OneHotEncoder
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stl.mstl import MSTL
import shap

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (SimpleRNN, LSTM, Conv1D,
                                     GlobalMaxPooling1D, Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

from statsforecast.models import (
    AutoARIMA,
    Naive,
    HistoricAverage,
    WindowAverage,
    SeasonalNaive,
)
from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import *


from keras import Input
from utilsforecast.plotting import plot_series


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# CELL 6
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# CELL 7
def sklearnEvaluate(model, X, y, cv, model_prop=None, model_step=None):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_mean_squared_error","neg_root_mean_squared_error"],
        return_estimator=model_prop is not None,
    )
    if model_prop is not None:
        if model_step is not None:
            values = [
                getattr(m[model_step], model_prop) for m in cv_results["estimator"]
            ]
        else:
            values = [getattr(m, model_prop) for m in cv_results["estimator"]]
        print(f"Mean model.{model_prop} = {np.mean(values)}")
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )

# CELL 8
import matplotlib.pyplot as plt
import numpy as np

def plot_tf_style_predictions(X_3D, y_3D, model, feature_index=0, num_plots=3):
    """
    Creates a stacked plot exactly like the TensorFlow Time Series tutorial.

    Parameters:
    X_3D: Your 3D testing features (Samples, Time_Steps, Features)
    y_3D: Your testing labels
    model: Your trained LSTM/CNN model
    feature_index: Which feature to draw the blue line for (e.g., 0 for PM2.5)
    num_plots: How many subplots to stack (the image has 3)
    """
    # 1. Get the model's predictions for the test set
    # Using > 0.5 to convert the probability back into a 1 or 0
    predictions = (model.predict(X_3D) > 0.5).astype(int)

    # 2. Set up the figure with multiple subplots stacked vertically
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 8))

    # 3. Loop through to draw a few random examples from your test set
    # (We multiply by 100 just to space out the examples we pick)
    for i, ax in enumerate(axes):
        sample_idx = i * 100

        # Extract the sequence of inputs for the blue line
        inputs = X_3D[sample_idx, :, feature_index]
        input_time_steps = list(range(len(inputs)))

        # The label/prediction happens at the time step immediately after the inputs
        target_time = len(inputs)
        label = y_3D[sample_idx]
        prediction = predictions[sample_idx][0]

        # --- DRAWING THE TF STYLE PLOT ---

        # Plot 1: The Inputs (Blue line with dots)
        ax.plot(input_time_steps, inputs, label='Inputs', marker='.', zorder=-10)

        # Plot 2: The True Label (Green Circle with black edge)
        ax.scatter(target_time, label, edgecolors='k', label='Labels',
                   c='#2ca02c', s=64, zorder=2)

        # Plot 3: The Prediction (Orange X with black edge)
        ax.scatter(target_time, prediction, marker='X', edgecolors='k',
                   label='Predictions', c='#ff7f0e', s=64, zorder=3)

        # Formatting to match the tutorial
        ax.set_ylabel('Scaled Feature')
        if i == 0:
            ax.legend() # Only put the legend on the top plot

    # Add the x-axis label only to the very bottom plot
    axes[-1].set_xlabel('Time [h]')

    plt.tight_layout()
    plt.show()

# How to use it:
# (Assuming your scaled PM2.5 is at index 0 of your features)
# plot_tf_style_predictions(X_test_3D, y_test_3D, lstm_model, feature_index=0, num_plots=3)

# CELL 9
def max_missing_sequence(ser):
    # 1. Mask for null values
    is_null = ser.isnull()

    # 2. Group consecutive booleans
    # (True -> False or False -> True increments the group)
    groups = (is_null != is_null.shift()).astype(int).cumsum()

    # 3. Filter to only look at the 'True' (null) groups and count them
    null_counts = is_null[is_null].groupby(groups).size()

    # 4. Return the max, or 0 if no nulls exist
    return null_counts.max() if not null_counts.empty else 0


# CELL 11
!wget https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip

# CELL 12
!unzip beijing+multi+site+air+quality+data.zip

# CELL 13
!unzip PRSA2017_Data_20130301-20170228.zip

# CELL 15
stations_to_keep = [
  'Aotizhongxin',
  'Dongsi',
  # 'Guanyuan',
  # 'Nongzhanguan',
  # 'Tiantan',
  # 'Wanliu',
  # 'Wanshouxigong'
]

# CELL 16
# Find all files that start with "PRSA_Data" and end with ".csv"
# file_paths = glob.glob("./beijing+multi+site+air+quality+data/PRSA_Data_20130301-20170228/PRSA_Data_*Nongzhanguan*.csv")
file_paths = glob.glob("./PRSA_Data_20130301-20170228/PRSA_Data_*.csv")

# Read all into a list
dataframes = [pd.read_csv(file) for file in file_paths if any(sub in file for sub in stations_to_keep)]

# Combine them all together into one giant dataset
df = pd.concat(dataframes, ignore_index=True)

df['wd'] = df['wd'].astype('category')
df['station'] = df['station'].astype('category')


df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df['DayOfWeek'] = df['datetime'].dt.dayofweek

# Create a binary feature: 1 if it is Saturday (5) or Sunday (6), 0 if weekday
df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

# Create a binary feature for "Rush Hour" (e.g., 7 AM to 9 AM, and 5 PM to 7 PM)
df['Is_Rush_Hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

date_time = df['datetime']

# Sort by station FIRST, then by time.
df = df.sort_values(by=['station', 'datetime']).reset_index(drop=True)
# df = df.drop('No',axis=1)

# CELL 18
FEATURE_COLS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd']
CHEM_COLUMNS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
categorical_columns = ['wd'] #df.columns[df.dtypes == "category"]
NUMERIC_columns = df.columns[df.dtypes == "numeric"]

datetime_columns = ['year', 'month', 'day', 'hour', 'datetime']

# CELL 19
categorical_columns

# CELL 21
KNNImputerPipe = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
])

# CELL 22
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# CELL 23
cyclicTransformer = ColumnTransformer(
    transformers=[
        ("categorical", one_hot_encoder, categorical_columns),
        ("month_sin", sin_transformer(12), ["month"]),
        ("month_cos", cos_transformer(12), ["month"]),
        ("weekday_sin", sin_transformer(7), ["DayOfWeek"]),
        ("weekday_cos", cos_transformer(7), ["DayOfWeek"]),
        ("hour_sin", sin_transformer(24), ["hour"]),
        ("hour_cos", cos_transformer(24), ["hour"]),
    ],
    remainder='passthrough',
)

cyclicTransformer.set_output(transform="pandas")

# CELL 24
# data_pipeline = ColumnTransformer([
#     ('numerical', num_pipeline, CHEM_COLUMNS),
#     ('categorical', OneHotEncoder(), cat_vars),

# ])

# CELL 27
df.info()

# CELL 28
# 1. Pivot for all target columns
# Columns will now be (variable, station) - e.g., ('temp', 'Station_1')
multi_pivot = df.pivot(index='datetime', columns='station', values=CHEM_COLUMNS)

# 2. Calculate the massive correlation matrix
multi_corr = multi_pivot.corr()

# 3. View a specific subset
# e.g., How does 'humidity' at Station A correlate with 'humidity' at other stations?
humidity_corr = multi_corr.loc['PM2.5', 'PM2.5']
sns.heatmap(humidity_corr, annot=True, cmap='YlGnBu')

# CELL 30
df.isnull().sum().sort_values(ascending=False)

# CELL 31
longest_gaps = df.apply(max_missing_sequence)
longest_gaps.sort_values(ascending=False)

# CELL 32
station_reliability = df.isnull().sum(axis=1).groupby(df['station']).sum().sort_values()

print(station_reliability)

# CELL 33


# CELL 34
# Calculate the percentage of missing values per station
missing_pct = df.groupby('station').apply(lambda x: x.isnull().mean().mean())

# Get the station with the minimum percentage
best_station_pct = missing_pct.idxmin()

print(f"The most reliable station (by percentage) is: {best_station_pct}")

# CELL 35
# df_with_nans = df[df.isnull().any(axis=1)]
# df_with_nans

# CELL 40
df.isnull().sum().sort_values(ascending=False)

# CELL 41
df.columns.to_list()

# CELL 42
df.info()

# CELL 43
df.describe().transpose()


# CELL 45
features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3','TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
df_single_station = df # [df['station'] == 'Aotizhongxin'].copy()
df_daily = df_single_station.set_index("datetime")
df_daily = df_daily[features].resample('H').mean().dropna()

# CELL 46
_ = df_daily.plot(subplots=True, figsize=(10, 20))

# CELL 47
_ = df_daily[:164].plot(subplots=True, figsize=(10, 20))

# CELL 49
scaler = StandardScaler()
df_daily_scaled  = pd.DataFrame(scaler.fit_transform(df_daily), columns=df_daily.columns.to_list())
df_daily_scaled.plot.kde(subplots=True, figsize=(10, 15))

# CELL 50
df_daily_scaled.plot.hist(subplots=True, figsize=(10, 20), bins=100)

# CELL 51
pd.plotting.scatter_matrix(df_daily_scaled, alpha=0.2, figsize=(20, 20))

# CELL 54
numeric_cols = [
    'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'month',
    'DayOfWeek', 'Is_Weekend', 'Is_Rush_Hour'
]

corr_matrix = df[numeric_cols].corr(method='pearson')

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax
)
ax.set_title('Feature Pearson Correlation Matrix', fontsize=14, pad=12)
plt.tight_layout()
plt.show()

# CELL 56
corr_matrix = df[numeric_cols].corr(method='spearman')

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax
)
ax.set_title('Feature Spearman Correlation Matrix', fontsize=14, pad=12)
plt.tight_layout()
plt.show()

# CELL 58
df_single_station = df.copy()
df_single_station.set_index("datetime", inplace=True)
df_daily = df_single_station['PM2.5'].resample('D').mean().dropna()

# Decompose with both monthly (30-day) and yearly (365-day) seasonal components
decomposition = MSTL(df_daily, periods=(7, 30, 365)).fit()

fig = decomposition.plot()
fig.set_size_inches(12, 10)
plt.tight_layout()
plt.show()

# CELL 59
df_single_station = df.copy()
df_single_station.set_index("datetime", inplace=True)
df_daily = df_single_station['PM2.5'][:1344]

# Decompose with both monthly (30-day) and yearly (365-day) seasonal components
decomposition = MSTL(df_single_station['PM2.5'][:1344], periods=(24)).fit()

fig = decomposition.plot()
fig.set_size_inches(12, 10)
plt.tight_layout()
plt.show()

# CELL 63
# Quick report of remaining missing values in your target columns
print(df[CHEM_COLUMNS].isnull().sum().sort_values(ascending=False))

# CELL 64
longest_gaps = df.apply(max_missing_sequence)
longest_gaps.sort_values(ascending=False)

# CELL 65
# Run this BEFORE you start the loop
# for col in FEATURE_COLS:
    # df[f'{col}_is_imputed'] = df[col].isnull().astype(int)

# CELL 66
for col in CHEM_COLUMNS:
    # Pass 1: Spatial Fill (Neighboring stations on same day)
    # This fills the majority of gaps based on regional conditions
    df[col] = df.groupby('datetime')[col].transform(lambda x: x.fillna(x.mean()))



    # Pass 2: Temporal Fill (Fallback for "Blackout Days")
    # If the whole region was missing data, we interpolate across time for that station
    # df[col] = df.groupby('station')[col].transform(
    #     lambda x: x.interpolate(method='linear', limit_direction='both')
    # )

df['wd'] = df.groupby('datetime')['wd'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

df['wd'] = df.groupby('station')['wd'].transform(
    lambda x: x.ffill(limit=5)
)

print("Imputation complete.")

# CELL 67
# Quick report of remaining missing values in your target columns
print(df[CHEM_COLUMNS].isnull().sum().sort_values(ascending=False))

# CELL 68
longest_gaps = df[CHEM_COLUMNS].apply(max_missing_sequence)
longest_gaps.sort_values(ascending=False)

# CELL 69
station_reliability = df[['CO', 'station']].isnull().sum(axis=1).groupby(df['station']).sum().sort_values()

print(station_reliability)

# CELL 70
# df[df['station'] == 'Nongzhanguan'].to_csv("PRSA_Data_20130301-20170228_Nongzhanguan_cleaned.csv")

# CELL 71
station_reliability = df.isnull().sum(axis=1).groupby(df['station']).sum().sort_values()

print(station_reliability)

# CELL 72
# 1. Pivot for all target columns
# Columns will now be (variable, station) - e.g., ('temp', 'Station_1')
multi_pivot = df.pivot(index='datetime', columns='station', values=CHEM_COLUMNS)

# 2. Calculate the massive correlation matrix
multi_corr = multi_pivot.corr()

# 3. View a specific subset
# e.g., How does 'humidity' at Station A correlate with 'humidity' at other stations?
humidity_corr = multi_corr.loc['PM2.5', 'PM2.5']
sns.heatmap(humidity_corr, annot=True, cmap='YlGnBu')

# CELL 79
samples = df.shape[0]

# CELL 80
df.select_dtypes(include=np.number).columns

# CELL 81
# num_train_samples = int(0.6 * samples)
# num_val_samples = int(0.2 * samples)
# num_test_samples = samples - num_train_samples - num_val_samples
# print("num_train_samples:", num_train_samples)
# print("num_val_samples:", num_val_samples)
# print("num_test_samples:", num_test_samples)

# CELL 82


# CELL 83
# train_df = df[:num_train_samples]

# CELL 84
# val_df = df[num_train_samples:num_train_samples + num_val_samples]
# test_df = df[num_train_samples + num_val_samples:]

# CELL 85
df = df.sort_values(['station', 'datetime'])

train_list = []
val_list = []
test_list = []

# 2. Iterate through each station and split
for station_id, group in df.groupby('station'):
    n = len(group)

    # Calculate split points
    train_end = int(n * 0.70)
    val_end = int(n * 0.85) # 70% + 15%

    # Slice the group
    train_list.append(group.iloc[:train_end])
    val_list.append(group.iloc[train_end:val_end])
    test_list.append(group.iloc[val_end:])

# 3. Concatenate back into single DataFrames
train_df = pd.concat(train_list)
val_df = pd.concat(val_list)
test_df = pd.concat(test_list)

print(f"Train size: {len(train_df)}")
print(f"Val size:   {len(val_df)}")
print(f"Test size:  {len(test_df)}")

# CELL 86


# CELL 89
def impute_knn(df, k=5):
    num = df.select_dtypes(include=np.number).columns
    out = df.copy()
    out[num] = KNNImputer(n_neighbors=k).fit_transform(df[num])
    return out

def impute_linear(df):
    num = df.select_dtypes(include=np.number).columns
    out = df.copy()
    out[num] = out[num].interpolate(method='linear', limit_direction='both')
    return out

def benchmark_imputation(df, col='PM2.5', gap_list=(2, 6, 24, 50), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in gap_list:
        s = rng.integers(100, len(df)-g-100)
        true_vals = df[col].iloc[s:s+g].copy()
        df_gap = df.copy()
        df_gap[col].iloc[s:s+g] = np.nan
        knn_err    = mean_absolute_error(true_vals, impute_knn(df_gap)[col].iloc[s:s+g])
        linear_err = mean_absolute_error(true_vals, impute_linear(df_gap)[col].iloc[s:s+g])
        rows.append({'Gap (h)': g, 'KNN MAE': knn_err, 'Linear MAE': linear_err})
    return pd.DataFrame(rows).set_index('Gap (h)')

bench = benchmark_imputation(train_df[CHEM_COLUMNS].dropna())
print(bench.round(3))

fig, ax = plt.subplots(figsize=(8, 4))
bench.plot.bar(ax=ax, color=['steelblue','coral'], edgecolor='white')
ax.set_title('Imputation MAE: KNN vs Linear Interpolation')
ax.set_xlabel('Gap length (hours)'); ax.set_ylabel('MAE (µg/m³)')
plt.tight_layout(); plt.show()



# CELL 90
df_imputed = train_df

knn_imputer = KNNImputer(n_neighbors=5)

df_imputed[CHEM_COLUMNS] = knn_imputer.fit_transform(df_imputed[CHEM_COLUMNS])
print(f"\nMissing after KNN imputation: {df_imputed.isnull().sum().sum()}")

# CELL 91
# Quick report of remaining missing values in your target columns
print(df_imputed[FEATURE_COLS].isnull().sum().sort_values(ascending=False))

# CELL 92
longest_gaps = df_imputed.apply(max_missing_sequence)
longest_gaps.sort_values(ascending=False)

# CELL 93
train_df['wd'] = train_df.groupby('station')['wd'].ffill(limit=5)

# CELL 94
print(df_imputed[FEATURE_COLS].isnull().sum().sort_values(ascending=False))

# CELL 95
# Impute missing values in validation and test sets using the KNN logic

val_df[CHEM_COLUMNS] = knn_imputer.transform(val_df[CHEM_COLUMNS])
test_df[CHEM_COLUMNS] = knn_imputer.transform(test_df[CHEM_COLUMNS])

print(f"NaNs remaining in val_df: {val_df.isnull().sum().sum()}")
print(f"NaNs remaining in test_df: {test_df.isnull().sum().sum()}")

# CELL 96
# Quick report of remaining missing values in your target columns
print("Validation Dataset")
print(val_df.isnull().sum().sort_values(ascending=False))

print("Testing Dataset")
print(test_df.isnull().sum().sort_values(ascending=False))

# CELL 97
longest_gaps = test_df.apply(max_missing_sequence)
longest_gaps.sort_values(ascending=False)

# CELL 98
test_df['wd'] = test_df.groupby('station')['wd'].ffill(limit=5)

# CELL 99
print(f"NaNs remaining in test_df: {test_df.isnull().sum().sum()}")

# CELL 102
# def remove_3sigma(df, lower=0):
#     num = df.select_dtypes(include=np.number).columns
#     out = df.copy(); n = 0
#     for col in num:
#         mu, s = out[col].mean(), out[col].std()
#         hi, lo = mu + 3*s, max(mu - 3*s, lower)
#         bad = (out[col] < lo) | (out[col] > hi)
#         out.loc[bad, col] = np.nan; n += bad.sum()
#     print(f"Flagged {n} extreme values; re-imputing...")
#     return impute_linear(out)

# NUM_COLS = df_imputed.select_dtypes(include=np.number).columns.tolist()
# df_clean = remove_3sigma(df_imputed)

# fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
# df['PM2.5'][:num_train_samples].plot(ax=axes[0], lw=0.5, color='steelblue', title='PM2.5 — Raw')
# df_clean['PM2.5'].plot(ax=axes[1], lw=0.5, color='darkorange', title='PM2.5 — After 3σ Filtering')
# for ax in axes: ax.set_ylabel('µg/m³')
# plt.tight_layout(); plt.show()



# CELL 103
df_imputed[CHEM_COLUMNS].describe().transpose()

# CELL 104
# df_clean[CHEM_COLUMNS].describe().transpose()

# CELL 108
from scipy.stats.mstats import winsorize

s_train = train_df[train_df['station'] == stations_to_keep[0]]
s_val   = val_df[val_df['station'] == stations_to_keep[0]]
s_test  = test_df[test_df['station'] == stations_to_keep[0]]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(s_train['datetime'], s_train['PM2.5'], color='steelblue', lw=0.5, label='Train')


df_winsorized = df_imputed.copy()
df_winsorized[CHEM_COLUMNS] = df_winsorized[CHEM_COLUMNS].apply(lambda x: winsorize(x ,limits=[0.0, 0.003]))

# df_clean['PM2.5'].plot(ax=axes[1], lw=0.5, color='darkorange', title='PM2.5 — After 3σ Filtering')
# df_winsorized['PM2.5'].plot(ax=axes[1], lw=0.5, color='darkorange', title='PM2.5 — After 3σ Filtering')

axes[1].plot(df_winsorized['datetime'], df_winsorized['PM2.5'], color='darkorange', lw=0.5, label='PM2.5 — After 3σ Filtering')

for ax in axes: ax.set_ylabel('µg/m³')
axes[0].hlines(y=df['PM2.5'].quantile(0.997), xmin=s_train['datetime'].iloc[0], xmax=s_train['datetime'].iloc[-1], linewidth=1, color='r')
plt.tight_layout(); plt.show()

# CELL 109
df_winsorized[CHEM_COLUMNS].describe().transpose()

# CELL 111
# df_imputed[CHEM_COLUMNS] = df_winsorized

# CELL 112


# CELL 114
# df_encoded = pd .get_dummies(df, columns=['station', 'wd'], drop_first=True)
# df_encoded['month_sin'] = np.sin(df_encoded['month'] / 12 * 2 * np.pi)
# df_encoded['month_cos'] = np.cos(df_encoded['month'] / 12 * 2 * np.pi)
# df_encoded['weekday_sin'] = np.sin(df_encoded['DayOfWeek'] / 7 * 2 * np.pi)
# df_encoded['weekday_cos'] = np.cos(df_encoded['DayOfWeek'] / 7 * 2 * np.pi)
# df_encoded['hour_sin'] = np.sin(df_encoded['hour'] / 24 * 2 * np.pi)
# df_encoded['hour_cos'] = np.cos(df_encoded['hour'] / 24 * 2 * np.pi)

# CELL 115
import re

new = cyclicTransformer.fit_transform(df_winsorized).rename(columns=lambda x: re.sub('remainder__','',x), inplace=False)

new.rename(columns=lambda x: re.sub('categorical__','',x), inplace=True)

# CELL 116
new.describe().transpose()

# CELL 117
new.info()

# CELL 118
new = new.drop(['day', 'year', "datetime", 'No'], axis = 1)

# CELL 120
# TARGET       = 'PM2.5'
# # FEATURE_COLS = [c for c in new.columns if c != TARGET]

# scaler_X = MinMaxScaler()
# scaler_y = MinMaxScaler()
# X_scaled = scaler_X.fit_transform(new[CHEM_COLUMNS].values)
# y_scaled = scaler_y.fit_transform(new[[TARGET]].values)

# df_norm = pd.DataFrame(
#     np.hstack([X_scaled, y_scaled]),
#     columns=FEATURE_COLS + [TARGET],
#     index=df_imputed.index)
# print("Normalised value ranges (sample cols):")
# df_norm[CHEM_COLUMNS].describe().round(3)


# CELL 123
mmScaler = MinMaxScaler()
mmScaler.set_output(transform='pandas')

new[CHEM_COLUMNS] = mmScaler.fit_transform(new[CHEM_COLUMNS])

# CELL 124
new.describe().transpose()

# CELL 125
val_df[CHEM_COLUMNS] = mmScaler.transform(val_df[CHEM_COLUMNS])

# CELL 126
val_df = cyclicTransformer.transform(val_df).rename(columns=lambda x: re.sub('remainder__','',x), inplace=False)

val_df.rename(columns=lambda x: re.sub('categorical__','',x), inplace=True)

# CELL 127
val_df.drop(['day', 'year', "datetime", 'No'], axis=1, inplace=True)


# CELL 128
val_df.info()

# CELL 129
test_df[CHEM_COLUMNS] = mmScaler.transform(test_df[CHEM_COLUMNS])

# CELL 130
test_df = cyclicTransformer.transform(test_df).rename(columns=lambda x: re.sub('remainder__','',x), inplace=False)

test_df.rename(columns=lambda x: re.sub('categorical__','',x), inplace=True)

# CELL 131
test_df.drop(['day', 'year', "datetime", 'No'], axis=1, inplace=True)

# CELL 132
df_clean = new

# CELL 133
df_baseline = df.copy()
df_baseline[CHEM_COLUMNS] = knn_imputer.transform(df_baseline[CHEM_COLUMNS])

# CELL 134
df_baseline.head()

# CELL 137
NN_EATURES = df_clean.columns.to_list()

# CELL 138


# CELL 139
NN_EATURES.remove('station')

# CELL 140
NN_EATURES

# CELL 141
df_clean[NN_EATURES] = df_clean[NN_EATURES].astype(np.float32)
val_df[NN_EATURES] = val_df[NN_EATURES].astype(np.float32)
test_df[NN_EATURES] = test_df[NN_EATURES].astype(np.float32)

# CELL 142
df_clean.info()

# CELL 143
print(f"NaNs in train_df: {df_clean.isnull().sum().sum()}")
print(f"NaNs in val_df:   {val_df.isnull().sum().sum()}")
print(f"NaNs in test_df:  {test_df.isnull().sum().sum()}")

# Display specific columns with NaNs in validation if any exist
if val_df.isnull().sum().sum() > 0:
    display(val_df.isnull().sum()[val_df.isnull().sum() > 0])

# CELL 144
sampling_rate = 1 # Observations will be sampled at every data point one per hour
sequence_length = 24 * 5 #  Observations will go back seven days (168 hours).
forecast_horizon = 24
delay = sampling_rate * (sequence_length + forecast_horizon - 1) # The target for a sequence will be the PM2.5 24 hours after the end of the sequence.
batch_size = 64

# CELL 145

TARGET = 'PM2.5'

def prepare_multi_station_ds(df, is_training=True):
    station_datasets = []

    # We group by station to ensure windows don't overlap between two different locations
    for _, group in df.groupby('station'):
        # 1. Convert to float32 for TensorFlow compatibility
        data = group[NN_EATURES].values.astype('float32')
        targets = group[TARGET].values.astype('float32')

        # 2. Create the windowed dataset for THIS station
        # Note: 'targets[WINDOW_SIZE:]' aligns the prediction to the step AFTER the window
        ds = keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=targets[sequence_length:],
            sequence_length=sequence_length,
            batch_size=batch_size
        )
        station_datasets.append(ds)

    # 3. Combine all stations into one master dataset
    final_ds = station_datasets[0]
    for next_ds in station_datasets[1:]:
        final_ds = final_ds.concatenate(next_ds)

    # 4. Optimize for performance
    if is_training:
        final_ds = final_ds.shuffle(1000)

    return final_ds.prefetch(tf.data.AUTOTUNE)

# Usage with your splits
train_ds = prepare_multi_station_ds(df_clean, is_training=True)
val_ds   = prepare_multi_station_ds(val_df, is_training=False)
test_ds  = prepare_multi_station_ds(test_df, is_training=False)

# CELL 146
# train_dataset = keras.utils.timeseries_dataset_from_array(
#     df_clean[:-delay],
#     targets=df_clean['PM2.5'][delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=False,
#     batch_size=batch_size,
#     # start_index=0,
#     # end_index=num_train_samples,
# )

# val_dataset = keras.utils.timeseries_dataset_from_array(
#     val_df[:-delay],
#     targets=val_df['PM2.5'][delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=False,
#     batch_size=batch_size,
#     # start_index=0,
#     # end_index=num_train_samples + num_val_samples,
# )

# test_dataset = keras.utils.timeseries_dataset_from_array(
#     test_df[:-delay],
#     targets=test_df['PM2.5'][delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=False,
#     batch_size=batch_size,
#     # start_index=num_train_samples + num_val_samples,
# )

# CELL 147
# # Re-create the Keras datasets with the cleaned data
# import keras

# train_dataset = keras.utils.timeseries_dataset_from_array(
#     df_clean[:-delay],
#     targets=df_clean['PM2.5'][delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=True,
#     batch_size=batch_size
# )

# val_dataset = keras.utils.timeseries_dataset_from_array(
#     val_df[:-delay],
#     targets=val_df['PM2.5'][delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=True,
#     batch_size=batch_size
# )

# test_dataset = keras.utils.timeseries_dataset_from_array(
#     test_df[:-delay],
#     targets=test_df['PM2.5'][delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=False
# )

# print("Datasets successfully re-created without NaNs.")

# CELL 150
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    """Custom transformer to apply winsorization to specified columns."""
    def __init__(self, limits=[0.003, 0.003]):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = winsorize(X_copy[col], limits=self.limits)
        return X_copy

    def get_feature_names_out(self): # Why does this work?
        pass

# Define feature groups
cyclic_features = ['month', 'DayOfWeek', 'hour']
cat_features = ['wd']
chemical_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

# 1. Cyclic Transformer for time features
time_pipeline = ColumnTransformer([
    ("month_sin", sin_transformer(12), ["month"]),
    ("month_cos", cos_transformer(12), ["month"]),
    ("weekday_sin", sin_transformer(7), ["DayOfWeek"]),
    ("weekday_cos", cos_transformer(7), ["DayOfWeek"]),
    ("hour_sin", sin_transformer(24), ["hour"]),
    ("hour_cos", cos_transformer(24), ["hour"]),
])

# 2. Main Preprocessing Pipeline
preprocessing_pipeline = ColumnTransformer([
    # Chemical features: Impute -> Winsorize -> Scale
    ('chem', Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('winsor', Winsorizer(limits=[0.0, 0.003])),
        ('scaler', MinMaxScaler())
    ]), chemical_cols),

    # Weather features: Impute -> Scale
    ('weather', Pipeline([
        ('imputer', KNNImputer(n_neighbors=3)),
        ('scaler', MinMaxScaler())
    ]), weather_cols),

    # Categorical: One-Hot
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),

    # Time: Cyclic encoding
    ('time_month_sin', sin_transformer(12), ['month']),
    ('time_month_cos', cos_transformer(12), ['month']),
    ('time_dow_sin', sin_transformer(7), ['DayOfWeek']),
    ('time_dow_cos', cos_transformer(7), ['DayOfWeek']),
    ('time_hour_sin', sin_transformer(24), ['hour']),
    ('time_hour_cos', cos_transformer(24), ['hour']),

    # Passthrough binary features
    ('binary', 'passthrough', ['Is_Weekend', 'Is_Rush_Hour'])
])

# Set output to pandas for easier inspection
preprocessing_pipeline.set_output(transform="pandas")

print("Preprocessing pipeline successfully defined.")

# CELL 151


# CELL 154
df_naive_baseline = df_baseline.copy()

# CELL 155
grouped = df_baseline['PM2.5']

# 1. Naive Forecast (Shift by 1 hour, grouped by station)
df_naive_baseline['Naive_Forecast'] = grouped.shift(1)

# 2. Moving Average Forecast (3-hour rolling average, grouped by station)
df_naive_baseline['Moving_Avg_Forecast'] = grouped.transform(lambda x: x.rolling(window=3).mean().shift(1))

# 3. Seasonal Naive Forecast (Shift by 24 hours, grouped by station)
df_naive_baseline['Seasonal_Naive_Forecast'] = grouped.shift(24)

# CELL 157
# 1. Isolate one station and make sure it's sorted chronologically
df_station = df_naive_baseline.copy() #[df['station'] == 'Aotizhongxin'].copy()
df_station = df_station.sort_values('datetime').reset_index(drop=True)

# 2. Set up lists to store the MAE for each horizon (1 to 24)
horizons = range(1, 25)
mae_naive, mae_ma, mae_seasonal = [], [], []

# 3. Loop through each hour in the future
for h in horizons:

    # The Ground Truth: Shift the PM2.5 column UP by 'h' rows to represent the future
    y_true = df_station['PM2.5'].shift(-h)

    # Baseline 1: Naive (Predict the current hour's value for the future)
    y_pred_naive = df_station['PM2.5']

    # Baseline 2: 3-Hour Moving Average (Predict the current average for the future)
    y_pred_ma = df_station['PM2.5'].rolling(window=3).mean()

    # Baseline 3: Seasonal Naive
    # To predict 'h' hours ahead, we look at what happened exactly 24 hours before that target time.
    y_pred_seasonal = df_station['PM2.5'].shift(24 - h)

    # Create a temporary dataframe to easily drop NAs created by all the shifting
    temp_df = pd.DataFrame({
        'True': y_true,
        'Naive': y_pred_naive,
        'MA': y_pred_ma,
        'Seasonal': y_pred_seasonal
    }).dropna()

    # Calculate and store the MAE for this specific horizon
    mae_naive.append(mean_absolute_error(temp_df['True'], temp_df['Naive']))
    mae_ma.append(mean_absolute_error(temp_df['True'], temp_df['MA']))
    mae_seasonal.append(mean_absolute_error(temp_df['True'], temp_df['Seasonal']))

    # (Optional: You could do the exact same thing here using mean_squared_error
    # wrapped in np.sqrt() to store RMSE!)

# CELL 158
df_stations = df_baseline.sort_values(by=['station', 'datetime'])


# CELL 159
for _, group in df_stations.groupby('station'):

  grouped = group['PM2.5']

  # 1. Naive Forecast (Shift by 1 hour, grouped by station)
  group['Naive_Forecast'] = grouped.shift(1)

  # 2. Moving Average Forecast (3-hour rolling average, grouped by station)
  group['Moving_Avg_Forecast'] = grouped.transform(lambda x: x.rolling(window=3).mean().shift(1))

  # 3. Seasonal Naive Forecast (Shift by 24 hours, grouped by station)
  group['Seasonal_Naive_Forecast'] = grouped.shift(24)


# CELL 160
df_stations['Naive_Forecast'] = df_stations.groupby('station')['PM2.5'].shift(1)
df_stations['Moving_Avg_Forecast'] = df_stations.groupby('station')['PM2.5'].transform(lambda x: x.rolling(window=3).mean().shift(1))
df_stations['Seasonal_Naive_Forecast'] = df_stations.groupby('station')['PM2.5'].shift(24)


# CELL 161
results = []

# Assuming df_stations is a GroupBy object: df.groupby('station_id')
for station_id, group in df_stations.groupby('station'):
    horizons = range(1, 25)

    for h in horizons:

        y_true = group['PM2.5'].shift(-h)
        y_pred_naive = group['PM2.5']
        y_pred_ma = group['PM2.5'].rolling(window=3).mean()

        # Seasonal Naive: Target is (t + h). 24 hours before target is (t + h - 24)
        # To use current/past data, we shift by (24 - h)
        y_pred_seasonal = group['PM2.5'].shift(24 - h)

        # 2. Align and drop NaNs
        temp_df = pd.DataFrame({
            'True': y_true,
            'Naive': y_pred_naive,
            'MA': y_pred_ma,
            'Seasonal': y_pred_seasonal
        }).dropna()

        # 3. Calculate metrics if we have data remaining
        if not temp_df.empty:
            results.append({
                'Station': station_id,
                'Horizon': h,
                'MAE_Naive': mean_absolute_error(temp_df['True'], temp_df['Naive']),
                'MAE_MA': mean_absolute_error(temp_df['True'], temp_df['MA']),
                'MAE_Seasonal': mean_absolute_error(temp_df['True'], temp_df['Seasonal'])
            })

# Convert results list to a clean DataFrame
df_results = pd.DataFrame(results)


# CELL 162
df_results

# CELL 163
# 1. Aggregate the results to get the mean MAE per horizon across all stations
avg_results = df_results.groupby('Horizon').mean(numeric_only=True).reset_index()

# 2. Plotting
plt.figure(figsize=(10, 6))

# Plot the three error curves based on the averages
plt.plot(avg_results['Horizon'], avg_results['MAE_Naive'],
         label='Naïve Forecast', marker='o', color='gray')
plt.plot(avg_results['Horizon'], avg_results['MAE_MA'],
         label='3-hr Moving Average', marker='s', color='orange')
plt.plot(avg_results['Horizon'], avg_results['MAE_Seasonal'],
         label='Seasonal Naïve (24hr cycle)', marker='^', color='green')

# Format the chart
plt.title('Baseline Forecast Degradation: Average Across All Stations')
plt.xlabel('Forecast Horizon (Hours Ahead)')
plt.ylabel('Mean Absolute Error (µg/m³)')
plt.xticks(range(1, 25))
plt.legend()
plt.grid(alpha=0.4)

plt.tight_layout()
plt.show()

# CELL 165
df_baseline.describe().transpose()

# CELL 166
columns_to_drop = [col for col in df_baseline.columns.to_list() if col not in ["PM2.5", "station", "datetime"]]

# CELL 167
from statsforecast import StatsForecast

horizon = 24 # 24 hour period

models = [Naive(), HistoricAverage(), WindowAverage(window_size=24), SeasonalNaive(season_length=24)]

test = df_baseline.drop(columns_to_drop, axis=1).groupby('station').tail(horizon)
train = df_baseline.drop(columns_to_drop, axis=1).drop(test.index)

sf = StatsForecast(models=models, freq='H', n_jobs=-1)

sf.fit(train, id_col="station", time_col='datetime', target_col="PM2.5")

preds = sf.predict(h=horizon)

eval_df = pd.merge(test, preds, on=['station', 'datetime'], how='left')

# CELL 168
preds.head()

# CELL 169


# CELL 170
from utilsforecast.losses import mae

from utilsforecast.evaluation import evaluate


evaluation = evaluate(
    eval_df,
    metrics=[mae],
    id_col="station", time_col='datetime', target_col="PM2.5"
)

evaluation

# CELL 171
evaluation = evaluation.drop(columns='station').groupby('metric').mean().reset_index()


methods = evaluation.columns[1:].tolist()

values = evaluation.iloc[0, 1:].tolist()

# CELL 172
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.title('Average MAE Across All Stations')
plt.xlabel('Methods')
plt.ylabel('Mean Absolute Error (MAE)')
plt.tight_layout()
plt.show()

# CELL 174
plot_series(test, forecasts_df=preds, max_insample_length=24, id_col="station", time_col='datetime', target_col="PM2.5")

# CELL 176
ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=10000,
    test_size=1000,
)

# CELL 180
# T-SNE Implementation
# X_reduced_tsne = TSNE(n_components=3, random_state=69420).fit_transform(df_clean.values)

# CELL 181
df_components = df_baseline[df['station'] == stations_to_keep[0]][FEATURE_COLS].drop('wd', axis=1)

# CELL 182
df_components

# CELL 183
# PCA Implementation
X_reduced_pca = PCA(n_components=4, random_state=69420).fit_transform(df_components.values)

# CELL 184
# TruncatedSVD
X_reduced_svd = TruncatedSVD(
    n_components=len(df_components.columns.to_list()),
    algorithm='randomized',
    random_state=69420).fit_transform(df_components.values)

# CELL 185
f, (ax2, ax3) = plt.subplots(2, 1, figsize=(9,16))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


# blue_patch = mpl.patches.Patch(color='#0A0AFF', label='No Ransom')
# red_patch = mpl.patches.Patch(color='#AF0000', label='Ransom')


# t-SNE scatter plot
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Ransom', linewidths=2)
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=df_clean['PM2.5'], cmap='coolwarm', label='PM2.5', linewidths=2)
# ax1.set_title('t-SNE', fontsize=14)

# ax1.grid(True)

# ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
# ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=df_components['PM2.5'], cmap='coolwarm', label='PM2.5', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

# ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
# ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=df_components['PM2.5'], cmap='coolwarm', label='PM2.5', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

# ax3.legend(handles=[blue_patch, red_patch])

plt.show()

# CELL 187
from scipy.stats import pearsonr
import warnings

# ── Config ────────────────────────────────────────────────────────────────────
TARGET    = 'PM2.5'                        # correlate against raw PM2.5
LAG_RANGE = range(1, 168)                   # test lags 1h → 48h
LAG_FEATURES  = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
             'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']


# CELL 188
# ── 1. Compute Pearson r at each lag for each feature ─────────────────────────
# Work on a single station to keep temporal ordering clean
df_station = df_clean.copy()

results = []

for feature in LAG_FEATURES:
    for lag in LAG_RANGE:
        lagged  = df_station[feature].shift(lag)       # feature at time t-lag
        target  = df_station[TARGET]                   # PM2.5 at time t

        # Drop NaNs introduced by shift
        valid   = pd.concat([lagged, target], axis=1).dropna()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])

        results.append({
            'Feature': feature,
            'Lag (hours)': lag,
            'Pearson r': r,
            'p-value': p,
            'Significant': p < 0.05
        })

results_df = pd.DataFrame(results)


# CELL 189
# ── 2. Find optimal lag per feature (highest |r|) ─────────────────────────────
best_lags = (
    results_df.loc[results_df.groupby('Feature')['Pearson r']
                              .apply(lambda x: x.abs().idxmax())]
    .sort_values('Pearson r', ascending=False)
    .reset_index(drop=True)
)
print("Optimal lag per feature:")
print(best_lags[['Feature', 'Lag (hours)', 'Pearson r', 'p-value']].to_string(index=False))


# CELL 190
# ── 3. Heatmap — Pearson r across all features and lags ──────────────────────
pivot = results_df.pivot(index='Feature', columns='Lag (hours)', values='Pearson r')

fig, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(
    pivot,
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    linewidths=0.2, ax=ax,
    cbar_kws={'label': 'Pearson r'}
)
ax.set_title(f'Time-Lag Correlation with {TARGET}')
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.show()


# CELL 191
# ── 4. Line plot — correlation decay curves per feature ──────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
for feature in LAG_FEATURES:
    subset = results_df[results_df['Feature'] == feature]
    ax.plot(subset['Lag (hours)'], subset['Pearson r'], label=feature, linewidth=1.5)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('Pearson r')
ax.set_title(f'Correlation Decay Curves — Feature vs {TARGET} at Increasing Lags')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

# ── 5. (Optional) Add optimal lag features back to the model ─────────────────
for _, row in best_lags.iterrows():
    feat    = row['Feature']
    optimal = int(row['Lag (hours)'])
    col     = f"{feat}_lag{optimal}"
    df[col] = df.groupby('station')[feat].shift(optimal)
    print(f"Added: {col}  (r={row['Pearson r']:.3f})")

# CELL 193
categorical_columns = df.columns[df.dtypes == "category"]
print("Categorical features:", categorical_columns.tolist())

# CELL 194
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import RidgeCV


# alphas = np.logspace(-6, 6, 25)

# cyclic_cossin_linear_pipeline = make_pipeline(
#     # cyclic_cossin_transformer,
#     RidgeCV(alphas=alphas),
# )

# tempX = df_clean.copy()
# tempy = df_clean['PM2.5']

# sklearnEvaluate(cyclic_cossin_linear_pipeline, tempX, tempy, cv=ts_cv)

# CELL 196
df_baseline.info()

# CELL 197
df_baseline.iloc[:, 1:22].head()

# CELL 198
df_tree = df_baseline.iloc[:, 1:22].drop('datetime', axis=1)

# CELL 201
gbrt = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)

# CELL 202
sklearnEvaluate(gbrt, df_tree.drop('PM2.5', axis = 1), df_tree['PM2.5'], cv=ts_cv, model_prop="n_iter_")

# CELL 203


# CELL 204


# CELL 206
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# CELL 207
sklearnEvaluate(rf_model, df_tree.drop(['PM2.5', 'wd', 'station'], axis = 1), df_tree['PM2.5'], cv=ts_cv)

# CELL 208


# CELL 209


# CELL 212
WINDOW     = 24   # use the past 24 hours as context
THRESHOLD  = 75
BATCH_SIZE = 128
EPOCHS     = 32 # https://link.springer.com/article/10.1007/s11063-024-11656-3/tables/3

# CELL 213
import sys

def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None: seen = set()
    obj_id = id(obj)
    if obj_id in seen: return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# Check top memory consuming variables
mem_usage = []
for name, value in list(globals().items()):
    if not name.startswith('_'):
        size_mb = sys.getsizeof(value) / (1024**2)
        if size_mb > 1: # Only show objects > 1MB
            mem_usage.append((name, size_mb))

mem_df = pd.DataFrame(mem_usage, columns=['Variable', 'Size (MB)']).sort_values('Size (MB)', ascending=False)
display(mem_df)

# CELL 214
import gc

# List of DataFrames that are likely no longer needed after preprocessing
vars_to_delete = ['df', 'df_stations', 'df_naive_baseline', 'df_baseline', 'df_imputed', 'df_winsorized']

for var in vars_to_delete:
    if var in globals():

        del globals()[var]
        print(f'Deleted {var}')

# Force garbage collection
gc.collect()
print('Memory cleanup complete.')

# CELL 215
df = None
df_stations = None
df_naive_baseline = None
df_baseline = None
df_imputed = None
df_winsorized = None
new = None
train_df = None
tempX = None
df_single_station = None
df_station = None
df_clean = None
multi_pivot = None
test_df = None
tempy = None
group = None
target = None
train = None
s_train = None
X_reduced_pca = None
X_reduced_svd = None
temp_df = None
s_val = None
s_test = None
df_daily_scaled = None
valid = None

# CELL 216
globals()

# CELL 217
# # ── 2. Scale features (required for neural nets) ─────────────────────────────
# scaler    = StandardScaler()
# X_scaled  = scaler.fit_transform(X)


# CELL 218
# # ── 3. Build sliding windows ──────────────────────────────────────────────────
# # Each sample becomes a (WINDOW, n_features) sequence
# def make_sequences(X, y, window):
#     Xs, ys = [], []
#     for i in range(window, len(X)):
#         Xs.append(X[i - window:i])   # shape: (window, n_features)
#         ys.append(y.iloc[i])
#     return np.array(Xs), np.array(ys)

# X_seq, y_seq = make_sequences(X_scaled, y, WINDOW)
# # X_seq shape: (n_samples, 24, n_features)



# CELL 219
df_clean.shape()

# CELL 220
n_steps    = sequence_length
n_features = len(NN_EATURES)

print("n_steps:", n_steps)
print("n_features:", n_features)

# CELL 222
METRICS = ['mae', 'mse', 'mape'] # 'rmse',

# CELL 223
# ── 4. Model definitions ──────────────────────────────────────────────────────
def build_rnn():
    m = Sequential()
    m.add(Input(shape=(n_steps, n_features)))
    m.add(SimpleRNN(64, return_sequences=False))
    m.add(Dropout(0.2))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m

def build_lstm():
    m = Sequential()
    m.add(Input(shape=(n_steps, n_features)))
    m.add(LSTM(64, return_sequences=False))
    m.add(Dropout(0.2))
    m.add(Dense(1, activation='sigmoid'))

    m.compile(optimizer='adam', loss='mse', metrics=METRICS)
    return m

def build_cnn():
    m = Sequential()
    m.add(Input(shape=(n_steps, n_features)))
    m.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    m.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    m.add(GlobalMaxPooling1D())
    m.add(Dropout(0.2))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m

builders = {
    'RNN':  build_rnn,
    'LSTM': build_lstm,
    'CNN':  build_cnn,
}

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# CELL 225
# ── 5. Time-series CV ─────────────────────────────────────────────────────────
metric_keys = ['MAE', 'MSE', 'RMSE']
nn_metrics = {name: {m: [] for m in metric_keys} for name in builders}

for fold, (train_idx, test_idx) in enumerate(ts_cv.split(train_dataset, y_seq), start=1):
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]

    fold_summary = f"Fold {fold}"

    for name, build_fn in builders.items():
        model = build_fn()   # fresh model each fold
        print(model.summary())
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )

        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        nn_metrics[name]['MAE'].append(mean_absolute_error(y_test, y_pred))
        nn_metrics[name]['MSE'].append(mean_squared_error(y_test, y_pred))
        nn_metrics[name]['RMSE'].append(root_mean_squared_error(y_test, y_pred))

        fold_summary += f" | {name} MAE: {nn_metrics[name]['MAE'][-1]:.3f}"
        fold_summary += f" | {name} MSE: {nn_metrics[name]['MSE'][-1]:.3f}"
        fold_summary += f" | {name} RMSE: {nn_metrics[name]['RMSE'][-1]:.3f}"
        fold_summary += '\n'

    print(fold_summary)


# CELL 226
# ── 6. Merge with existing results and plot ───────────────────────────────────
nn_results_df = pd.DataFrame({
    name: {m: np.mean(v) for m, v in metrics.items()}
    for name, metrics in nn_metrics.items()
}).T

all_results_df = pd.concat([results_df, nn_results_df]).sort_values('M', ascending=False)

print("\nFull model comparison:")
print(all_results_df.to_string(float_format='{:.3f}'.format))

all_results_df.plot(kind='bar', figsize=(12, 5), ylim=(0, 1), rot=20)
plt.title('All Models — 24h Ahead Forecast (Time-Series CV)')
plt.ylabel('Score (averaged across folds)')
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.4)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# CELL 227
# ── 5. Time-series CV ─────────────────────────────────────────────────────────
metric_keys = ['MAE', 'MSE', 'RMSE']
nn_metrics = {name: {m: [] for m in metric_keys} for name in builders}

fold_summary = f"Fold "

for name, build_fn in builders.items():
    model = build_fn()   # fresh model each fold
    print(model.summary())
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        # verbose=0
    )

    y_prob = model.predict(test_dataset).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    print(model.evaluate(test_dataset))

    nn_metrics[name]['MAE'].append(model.evaluate(test_dataset)[1])
    # nn_metrics[name]['MSE'].append(mean_squared_error(y_test, y_pred))
    # nn_metrics[name]['RMSE'].append(root_mean_squared_error(y_test, y_pred))

    fold_summary += f" | {name} MAE: {nn_metrics[name]['MAE'][-1]:.3f}"
    # fold_summary += f" | {name} MSE: {nn_metrics[name]['MSE'][-1]:.3f}"
    # fold_summary += f" | {name} RMSE: {nn_metrics[name]['RMSE'][-1]:.3f}"
    fold_summary += '\n'

print(fold_summary)


# CELL 229


# CELL 230
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras", save_best_only=True)
]

model = build_rnn()
model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)

model = keras.models.load_model("jena_dense.keras")
print(f"Test MAE: {model.evaluate(test_ds)[1]:.3f}")
print(f"Test MAE: {scaler.inverse_transform(model.evaluate(test_ds)[1]):.3f}")

# CELL 231
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras", save_best_only=True)
]

model = build_lstm()
model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)

model = keras.models.load_model("jena_dense.keras")
print(f"Test MAE: {model.evaluate(test_ds)[1]:.3f}")
print(f"Test MAE: {scaler.inverse_transform(model.evaluate(test_ds)[1]):.3f}")

# CELL 232
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras", save_best_only=True)
]

model = build_cnn()

model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)

model = keras.models.load_model("jena_dense.keras")
print(f"Test MAE: {model.evaluate(test_ds)[1]:.3f}")
print(f"Test MAE: {scaler.inverse_transform(model.evaluate(test_ds)[1]):.3f}")

# CELL 233
x_test, y = next(test_ds.as_numpy_iterator())
y_pred = model.predict(x_test)

plt.figure(figsize=(18, 6))
plt.plot(y[:, 0, 0])
plt.plot(y_pred[:, 0, 0])
plt.legend(["actual", "forecast"])

naive_mse, model_mse = (
    np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean(),
    np.square(y_pred[:, 0, :] - y[:, 0, :]).mean(),
)

naive24_mse = np.square(x_test[:, -24, :, 0] - y[:, 0, :]).mean()
naive12_mse = np.square(x_test[:, -12, :, 0] - y[:, 0, :]).mean()


print(f"naive MAE: {naive_mse}, model MAE: {model_mse}")

# CELL 234
x_test, y = next(test_ds.as_numpy_iterator())
y_pred = model.predict(x_test)

plt.figure(figsize=(18, 6))
plt.plot(y)
plt.plot(y_pred[:, 0])
plt.legend(["actual", "forecast"])

naive_mse, model_mse = (
    np.square(x_test[:, -1, :] - y).mean(),
    np.square(y_pred - y).mean(),
)

# CELL 235
x_test.shape

# CELL 236
y_pred.shape

# CELL 237
x_test, y = next(test_ds.as_numpy_iterator())
y_pred = model.predict(x_test)

plt.figure(figsize=(18, 6))
plt.plot(y[:, 0, 0])
plt.plot(y_pred[:, 0, 0])
plt.legend(["actual", "forecast"])

naive_mse, model_mse = (
    np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean(),
    np.square(y_pred[:, 0, :] - y[:, 0, :]).mean(),
)

# CELL 239
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers import BatchNormalization


# CELL 240
# ── Experiment variants ───────────────────────────────────────────────────────
def build_baseline():
    """Large network, no regularization — expected to overfit"""
    m = Sequential([
        LSTM(128, input_shape=(n_steps, n_features), return_sequences=True),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_reduced():
    """Smaller network — fewer parameters = less capacity to memorise"""
    m = Sequential([
        LSTM(32, input_shape=(n_steps, n_features)),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_l1_l2():
    """Full size + L1/L2 weight penalties on LSTM kernel and recurrent kernel"""
    m = Sequential([
        LSTM(128, input_shape=(n_steps, n_features), return_sequences=True,
             kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3),
             recurrent_regularizer=l2(1e-3)),
        LSTM(64,
             kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_dropout():
    """Full size + dropout on outputs and recurrent connections"""
    m = Sequential([
        LSTM(128, input_shape=(n_steps, n_features), return_sequences=True,
             dropout=0.3,              # drops input/output connections
             recurrent_dropout=0.2),   # drops recurrent connections
        LSTM(64,
             dropout=0.3,
             recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

# CELL 241

variants = {
    'Baseline (no reg)': build_baseline,
    'Reduced Size':      build_reduced,
    'L1 + L2':           build_l1_l2,
    'Dropout':           build_dropout,
}

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# CELL 242
# ── Run experiment: one fold only for speed, use last fold for fair comparison ─
train_idx, test_idx = list(ts_cv.split(X_seq, y_seq))[-1]   # final fold
X_train_e, X_test_e = X_seq[train_idx], X_seq[test_idx]
y_train_e, y_test_e = y_seq[train_idx], y_seq[test_idx]

histories   = {}
exp_results = {}

for name, build_fn in variants.items():
    print(f"\nTraining: {name}")
    model = build_fn()

    history = model.fit(
        X_train_e, y_train_e,
        validation_split=0.15,
        epochs=60,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )
    histories[name] = history.history

    y_pred = (model.predict(X_test_e, verbose=0).flatten() > 0.5).astype(int)
    exp_results[name] = {
        'F1':        f1_score(y_test_e, y_pred),
        'Precision': precision_score(y_test_e, y_pred),
        'Recall':    recall_score(y_test_e, y_pred),
        'Accuracy':  accuracy_score(y_test_e, y_pred),
        'Best epoch': len(history.history['loss']),    # epochs before early stop
        'Train loss': min(history.history['loss']),
        'Val loss':   min(history.history['val_loss']),
    }


# CELL 243
# ── Results table ─────────────────────────────────────────────────────────────
exp_df = pd.DataFrame(exp_results).T
print("\nOverfitting Experiment Results:")
print(exp_df.to_string(float_format='{:.3f}'.format))

# ── Plot 1: Train vs Val loss curves (overfitting diagnostic) ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (name, hist) in zip(axes, histories.items()):
    epochs = range(1, len(hist['loss']) + 1)
    ax.plot(epochs, hist['loss'],     label='Train loss', linewidth=1.8)
    ax.plot(epochs, hist['val_loss'], label='Val loss',   linewidth=1.8, linestyle='--')
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # Highlight the gap between train and val (overfitting zone)
    ax.fill_between(epochs,
                    hist['loss'], hist['val_loss'],
                    alpha=0.1, color='red', label='Overfit gap')

plt.suptitle('Overfitting Experiment — Train vs Validation Loss', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()


# CELL 244
# ── Plot 2: F1 comparison bar chart ──────────────────────────────────────────
exp_df[['F1', 'Precision', 'Recall', 'Accuracy']].plot(
    kind='bar', figsize=(10, 5), ylim=(0, 1), rot=15
)
plt.title('Overfitting Experiment — Test Metrics Comparison')
plt.ylabel('Score')
plt.tight_layout()
plt.show()

# ── Plot 3: Overfit gap bar (train loss − val loss per variant) ───────────────
gap = exp_df['Train loss'] - exp_df['Val loss']
gap.sort_values().plot(
    kind='barh', figsize=(8, 4),
    color=['green' if v <= 0 else 'tomato' for v in gap.sort_values()]
)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Train − Val Loss Gap  (negative = generalising well, positive = overfitting)')
plt.xlabel('Loss gap')
plt.tight_layout()
plt.show()

# CELL 245


# CELL 246


# CELL 248
from sklearn.model_selection import GridSearchCV

# Define the model
rf = RandomForestRegressor(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Initialize GridSearchCV with the time-series cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=ts_cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

# Since df_clean is our processed training data, we split it into X and y
# Assuming 'PM2.5' is the target
X_tuning = df_clean.drop(columns=['PM2.5'])
y_tuning = df_clean['PM2.5']

# Fit the grid search
print("Starting Grid Search...")
grid_search.fit(X_tuning, y_tuning)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best Score (MAE): {-grid_search.best_score_:.3f}")
