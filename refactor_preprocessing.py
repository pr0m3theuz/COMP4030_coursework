import json

def apply_refactor():
    with open("beijing_air_quality.ipynb", "r") as f:
        nb = json.load(f)

    # I'll create a dedicated cell with the new preprocessing functions.
    # We will place it after the load_data cell (around index 18)

    preprocess_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Data Preprocessing\n",
            "This section handles filling missing data, resolving outliers via winsorization, and creating an overarching preprocessing pipeline."
        ]
    }

    preprocess_functions = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "from scipy.stats.mstats import winsorize\n",
            "from sklearn.impute import KNNImputer\n",
            "\n",
            "def impute_cross_sectional(df, columns):\n",
            "    \"\"\"\n",
            "    Fills missing values using the mean of other stations for the same date.\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): Dataframe containing 'datetime' and 'station' columns.\n",
            "        columns (list): List of numerical columns to impute.\n",
            "    Returns:\n",
            "        pd.DataFrame: Dataframe with cross-sectional imputations applied.\n",
            "    \"\"\"\n",
            "    df_out = df.copy()\n",
            "    for col in columns:\n",
            "        df_out[col] = df_out.groupby('datetime')[col].transform(lambda x: x.fillna(x.mean()))\n",
            "    return df_out\n",
            "\n",
            "def impute_knn_station(df, columns, k=5):\n",
            "    \"\"\"\n",
            "    Fills remaining missing values using a KNN imputer per station.\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): Dataframe containing 'station'.\n",
            "        columns (list): Numerical columns to impute.\n",
            "        k (int): Number of neighbors.\n",
            "    Returns:\n",
            "        pd.DataFrame: Imputed Dataframe.\n",
            "    \"\"\"\n",
            "    df_out = df.copy()\n",
            "    knn = KNNImputer(n_neighbors=k)\n",
            "    \n",
            "    for station, group in df_out.groupby('station'):\n",
            "        idx = group.index\n",
            "        if not group[columns].empty:\n",
            "            df_out.loc[idx, columns] = knn.fit_transform(group[columns])\n",
            "            \n",
            "    return df_out\n",
            "\n",
            "def winsorize_features(df, columns, limits=(0.01, 0.05)):\n",
            "    \"\"\"\n",
            "    Winsorizes (caps/floors) extreme outliers in the specified columns.\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): The Dataframe.\n",
            "        columns (list): List of columns to winsorize.\n",
            "        limits (tuple): Lower and upper percentiles for clipping.\n",
            "    Returns:\n",
            "        pd.DataFrame: Dataframe with winsorized columns.\n",
            "    \"\"\"\n",
            "    df_out = df.copy()\n",
            "    for col in columns:\n",
            "        # Apply winsorize on 1D array\n",
            "        df_out[col] = winsorize(df_out[col].values, limits=limits)\n",
            "    return df_out\n",
            "\n",
            "def preprocess_data(df, numerical_cols, categorical_cols):\n",
            "    \"\"\"\n",
            "    Executes the full preprocessing pipeline: cross-sectional imputation, \n",
            "    KNN imputation, categorical forward-fill, and winsorization.\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): Raw dataframe.\n",
            "        numerical_cols (list): Numerical columns to clean.\n",
            "        categorical_cols (list): Categorical columns to clean.\n",
            "    Returns:\n",
            "        pd.DataFrame: The cleaned dataframe.\n",
            "    \"\"\"\n",
            "    # 1. Impute Numerical (Cross-sectional then KNN)\n",
            "    df_clean = impute_cross_sectional(df, numerical_cols)\n",
            "    df_clean = impute_knn_station(df_clean, numerical_cols, k=5)\n",
            "    \n",
            "    # 2. Impute Categorical (Forward-fill within station)\n",
            "    for col in categorical_cols:\n",
            "        df_clean[col] = df_clean.groupby('station')[col].ffill(limit=5)\n",
            "        df_clean[col] = df_clean.groupby('station')[col].bfill(limit=5) # backup bfill\n",
            "        \n",
            "    # 3. Handle Outliers (Winsorization)\n",
            "    df_clean = winsorize_features(df_clean, numerical_cols, limits=(0.01, 0.05))\n",
            "    \n",
            "    return df_clean\n"
        ]
    }

    preprocess_execution = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# We apply our preprocessing to the entire dataframe before splitting.\n",
            "CHEM_COLUMNS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']\n",
            "CAT_COLUMNS = ['wd']\n",
            "\n",
            "df = preprocess_data(df, CHEM_COLUMNS, CAT_COLUMNS)\n",
            "print(f\"Missing values remaining: {df.isnull().sum().sum()}\")\n"
        ]
    }

    # Insert right after the data loading cell (which is around index 18)
    nb["cells"].insert(19, preprocess_markdown)
    nb["cells"].insert(20, preprocess_functions)
    nb["cells"].insert(21, preprocess_execution)

    with open("beijing_air_quality.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

apply_refactor()
