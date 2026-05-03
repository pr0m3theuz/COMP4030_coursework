import json

def apply_refactor():
    with open("beijing_air_quality.ipynb", "r") as f:
        nb = json.load(f)

    # Add feature engineering and scaling functions
    feat_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Feature Engineering & Scaling Functions\n",
            "This section encapsulates functions to handle cyclic feature engineering and data scaling (Normalization)."
        ]
    }

    feat_functions = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "from sklearn.preprocessing import MinMaxScaler, StandardScaler\n",
            "\n",
            "def engineer_features(df):\n",
            "    \"\"\"\n",
            "    Applies feature engineering such as cyclic transforms for datetime and one-hot encoding.\n",
            "    Note: We rely on ColumnTransformer defined earlier (cyclicTransformer) for standard ops.\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): Dataframe with datetime features.\n",
            "    Returns:\n",
            "        pd.DataFrame: Engineered dataframe.\n",
            "    \"\"\"\n",
            "    df_out = df.copy()\n",
            "    # If cyclicTransformer is used, it usually outputs numpy arrays.\n",
            "    # This is a placeholder for direct pandas transformations if desired.\n",
            "    # Currently, pandas 'get_dummies' can be easily used for one-hot encoding wind direction:\n",
            "    if 'wd' in df_out.columns:\n",
            "        df_out = pd.get_dummies(df_out, columns=['wd'], prefix='wd')\n",
            "    return df_out\n",
            "\n",
            "def scale_data(train_df, val_df, test_df, columns_to_scale, scaler_type='minmax'):\n",
            "    \"\"\"\n",
            "    Scales numerical features using the specified scaler.\n",
            "    Fits ONLY on the training data to prevent data leakage, then transforms train, val, and test.\n",
            "    \n",
            "    Args:\n",
            "        train_df (pd.DataFrame): Training dataframe.\n",
            "        val_df (pd.DataFrame): Validation dataframe.\n",
            "        test_df (pd.DataFrame): Testing dataframe.\n",
            "        columns_to_scale (list): List of numerical columns.\n",
            "        scaler_type (str): 'minmax' or 'standard'.\n",
            "        \n",
            "    Returns:\n",
            "        tuple: (train_scaled, val_scaled, test_scaled, scaler_object)\n",
            "    \"\"\"\n",
            "    if scaler_type == 'minmax':\n",
            "        scaler = MinMaxScaler()\n",
            "    else:\n",
            "        scaler = StandardScaler()\n",
            "        \n",
            "    # Copy data to avoid modifying original\n",
            "    train_s = train_df.copy()\n",
            "    val_s = val_df.copy()\n",
            "    test_s = test_df.copy()\n",
            "    \n",
            "    # Fit on training\n",
            "    scaler.fit(train_s[columns_to_scale])\n",
            "    \n",
            "    # Transform all\n",
            "    train_s[columns_to_scale] = scaler.transform(train_s[columns_to_scale])\n",
            "    val_s[columns_to_scale] = scaler.transform(val_s[columns_to_scale])\n",
            "    test_s[columns_to_scale] = scaler.transform(test_s[columns_to_scale])\n",
            "    \n",
            "    return train_s, val_s, test_s, scaler\n"
        ]
    }

    # Let's insert this before "# Preporcessing for Validation & Testing Datasets"
    insert_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and "Preporcessing for Validation" in "".join(cell.get("source", [])):
            insert_idx = i
            break

    if insert_idx != -1:
        nb["cells"].insert(insert_idx, feat_markdown)
        nb["cells"].insert(insert_idx + 1, feat_functions)

    with open("beijing_air_quality.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

apply_refactor()
