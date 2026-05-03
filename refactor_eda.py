import json

def apply_refactor():
    with open("beijing_air_quality.ipynb", "r") as f:
        nb = json.load(f)

    # Add EDA functions
    eda_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Data Exploration (EDA) Functions\n",
            "This section defines helper functions to visualize and understand the data distribution, correlations, and missing values."
        ]
    }

    eda_functions = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "def plot_missing_data(df):\n",
            "    \"\"\"\n",
            "    Plots the count of missing values per column.\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): Dataframe to analyze.\n",
            "    \"\"\"\n",
            "    missing = df.isnull().sum().sort_values(ascending=False)\n",
            "    missing = missing[missing > 0]\n",
            "    if missing.empty:\n",
            "        print(\"No missing values found!\")\n",
            "        return\n",
            "    \n",
            "    plt.figure(figsize=(10, 5))\n",
            "    missing.plot(kind='bar', color='salmon')\n",
            "    plt.title('Missing Values by Column')\n",
            "    plt.ylabel('Count')\n",
            "    plt.show()\n",
            "\n",
            "def plot_feature_evolution(df, feature, station=None):\n",
            "    \"\"\"\n",
            "    Plots the time-series evolution of a specific feature.\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): Dataframe to plot.\n",
            "        feature (str): Column name to plot.\n",
            "        station (str, optional): Station to filter by.\n",
            "    \"\"\"\n",
            "    plt.figure(figsize=(15, 5))\n",
            "    if station:\n",
            "        subset = df[df['station'] == station]\n",
            "        plt.plot(subset['datetime'], subset[feature], linewidth=0.5)\n",
            "        plt.title(f'{feature} Evolution over Time for {station}')\n",
            "    else:\n",
            "        # If no station is provided, plot the mean across all stations\n",
            "        mean_series = df.groupby('datetime')[feature].mean()\n",
            "        plt.plot(mean_series.index, mean_series.values, linewidth=0.5)\n",
            "        plt.title(f'Mean {feature} Evolution over Time Across All Stations')\n",
            "    \n",
            "    plt.xlabel('Date')\n",
            "    plt.ylabel(feature)\n",
            "    plt.show()\n",
            "\n",
            "def plot_correlations(df, cols, title='Feature Correlation Map'):\n",
            "    \"\"\"\n",
            "    Plots a seaborn heatmap of the correlation matrix for specified columns.\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): Dataframe to correlate.\n",
            "        cols (list): List of numerical columns.\n",
            "        title (str): Title for the plot.\n",
            "    \"\"\"\n",
            "    corr = df[cols].corr()\n",
            "    plt.figure(figsize=(10, 8))\n",
            "    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\n",
            "    plt.title(title)\n",
            "    plt.show()\n",
            "\n",
            "def plot_time_series_decomposition(df, feature, station, period=365):\n",
            "    \"\"\"\n",
            "    Decomposes and plots a time series into trend, seasonal, and residual components.\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): Dataframe.\n",
            "        feature (str): Column to decompose.\n",
            "        station (str): Station to analyze.\n",
            "        period (int): Period for seasonal decomposition (e.g. 365 for daily data).\n",
            "    \"\"\"\n",
            "    subset = df[df['station'] == station].copy()\n",
            "    subset.set_index('datetime', inplace=True)\n",
            "    # Resample daily to smooth and avoid missing timestamps\n",
            "    daily_series = subset[feature].resample('D').mean().fillna(method='ffill')\n",
            "    \n",
            "    result = seasonal_decompose(daily_series, model='additive', period=period)\n",
            "    \n",
            "    fig = result.plot()\n",
            "    fig.set_size_inches(12, 8)\n",
            "    plt.suptitle(f'{feature} Decomposition for {station}', fontsize=14)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n"
        ]
    }

    # Insert right before "# Data Exploration" which is around cell 30
    insert_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and "# Data Exploration" in "".join(cell.get("source", [])):
            insert_idx = i
            break

    if insert_idx != -1:
        nb["cells"].insert(insert_idx, eda_markdown)
        nb["cells"].insert(insert_idx + 1, eda_functions)

    with open("beijing_air_quality.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

apply_refactor()
