import json

def apply_refactor():
    with open("beijing_air_quality.ipynb", "r") as f:
        nb = json.load(f)

    # Add splitting and sequence generation functions
    split_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Data Splitting & Sequence Generation Functions\n",
            "Functions to handle chronological splitting for time-series data and generating sliding window sequences for deep learning models."
        ]
    }

    split_functions = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "\n",
            "def split_data_chronologically(df, train_ratio=0.7, val_ratio=0.15):\n",
            "    \"\"\"\n",
            "    Splits a dataframe chronologically, ensuring that each station is split at the same relative timestamps.\n",
            "    Assumes df is already sorted by ['station', 'datetime'].\n",
            "    \n",
            "    Args:\n",
            "        df (pd.DataFrame): Sorted dataframe.\n",
            "        train_ratio (float): Proportion of data for training.\n",
            "        val_ratio (float): Proportion of data for validation.\n",
            "        \n",
            "    Returns:\n",
            "        tuple: (train_df, val_df, test_df)\n",
            "    \"\"\"\n",
            "    train_list, val_list, test_list = [], [], []\n",
            "    \n",
            "    for station_id, group in df.groupby('station'):\n",
            "        n = len(group)\n",
            "        n_train = int(n * train_ratio)\n",
            "        n_val = int(n * val_ratio)\n",
            "        \n",
            "        train_list.append(group.iloc[:n_train])\n",
            "        val_list.append(group.iloc[n_train:n_train + n_val])\n",
            "        test_list.append(group.iloc[n_train + n_val:])\n",
            "        \n",
            "    train_df = pd.concat(train_list).sort_values(['station', 'datetime']).reset_index(drop=True)\n",
            "    val_df = pd.concat(val_list).sort_values(['station', 'datetime']).reset_index(drop=True)\n",
            "    test_df = pd.concat(test_list).sort_values(['station', 'datetime']).reset_index(drop=True)\n",
            "    \n",
            "    return train_df, val_df, test_df\n",
            "\n",
            "def create_sequences(data_features, data_target, sequence_length):\n",
            "    \"\"\"\n",
            "    Creates sliding window sequences for Deep Learning models (e.g. LSTMs).\n",
            "    \n",
            "    Args:\n",
            "        data_features (np.array): Array of feature values.\n",
            "        data_target (np.array): Array of target values.\n",
            "        sequence_length (int): The window size.\n",
            "        \n",
            "    Returns:\n",
            "        tuple: (X_seq, y_seq) as numpy arrays.\n",
            "    \"\"\"\n",
            "    Xs, ys = [], []\n",
            "    for i in range(sequence_length, len(data_features)):\n",
            "        Xs.append(data_features[i - sequence_length:i])\n",
            "        ys.append(data_target[i])\n",
            "    return np.array(Xs), np.array(ys)\n"
        ]
    }

    # Insert right before "# Cross Validation: Create Sequences for Deep Learning"
    insert_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and "Cross Validation: Create Sequences" in "".join(cell.get("source", [])):
            insert_idx = i
            break

    if insert_idx != -1:
        nb["cells"].insert(insert_idx, split_markdown)
        nb["cells"].insert(insert_idx + 1, split_functions)

    with open("beijing_air_quality.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

apply_refactor()
