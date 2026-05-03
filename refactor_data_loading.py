import json

def apply_refactor():
    with open("beijing_air_quality.ipynb", "r") as f:
        nb = json.load(f)

    # Find the "# Load Dataframes" cell
    load_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and "# Load Dataframes" in "".join(cell.get("source", [])):
            load_idx = i
            break

    # We will replace cells around load_idx with a new function and usage.
    # Current indices:
    # 16: # Load Dataframes
    # 17: stations_to_keep = ...
    # 18: file loading and df operations

    new_cells = [
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "def load_data(file_pattern, stations_to_keep=None):\n",
                "    \"\"\"\n",
                "    Loads and concatenates Beijing Air Quality dataset CSVs.\n",
                "    \n",
                "    Args:\n",
                "        file_pattern (str): Glob pattern to find the CSV files.\n",
                "        stations_to_keep (list, optional): List of station names to keep.\n",
                "                                           If None, loads all stations.\n",
                "                                           \n",
                "    Returns:\n",
                "        pd.DataFrame: A combined dataframe sorted by station and time, with \n",
                "                      datetime and engineered temporal features.\n",
                "    \"\"\"\n",
                "    file_paths = glob.glob(file_pattern)\n",
                "    \n",
                "    if stations_to_keep:\n",
                "        dataframes = [pd.read_csv(file) for file in file_paths if any(sub in file for sub in stations_to_keep)]\n",
                "    else:\n",
                "        dataframes = [pd.read_csv(file) for file in file_paths]\n",
                "        \n",
                "    df = pd.concat(dataframes, ignore_index=True)\n",
                "    \n",
                "    # Convert categorical variables\n",
                "    df['wd'] = df['wd'].astype('category')\n",
                "    df['station'] = df['station'].astype('category')\n",
                "\n",
                "    # Create Temporal features\n",
                "    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])\n",
                "    df['DayOfWeek'] = df['datetime'].dt.dayofweek\n",
                "    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)\n",
                "    df['Is_Rush_Hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)\n",
                "\n",
                "    # Sort by station and time\n",
                "    df = df.sort_values(by=['station', 'datetime']).reset_index(drop=True)\n",
                "    \n",
                "    return df\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "stations_to_keep = [\n",
                "  'Aotizhongxin',\n",
                "  'Dongsi',\n",
                "  # 'Guanyuan',\n",
                "  # 'Nongzhanguan',\n",
                "  # 'Tiantan',\n",
                "  # 'Wanliu',\n",
                "  # 'Wanshouxigong'\n",
                "]\n",
                "\n",
                "file_pattern = \"./PRSA_Data_20130301-20170228/PRSA_Data_*.csv\"\n",
                "df = load_data(file_pattern, stations_to_keep)\n",
                "date_time = df['datetime']\n"
            ]
        }
    ]

    # Replace cells 17 and 18 with the new cells
    nb["cells"][17] = new_cells[0]
    nb["cells"][18] = new_cells[1]

    with open("beijing_air_quality.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

apply_refactor()
