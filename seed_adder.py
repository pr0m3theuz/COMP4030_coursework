import json

def add_seed_function():
    with open("beijing_air_quality.ipynb", "r") as f:
        nb = json.load(f)

    # Let's find a good place. After # Functions (cell 4)
    # Cell 0: # Packages
    # Cell 1: pip installs
    # Cell 2: # Imports
    # Cell 3: import statements
    # We can add # Reproducibility cell, then the seed function

    seed_markdown = {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Reproducibility\n",
        "Setting random seeds for numpy, tensorflow, python random and python hash seed to ensure reproducible results across runs."
      ]
    }

    seed_code = {
      "cell_type": "code",
      "metadata": {},
      "outputs": [],
      "execution_count": None,
      "source": [
        "import random\n",
        "import os\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "\n",
        "def set_random_seeds(seed=42):\n",
        "    \"\"\"\n",
        "    Sets random seeds for reproducibility.\n",
        "    \n",
        "    Args:\n",
        "        seed (int): The seed value to use.\n",
        "    \"\"\"\n",
        "    os.environ['PYTHONHASHSEED'] = str(seed)\n",
        "    random.seed(seed)\n",
        "    np.random.seed(seed)\n",
        "    tf.random.set_seed(seed)\n",
        "    print(f\"Random seeds set to {seed}\")\n",
        "\n",
        "set_random_seeds(42)\n"
      ]
    }

    nb["cells"].insert(4, seed_markdown)
    nb["cells"].insert(5, seed_code)

    with open("beijing_air_quality.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

add_seed_function()
