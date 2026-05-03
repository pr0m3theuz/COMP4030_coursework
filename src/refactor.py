import json
import re

def process_notebook(filepath, out_filepath):
    with open(filepath, "r") as f:
        nb = json.load(f)

    # We will just print out where functions might be useful and the general flow.
    # We can also add a seed setting function at the start.

process_notebook("beijing_air_quality.ipynb", "beijing_air_quality_refactored.ipynb")
