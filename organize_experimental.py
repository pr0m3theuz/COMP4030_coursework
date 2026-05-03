import json

def apply_refactor():
    with open("beijing_air_quality.ipynb", "r") as f:
        nb = json.load(f)

    # Wrap the rest of the cells after the core logic in an "Experiments / Unused Code" markdown
    exp_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Experiments & Unused Code\n",
            "---\n",
            "The following sections contain experimental workflows, alternative models, time-lag tests, and hyperparameter tuning code that were tested during the development phase."
        ]
    }

    # We'll insert this right before "## Not using" or similar. Let's find "## Not using"
    insert_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and "## Not using" in "".join(cell.get("source", [])):
            insert_idx = i
            break

    if insert_idx != -1:
        # replace the "## Not using" markdown
        nb["cells"][insert_idx] = exp_markdown

    with open("beijing_air_quality.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

apply_refactor()
