import json

def get_code(nb_path):
    with open(nb_path, "r") as f:
        nb = json.load(f)
    markdown_titles = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell.get("source", []))
            if source.startswith("#"):
                markdown_titles.append(f"Cell {i}: " + source.split("\n")[0])
    return markdown_titles

print("\n".join(get_code("beijing_air_quality.ipynb")))
