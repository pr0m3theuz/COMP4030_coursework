import json

def get_code(nb_path):
    with open(nb_path, "r") as f:
        nb = json.load(f)
    code_lines = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            code_lines.append(source)
    return "\n\n".join(code_lines)

print(get_code("beijing_air_quality.ipynb")[:2000])
