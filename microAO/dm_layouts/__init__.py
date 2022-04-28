import pathlib
import json
import copy

# Check local files
_layouts = {}
for layout_file in pathlib.Path(__file__).parent.iterdir():
    # Skip non-JSON files
    if layout_file.suffix != ".json":
        continue
    # Open the file for reading
    with layout_file.open("r", encoding="utf-8") as fi:
        _layouts[layout_file.stem] = json.load(fi)

def get_layout(name):
    if name not in _layouts:
        raise Exception(f"A layout with name '{name}' does not exist.")
    return copy.deepcopy(_layouts[name])