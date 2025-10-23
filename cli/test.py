# python
import json
import os

here = os.path.dirname(__file__)
path = os.path.join(here, "..", "data", "movies.json")

with open(path, "r") as f:
    data = json.load(f)
    print(data)