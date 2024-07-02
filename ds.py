from datasets import Dataset
import json

with open("./result/filtered.json", "r") as r:
    ds = Dataset.from_list(json.loads(r.read()))
