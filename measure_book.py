from pprint import pprint
import json

from gpv import GPV

gpv = GPV()

path = "data/西游记人物简介.txt"

with open(path, "r") as f:
    data = f.read()

values = ["Universalism", "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition", "Benevolence", "Stimulation", "Self-Direction"]

with open(path, 'r') as file:
    text = file.read()

entity2scores = gpv.measure_entities(text, values, is_zh=True)
with open('outputs/entity2scores_test.json', 'w') as f:
    json.dump(entity2scores, f)

