from pprint import pprint

from gpv import GPV


gpv = GPV()

path = "data/西游记人物简介.txt"

with open(path, "r") as f:
    data = f.read()

values = ["Universalism", "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition", "Benevolence", "Stimulation", "Self-Direction"]

with open(path, 'r') as file:
    text = file.read()

results = gpv.measure_entities(text, values, is_zh=True)
pprint(results)