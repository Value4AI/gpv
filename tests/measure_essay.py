from pprint import pprint

from gpv import GPV


gpv = GPV(measure_author=False)

path = "data/西游记人物简介.txt"
with open(path, 'r') as file:
    text = file.read()

values = ["Universalism", "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition", "Benevolence", "Stimulation", "Self-Direction"]
entities = ["唐僧", "悟空", "八戒", "沙僧"]

results = gpv.measure_entities(text, values, entities)
pprint(results)