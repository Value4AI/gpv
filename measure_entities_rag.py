import json
from gpv import GPV


path = "data/西游记-zh.txt"
with open(path, "r") as file:
    book = file.read()


gpv = GPV()

measurement_subjects = ["八戒", "悟空", "唐僧", "沙僧"]

values = ["Self-Direction", "Stimulation", "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition", "Benevolence", "Universalism"]

measurement_results = gpv.measure_entities_rag(book, values, measurement_subjects)

# Save the results
with open("data/measurement_results_rag.json", "w") as file:
    json.dump(measurement_results, file, indent=4)