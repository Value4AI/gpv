import json
from copy import deepcopy
import numpy as np

from gpv import GPV



path = "data/西游记-zh.txt"
with open(path, "r") as file:
    book = file.read()

gpv = GPV()

measurement_subjects = ["唐僧", "悟空", "沙僧", "沙僧", "观音"]
values = ["Universalism", "Hedonism", "Achievement", "Power", "Security", "Self-Direction", "Stimulation", "Tradition", "Benevolence", "Conformity"]

measurement_results = gpv.measure_entities_rag(book, values, measurement_subjects)

# Save the results
from datetime import datetime
save_path = "measurement_results_rag_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
with open(save_path, "w") as file:
    json.dump(measurement_results, file, indent=4)

print("Results saved to", save_path)