import json

from gpv import GPV


path = "data/西游记-zh.txt"
with open(path, "r") as file:
    book = file.read()

gpv = GPV(measure_author=False)

measurement_subjects = ["唐僧", "悟空", "八戒", "沙僧"]
coref_resolve = {
    "唐僧": ["唐三藏", "师父"],
    "悟空": ["猴王", "行者"],
    "八戒": ["猪八戒", "猪悟能"],
    "沙僧": ["沙和尚", "沙悟净"],
}
values = ["Universalism", "Hedonism", "Achievement", "Power", "Security", "Self-Direction", "Stimulation", "Tradition", "Benevolence", "Conformity"]

results = gpv.measure_entities_rag(
    text=book,
    values=values,
    measurement_subjects=measurement_subjects,
    coref_resolve=coref_resolve,
    K=50,
    threshold=5,
    )

# Save the results
from datetime import datetime
save_path = "measurement_results_rag_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
with open(save_path, "w") as file:
    json.dump(results, file, indent=4)

print("Results saved to", save_path)
