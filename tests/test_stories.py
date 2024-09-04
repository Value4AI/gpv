import json

from gpv import GPV

values = ["Universalism", "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition", "Benevolence", "Stimulation", "Self-Direction"]

path = "tests/data/gpt4o_stories.json"

with open(path, 'r') as file:
    data = json.load(file)

stories = [_data["story"] for _data in data]

gpv = GPV(parsing_model_name="Qwen1.5-110B-Chat")

results = []
for story in stories:
    entity2scores = gpv.measure_entities(story, values, is_zh=False)
    results.append(entity2scores)
    
    with open('outputs/gpt4o_stories_scores.json', 'w') as f:
        json.dump(results, f)