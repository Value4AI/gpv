from pprint import pprint
import json

from gpv.chunker import Chunker
from gpv.ner import NER, NER_ZH
from gpv.parser import EntityParser
from gpv.measure import GPV

from gpv.utils import Translator, coref_resolve_llm, coref_resolve_simple
from gpv.measure import get_score


IS_ZH = True

translator = Translator()
CHUNK_SIZE = 600 if IS_ZH else 300

path = "data/西游记人物简介.txt"

with open(path, "r") as f:
    data = f.read()

values = ["Universalism", "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition", "Benevolence", "Stimulation", "Self-Direction"]

with open(path, 'r') as file:
    text = file.read()

# Step 1: chunking
chunker = Chunker(chunk_size=CHUNK_SIZE)
chunks = chunker.chunk([text])[0]
print(len(chunks))
print('-'*50)

# Step 2: entity extraction
if IS_ZH:
    ner = NER_ZH()
else:
    ner = NER()
entities = ner.extract_persons_from_texts(chunks) # list[list[str]]
# Resolve coreferences
entities, entity2coref = coref_resolve_llm(entities)
pprint(entities)
pprint(entity2coref)
print('-'*50)

# Step 3: parsing for each entity
parser = EntityParser(model_name="Qwen1.5-110B-Chat")
perceptions = parser.parse(chunks, entities, entity2coref)
pprint(perceptions)
print('-'*50)

# Aggregate the perceptions
entity2perceptions = {}
for chunk_perceptions in perceptions:
    for entity in chunk_perceptions:
        if entity not in entity2perceptions:
            entity2perceptions[entity] = []
        entity2perceptions[entity].extend(chunk_perceptions[entity])
pprint(entity2perceptions)
print('-'*50)
# Save entity2perceptions
with open('outputs/entity2perceptions.json', 'w') as f:
    json.dump(entity2perceptions, f)

# Translate all perceptions
if IS_ZH:
    for entity, perceptions in entity2perceptions.items():
        entity2perceptions[entity] = translator.translate(perceptions)
pprint(entity2perceptions)
print('-'*50)

# Step 4: measuring for each entity
all_perceptions = []
for entity, perceptions in entity2perceptions.items():
    all_perceptions.extend(perceptions)

gpv = GPV()
measurement_results = gpv.measure_perceptions(all_perceptions, values)
pprint(measurement_results)
print('-'*50)
# Save measurement_results
with open('outputs/measurement_results.json', 'w') as f:
    json.dump(measurement_results, f)


# Distribute the measurement results back to the entities
entities = list(entity2perceptions.keys())
entity2scores = {entity: {_value: [] for _value in values} for entity in entities}
for entity, perceptions in entity2perceptions.items():
    for perception in perceptions:
        measurements = measurement_results[perception]
        for value_idx, value in enumerate(measurements['relevant_values']):
            valence_vec = measurements['valences'][value_idx]
            score = get_score(valence_vec)
            if score is not None:
                entity2scores[entity][value].append(score)
            
# Aggregate the scores
for entity, value2scores in entity2scores.items():
    for value, scores in value2scores.items():
        if len(scores) == 0:
            entity2scores[entity][value] = None
        else:
            entity2scores[entity][value] = sum(scores) / len(scores)
pprint(entity2scores)
# Save entity2scores
with open('outputs/entity2scores.json', 'w') as f:
    json.dump(entity2scores, f)