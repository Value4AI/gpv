from pprint import pprint

from gpv.chunker import Chunker
from gpv.summarizer import Summarizer
from gpv.ner import NER
from gpv.parser import EntityParser
from gpv.measure import GPV

from gpv.utils import coref_resolve
from gpv.measure import get_score

path = "data/The Road to Evergreen.txt"

values = ["Universalism", "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition", "Benevolence", "Stimulation", "Self-Direction"]

with open(path, 'r') as file:
    text = file.read()

# Step 1: chunking (larger chunks, about 400 tokens)
chunker = Chunker(chunk_size=200)
chunks = chunker.chunk([text])[0]
print(len(chunks))
print('-'*50)

# Step 3: entity extraction
ner = NER()
entities = ner.extract_persons_from_texts(chunks) # list[list[str]]
# Resolve coreferences
entities = coref_resolve(entities)
pprint(entities)
print('-'*50)

# Step 4: parsing for each entity
parser = EntityParser(model_name="gemma-7b")
perceptions = parser.parse(chunks, entities)
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

# Step 5: measuring for each entity
all_perceptions = []
for entity, perceptions in entity2perceptions.items():
    all_perceptions.extend(perceptions)

gpv = GPV()
measurement_results = gpv.measure_perceptions(all_perceptions, values)

pprint(measurement_results)
print('-'*50)

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