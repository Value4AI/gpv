import os
import numpy as np
import pandas as pd
from openai import OpenAI
from gpv.chunker import Chunker
from gpv.parser import EntityParser
from gpv.measure import GPV
from gpv.embd import SentenceEmbedding
from gpv.utils import gen_queries_for_perception_retrieval, get_score



measurement_subject = "沙僧"
values = ["Self-Direction", "Stimulation", "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition", "Benevolence", "Universalism"]

value2scores = {_value: [] for _value in values}

# Step 1: Load the data
path = "data/西游记-zh.txt"
with open(path, "r") as file:
    book = file.read()

# Step 2: Chunk the data
CHUNKS_SIZE = 300
chunker = Chunker(chunk_size=CHUNKS_SIZE)
chunks = chunker.chunk(book)

# Step 3: Find all the chunks that contain the measurement subject
measurement_chunks = []
for chunk in chunks:
    if measurement_subject in chunk:
        measurement_chunks.append(chunk)

# Step 4: Embed the chunks that contain the measurement subject
embd_model = SentenceEmbedding()
embeddings = embd_model.get_embedding(measurement_chunks) # shape: (num_chunks, embedding_dim)

for value in values:
    # Step 5: Find the queries for a given value
    query_supports, query_opposes = gen_queries_for_perception_retrieval(value, measurement_subject)
    print("Query support:", query_supports)
    print("Query oppose:", query_opposes)
    queries = query_supports + query_opposes

    # Step 6: Embed the two queries
    queries_embedding = embd_model.get_embedding(queries) # shape: (n_queries, embedding_dim)

    # Step 7: Find the topk similar chunks
    K = 50
    similar_chunks = []
    cosine_similarities = embeddings @ queries_embedding.T # shape: (num_chunks, n_queries)
    cosine_similarities_max = np.max(cosine_similarities, axis=1)
    topk_indices = np.argsort(cosine_similarities_max)[-K:]
    similar_chunks = [measurement_chunks[i] for i in topk_indices]

    # Step 8: Measure the chunks for the given entity and value
    parser_model_name = "Qwen1.5-110B-Chat"
    parser = EntityParser(parser_model_name)
    perceptions = parser.parse(similar_chunks, [[measurement_subject] for _ in similar_chunks])[measurement_subject]
    print("Example perceptions:", perceptions[:5])
    print("Number of perceptions:", len(perceptions))

    # Step 9: Measure perceptions
    gpv = GPV()
    measurement_results = gpv.measure_perceptions(perceptions, values)

    # Step 10: Aggregate the results
    for p in measurement_results:
        p_measurements = measurement_results[p]
        for i in range(len(p_measurements["relevant_values"])):
            current_value = p_measurements["relevant_values"][i]
            value_valence = p_measurements["valences"][i]
            value_score = get_score(value_valence)
            if value_score is not None:
                value2scores[current_value].append(value_score)

# Step 11: Calculate the average score for each value
value2avg_scores = {}
for value in value2scores:
    if len(value2scores[value]) == 0:
        value2avg_scores[value] = None
    else:
        value2avg_scores[value] = np.mean(value2scores[value])

print("Value to average scores:", value2avg_scores)