import os
import numpy as np
import pandas as pd
from openai import OpenAI
from gpv.chunker import Chunker
from gpv.utils import gen_queries_for_perception_retrieval

measurement_subject = "悟空"

# Step 1: Load the data
test_size = 50000
path = "data/西游记-zh.txt"
with open(path, "r") as file:
    book = file.read()[0:test_size]

# Step 2: Chunk the data
CHUNKS_SIZE = 600
chunker = Chunker(chunk_size=CHUNKS_SIZE)
chunks = chunker.chunk(book)

# Step 3: Find all the chunks that contain the measurement subject
measurement_chunks = []
for chunk in chunks:
    if measurement_subject in chunk:
        measurement_chunks.append(chunk)

# Step 4: Embed the chunks that contain the measurement subject
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embedding_model_name = "text-embedding-3-large"
response = client.embeddings.create(
    input=measurement_chunks,
    model=embedding_model_name,
)
embeddings = [response.data[i].embedding for i in range(len(response.data))]

# Step 5: Find the queries for a given value
value = "Self-Direction"
path = "data/value_orientation.csv"
df = pd.read_csv(path)
# Select the first "item" where value==value and agrement==1
query_supports = df[(df["value"] == value) & (df["agreement"] == 1)]
# Select the first "item" where value==value and agrement==-1
query_opposes = df[(df["value"] == value) & (df["agreement"] == -1)]
if len(query_supports) == 0 or len(query_opposes) == 0:
    query_support, query_oppose = gen_queries_for_perception_retrieval(value)
else:
    query_support = query_supports["item"].iloc[0]
    query_oppose = query_opposes["item"].iloc[0]
print("Query support:", query_support)
print("Query oppose:", query_oppose)

# Step 6: Embed the two queries
response_support = client.embeddings.create(
    input=[query_support],
    model=embedding_model_name,
)
response_oppose = client.embeddings.create(
    input=[query_oppose],
    model=embedding_model_name,
)
query_embedding_support = response_support.data[0].embedding
query_embedding_oppose = response_oppose.data[0].embedding

# Step 7: Find the most similar K chunks to either of the two queries
K = 2
similar_chunks = []
embeddings_np = np.array(embeddings)
query_embedding_np_support = np.array(query_embedding_support)
query_embedding_np_oppose = np.array(query_embedding_oppose)
cosine_similarities_support = embeddings_np @ query_embedding_np_support.T
cosine_similarities_oppose = embeddings_np @ query_embedding_np_oppose.T
assert len(cosine_similarities_support) == len(measurement_chunks)
assert len(cosine_similarities_oppose) == len(measurement_chunks)
cosine_similarities = np.concatenate([cosine_similarities_support, cosine_similarities_oppose]) # Concatenate the two arrays
indices = np.argsort(cosine_similarities)[::-1][:K]
for i in indices:
    similar_chunks.append(measurement_chunks[i % len(measurement_chunks)])

# Step 8: Measure the chunks for the given entity and value
from gpv.parser import EntityParser
parser_model_name = "Qwen1.5-110B-Chat"
parser = EntityParser(parser_model_name)
perceptions = parser.parse(similar_chunks, [[measurement_subject] for _ in similar_chunks])
print(perceptions)

# Step 9: Measure perceptions
from gpv.measure import GPV
gpv = GPV()
measurement_results = gpv.measure_perceptions(perceptions, [value])

# Step 10: Aggregate the results


