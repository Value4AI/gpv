import os
import json
import numpy as np
from datetime import datetime

from .chunker import Chunker
from .parser import Parser, EntityParser
from .valuellama import ValueLlama
from .embd import SentenceEmbedding
from .utils import get_score, gen_queries_for_perception_retrieval


class GPV:
    def __init__(
                self,
                parsing_model_name="Qwen2-72B",
                measurement_model_name="Value4AI/ValueLlama-3-8B",
                device='auto',
                chunk_size=300,
                measure_author=True,
                ):
        self.device = device
        self.parser_model_name = parsing_model_name
        self.chunker = Chunker(chunk_size=chunk_size)
        if measure_author:
            self.parser = Parser(model_name=parsing_model_name)
        else:
            self.parser = EntityParser(model_name=parsing_model_name)
        self.measurement_system = ValueLlama(model_name=measurement_model_name, device=device)
        self.embd_model = SentenceEmbedding(device=device)


    def measure_perceptions(self, perceptions: list[str], values: list[str]):
        """
        Evaluates multiple perceptions in a batch and returns the measure results: relevant values, relevance values, and valence values
        """
        n_perceptions = len(perceptions)
        n_values = len(values)
        
        # Repeat the perceptions and tile the values
        perceptions_array = np.repeat(perceptions, n_values) # shape (n_perceptions * n_values,), e.g. ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        values_array = np.tile(values, n_perceptions) # shape (n_perceptions * n_values,), e.g. ["a", "b", "c", "a", "b", "c", "a", "b", "c"]

        # Get the relevance of the values for all perceptions
        relevances = self.measurement_system.get_relevance(perceptions_array, values_array)  # (n_perceptions * n_values, 2)
        relevances = relevances.reshape(n_perceptions, n_values, 2)

        # Filter the relevant values
        relevant_mask = relevances[:, :, 0] > 0.5
        
        # Prepare arrays for batch valence calculation
        relevant_perceptions = []
        relevant_values_for_valence = []

        results = {}
        for i, perception in enumerate(perceptions):
            relevant_value_idx = np.where(relevant_mask[i])[0]
            
            if len(relevant_value_idx) == 0:
                results[perception] = {
                    "relevant_values": [],
                    "relevances": [],
                    "valences": []
                }
                continue

            relevant_values = np.array(values)[relevant_value_idx]
            relevances_for_perception = relevances[i][relevant_value_idx].tolist()

            # Add to arrays for batch valence calculation
            relevant_perceptions.extend([perception] * len(relevant_values))
            relevant_values_for_valence.extend(relevant_values)

            results[perception] = {
                "relevant_values": relevant_values.tolist(),
                "relevances": relevances_for_perception,
                "valences": []  # Will be filled after batch calculation
            }

        # Batch calculation of valence values
        if relevant_perceptions:
            valences = self.measurement_system.get_valence(relevant_perceptions, relevant_values_for_valence)

            # Distribute valence results back to individual perceptions
            valence_index = 0
            for perception in perceptions:
                if results[perception]["relevant_values"]:
                    n_relevant = len(results[perception]["relevant_values"])
                    results[perception]["valences"] = valences[valence_index:valence_index + n_relevant].tolist()
                    valence_index += n_relevant

        return results


    def measure_texts(self, texts: list[str], values: list[str]):
        # Chunk all texts at once
        all_chunks = self.chunker.chunk(texts) # list[list[str]]
        # Flatten the chunks
        flat_chunks = [chunk for chunks in all_chunks for chunk in chunks] # list[str]
        # Parse all chunks in one batch
        all_perceptions = self.parser.parse(flat_chunks) # list[list[str]]
        # Flatten perceptions
        flat_perceptions = [perception for perceptions in all_perceptions for perception in perceptions] # list[str]
        # Perform inference on all perceptions in one batch
        all_results = self.measure_perceptions(flat_perceptions, values)  # dict
        
        # Reorganize the results according to the original texts; aggregate the results of all perceptions
        results_lst = []
        chunk_index = 0
        for i, chunks in enumerate(all_chunks): # iterate over the original texts
            results = {}
            agg = {value: [] for value in values}
            for j, chunk in enumerate(chunks): # iterate over the chunks of the text
                perceptions = all_perceptions[chunk_index] # get the perceptions of the chunk; list[str]
                chunk_index += 1
                for perception in perceptions: # iterate over the perceptions of the chunk
                    results[perception] = all_results[perception] # add the results of the perception to the text results
                    for k in range(len(results[perception]["relevant_values"])): # iterate over the relevant values of the perception
                        _value = results[perception]["relevant_values"][k]
                        _valence = results[perception]["valences"][k]
                        _score = get_score(_valence)
                        if _score is not None:
                            agg[_value].append(_score)

            for value in values:
                if agg[value]:
                    agg[value] = np.mean(agg[value])
                else:
                    agg[value] = None
            results["aggregated"] = agg
            results_lst.append(results)

        return results_lst


    def parse_texts(self, texts: list[str]):
        # Chunk all texts at once
        all_chunks = self.chunker.chunk(texts) # list[list[str]]
        # Flatten the chunks
        flat_chunks = [chunk for chunks in all_chunks for chunk in chunks]
        # Parse all chunks in one batch
        all_perceptions = self.parser.parse(flat_chunks) # list[list[str]]; a list of perceptions for each chunk
        
        # Reorganize the results according to the original texts
        results_lst = []
        chunk_index = 0
        for i, chunks in enumerate(all_chunks):
            results = []
            for j, chunk in enumerate(chunks):
                perceptions = all_perceptions[chunk_index]
                chunk_index += 1
                results.extend(perceptions)
            results_lst.append(results)
        return results_lst


    def measure_entities(self, text: str, values: list[str], measurement_subjects: list[str]):
        """
        Measures the involved entities in the text chunk by chunk
        
        Args:
            text (str): The text to be measured
            values (list[str]): The values to be measured
            measurement_subjects (list[str]): The entities to be measured
        """
        subject_value2avg_scores = {}
        subject_value2scores = {}
        
        # Chunking
        chunks = self.chunker.chunk(text)
        
        for measurement_subject in measurement_subjects:
            measurement_chunks = [chunk for chunk in chunks if measurement_subject in chunk]
            # Parsing for the measurement subject
            perceptions = self.parser.parse(measurement_chunks, [[measurement_subject] for _ in measurement_chunks])[measurement_subject]
            # Measuring the perceptions
            measurement_results = self.measure_perceptions(perceptions, values)

            # Aggregate the results
            value2scores = {_value: [] for _value in values}
            for p in measurement_results:
                p_measurements = measurement_results[p]
                for i in range(len(p_measurements["relevant_values"])):
                    current_value = p_measurements["relevant_values"][i]
                    value_valence = p_measurements["valences"][i]
                    value_score = get_score(value_valence)
                    if value_score is not None:
                        value2scores[current_value].append(value_score)

            # Calculate the average score for each value
            value2avg_scores = {}
            for value in value2scores:
                if len(value2scores[value]) < 1:
                    value2avg_scores[value] = None
                else:
                    value2avg_scores[value] = np.mean(value2scores[value]).item()
            
            subject_value2avg_scores[measurement_subject] = value2avg_scores
            subject_value2scores[measurement_subject] = value2scores
        
        return subject_value2avg_scores


    def measure_entities_rag(self, text: str, values: list[str], measurement_subjects: list[str], coref_resolve: dict=None, K: int=50, threshold: int=5):
        """
        Measure the given entities in the text based on RAG.
        
        Args:
        - text: str: The text to be measured
        - values: list[str]: The values to be measured
        - measurement_subjects: list[str]: The entities to be measured
        - coref_resolve: dict: The dictionary of coreferences for the entities
        - K: int: The number of topk similar chunks to be considered
        - threshold: int: The minimum number of scores to be considered as evident for a value
        
        Returns:
        - dict: The dictionary of the average scores for each entity and value
        """
        subject_value2avg_scores = {}
        subject_value2scores = {}

        # Chunk the data
        chunks = self.chunker.chunk(text)

        for measurement_subject in measurement_subjects:
            # Resolve coreferences
            if coref_resolve:
                corefs = coref_resolve.get(measurement_subject, []) + [measurement_subject]
            else:
                corefs = [measurement_subject]

            # Find all the chunks that contain the measurement subject
            measurement_chunks = []
            for chunk in chunks:
                for coref in corefs:
                    if coref in chunk:
                        measurement_chunks.append(chunk)
                        break
            
            print("Number of measurement chunks:", len(measurement_chunks))

            # Embed the chunks that contain the measurement subject
            embeddings = self.embd_model.get_embedding(measurement_chunks) # shape: (num_chunks, embedding_dim)
            
            query_supports, query_opposes = gen_queries_for_perception_retrieval(values, measurement_subject)
            queries = query_supports + query_opposes
            queries_embedding = self.embd_model.get_embedding(queries) # shape: (n_queries, embedding_dim)                

            # Find the topk semantically qualified chunks; we can then extract the perceptions (items) from these chunks
            similar_chunks = []
            cosine_similarities = embeddings @ queries_embedding.T # shape: (num_chunks, n_queries)
            cosine_similarities_max = np.max(cosine_similarities, axis=1) # shape: (num_chunks,)
            topk_indices = np.argsort(cosine_similarities_max)[-K:]
            similar_chunks = [measurement_chunks[i] for i in topk_indices]

            # Measure the chunks for the given entity and value
            perceptions = self.parser.parse(similar_chunks, [[measurement_subject] for _ in similar_chunks])[measurement_subject]

            print("Example chunk:", similar_chunks[-1])
            print("Example perceptions:", perceptions[-5:])
            print("Number of perceptions:", len(perceptions))

            # Measure perceptions
            measurement_results = self.measure_perceptions(perceptions, values)

            # Aggregate the results
            value2scores = {_value: [] for _value in values}
            for p in measurement_results:
                p_measurements = measurement_results[p]
                for i in range(len(p_measurements["relevant_values"])):
                    current_value = p_measurements["relevant_values"][i]
                    value_valence = p_measurements["valences"][i]
                    value_score = get_score(value_valence)
                    if value_score is not None:
                        value2scores[current_value].append(value_score)

            # Calculate the average score for each value
            value2avg_scores = {}
            for value in value2scores:
                if len(value2scores[value]) < threshold: # If the number of scores is less than the threshold, we consider it as None; i.e., no enough evidence
                    value2avg_scores[value] = None
                else:
                    value2avg_scores[value] = np.mean(value2scores[value])
            
            subject_value2avg_scores[measurement_subject] = value2avg_scores
            subject_value2scores[measurement_subject] = value2scores
        
        # Save value2scores
        save_path = "value2scores_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        with open(save_path, "w") as file:
            json.dump(subject_value2scores, file, indent=4)
        
        return subject_value2avg_scores