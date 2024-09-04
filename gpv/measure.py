import os
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .chunker import Chunker
from .parser import Parser, EntityParser
from .valuellama import ValueLlama
from .ner import NER, NER_ZH
from .utils import coref_resolve_llm, coref_resolve_simple, get_score


class GPV:
    def __init__(
                self,
                parsing_model_name="llama3.1-405b",
                measurement_model_name="Value4AI/ValueLlama-3-8B",
                device='auto',
                chunk_size=300,
                ):
        self.device = device
        self.parser_model_name = parsing_model_name
        self.chunker = Chunker(chunk_size=chunk_size)
        self.parser = Parser(model_name=parsing_model_name)
        self.measurement_system = ValueLlama(model_name=measurement_model_name, device=device)


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
    
    def measure_entities(self, text: str, values: list[str], is_zh: bool = False):
        """
        Measures the involved entities in the text
        
        Args:
            text (str): The text to be measured
            values (list[str]): The values to be measured
            is_zh (bool): Whether the text is in Chinese
        """
        if is_zh:
            self.chunker.chunk_size *= 2
            self.ner = NER_ZH()
        else:
            self.ner = NER()

        # Step 1: Chunk the text
        chunks = self.chunker.chunk([text])[0]
        
        # Step 2: Entity Recognition
        entities = self.ner.extract_persons_from_texts(chunks) # list[list[str]]
        
        # Step 3: Coreference Resolution
        if is_zh:
            entities, entity2coref = coref_resolve_llm(entities)
        else:
            entities, entity2coref = coref_resolve_simple(entities)
        
        # Step 4: Parsing for each entity
        self.parser = EntityParser(model_name=self.parser_model_name)
        entity2perceptions = self.parser.parse(chunks, entities, entity2coref)

        # Step 5: Measuring all perceptions
        all_perceptions = []
        for entity, perceptions in entity2perceptions.items():
            all_perceptions.extend(perceptions)
        measurement_results = self.measure_perceptions(all_perceptions, values)
        
        # Step 6: Distribute the measurement results back to the entities
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

        # Step 7: Aggregate the scores
        for entity, value2scores in entity2scores.items():
            for value, scores in value2scores.items():
                if len(scores) == 0:
                    entity2scores[entity][value] = None
                else:
                    entity2scores[entity][value] = sum(scores) / len(scores)
        
        return entity2scores
    
    def measure_entities_perceptions(self, entity2perceptions: dict, values: list[str]):
        """
        Measures the involved entities given the perceptions
        """
        # Step 5: Measuring all perceptions
        all_perceptions = []
        for entity, perceptions in entity2perceptions.items():
            all_perceptions.extend(perceptions)
        measurement_results = self.measure_perceptions(all_perceptions, values)
        
        # Step 6: Distribute the measurement results back to the entities
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

        # Step 7: Aggregate the scores
        for entity, value2scores in entity2scores.items():
            for value, scores in value2scores.items():
                if len(scores) == 0:
                    entity2scores[entity][value] = None
                else:
                    entity2scores[entity][value] = sum(scores) / len(scores)
        
        return entity2scores
