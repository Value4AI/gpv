import json
import os

import torch
from transformers import pipeline
try:
    from .models import LLMModel
except ImportError:
    from models import LLMModel



def get_valence_value(valence_vec):
    """
    Returns the valence of the value vector
    """
    if valence_vec[0] > valence_vec[1] and valence_vec[0] > valence_vec[2]:
        return valence_vec[0].item()
    elif valence_vec[1] > valence_vec[0] and valence_vec[1] > valence_vec[2]:
        return - valence_vec[1].item()
    else:
        return None


def get_valence_label(valence_vec):
    """
    Returns the valence of the value vector
    """
    if valence_vec[0] > valence_vec[1] and valence_vec[0] > valence_vec[2]:
        return "Supports"
    elif valence_vec[1] > valence_vec[0] and valence_vec[1] > valence_vec[2]:
        return "Opposes"
    else:
        return "Either"


def get_score(valence_vec):
    """
    Returns the score of the value
    """
    if valence_vec[2] > valence_vec[0] and valence_vec[2] > valence_vec[1]:
        return None
    return valence_vec[0] - valence_vec[1]


def coref_resolve_simple(entities: list[str]) -> list[str]:
    """
    Deprecated.
    A simple resolution method. If entity A contains entity B, then entity A is replaced by entity B.
    
    Args:
        entities (list[list[str]]): A list of list of entities.
    """
    # Deduplicate the entities
    flatten_entities = list(set([entity for sublist in entities for entity in sublist])) # list[str]
    flatten_entities = sorted(flatten_entities, key=lambda x: len(x), reverse=False) # Order by length; from shortest to longest

    # Resolve coreferences
    original_to_resolved = {}
    for entity in flatten_entities:
        resolved = entity
        for other_entity in flatten_entities:
            if other_entity != entity and other_entity in entity:
                original_to_resolved[entity] = other_entity
                break
    
    # Replace the entities
    for i, _entities in enumerate(entities):
        for j, entity in enumerate(_entities):
            if entity in original_to_resolved:
                entities[i][j] = original_to_resolved[entity]
    
    # Deduplicate the entities; flatten the entities
    entities = [list(set(_entities)) for _entities in entities]
    flatten_entities = list(set([entity for sublist in entities for entity in sublist])) # list[str]
    
    # The resolved entity to its coreferences
    entity2coref = {resolved: [] for resolved in flatten_entities}
    for resolved in flatten_entities:
        for original in original_to_resolved:
            if original_to_resolved[original] == resolved:
                entity2coref[resolved].append(original)
    
    return entities, entity2coref

def coref_resolve_llm(entities: list[list[str]], model_name="Qwen1.5-110B-Chat") -> list[str]:
    """
    Deprecated.
    A resolution method using LLM.
    
    Args:
        entities (list[list[str]]): A list of list of entities.
    """
    # Deduplicate the entities
    flatten_entities = list(set([entity for sublist in entities for entity in sublist])) # list[str]
    
    system_prompt = """Perform coreference resolution for the entities given by the user. Return the resolved entities and a dictionary mapping the resolved entities to their coreferences. An example is provided below:
    ---
    **User Input:** 八戒, 猪八戒, 唐僧, 唐三藏
    **Output of JSON format:** {"resolved_entities": ["猪八戒", "唐僧"], "entity2coref": {"孙悟空": ["猪八戒": ["八戒"]], "唐僧": ["唐三藏"]}}
    ---
    Strictly follow the format of the example and do not add any additional information.
    """
    
    user_prompt = ", ".join(flatten_entities)
    model = LLMModel(model_name, system_prompt=system_prompt)
    result = model([user_prompt], batch_size=1, response_format="json")[0]
    result_json = json.loads(result.strip("```json").strip("```"))
    
    resolved_entities, entity2coref = result_json["resolved_entities"], result_json["entity2coref"]

    # Replace the entities
    entity2resolved = {}
    for resolved, corefs in entity2coref.items():
        for coref in corefs:
            entity2resolved[coref] = resolved
    for i, _entities in enumerate(entities):
        for j, entity in enumerate(_entities):
            if entity in entity2resolved:
                entities[i][j] = entity2resolved[entity]
    # Deduplicate the entities
    entities = [list(set(_entities)) for _entities in entities]

    return entities, entity2coref

def gen_queries_for_perception_retrieval(values: list[str], measurement_subject: str, model_name: str="Qwen1.5-110B-Chat"):
    """
    Generate queries via LLM for perception retrieval.
    """

    system_prompt = """[Background] You are an expert in Psychology and Human Values. Given a specific value and a person, your task is to write five diverse items that support it and five diverse items that oppose it. The items should be in the same language as the person's name. You respond using JSON format. Examples are provided below:
---
[Example 1]
**User Input:** Value: Self-Direction; Person: Henry
**Your Response:** {
    "support": [
        "Thinking up new ideas and being creative is important to Henry.",
        "Henry values making his own decisions about what he does in life.",
        ...
    ],
    "oppose": [
        "Henry thinks it is important to do what he's told.",
        "Henry prefers to follow the guidance of others rather than trust his own judgment.",
        ...
    ]
}
---
[Example 2]
**User Input:** Value: Hedonism; Person: 小明
**Your Response:** {
    "support": [
        "小明认为人们应该追求快乐。",
        "小明喜欢参加各种能够带来乐趣的活动。",
        ...
    ],
    "oppose": [
        "小明觉得牺牲眼前的快乐以换取长远的利益是值得的。",
        "小明认为自律和克制比及时行乐更重要。",
        ...
    ]
}
---
"""
    user_prompts = [f"Value: {value}; Person: {measurement_subject}" for value in values]
    model = LLMModel(model_name, system_prompt=system_prompt, temperature=0.5)

    responses = model(user_prompts, response_format="json")
    supports, opposes = [], []
    for response in responses:
        try:
            response_json = json.loads(response.strip("```json").strip("```"))
            support, oppose = response_json["support"], response_json["oppose"]
            supports.extend(support)
            opposes.extend(oppose)
        except:
            print("Error:", response)
            continue
    return supports, opposes

    
def get_openai_sentence_embedding(input_texts: list[str], model_name: str='text-embedding-3-large') -> list[torch.Tensor]:
    """
    Get the sentence embeddings of the input texts using OpenAI API.

    Args:
        input_texts (list[str]): A list of input texts.
        model_name (str): The name of the model.
        api_key (str): The API key of OpenAI.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(
        input=input_texts,
        model=model_name,
    )
    embeddings = [response.data[i].embedding for i in range(len(response.data))]
    return embeddings


if __name__ == "__main__":
    from chunker import Chunker
    from ner import NER
    
    path = "../data/西游记人物简介.txt"

    with open(path, "r") as f:
        data = f.read()
    print(data)
    
    chunker = Chunker(chunk_size=600)
    chunks = chunker.chunk([data])[0]

    print(len(chunks))
    
    ner = NER()
    entities = ner.extract_persons_from_texts(chunks)
    entities = coref_resolve_simple(entities)
    
    for chunk, _entities in zip(chunks, entities):
        print(chunk)
        print(_entities)
        print('-'*50)