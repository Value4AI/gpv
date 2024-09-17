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