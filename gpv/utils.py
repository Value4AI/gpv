from transformers import pipeline
try:
    from .models import LLMModel
except ImportError:
    from models import LLMModel

def coref_resolve_simple(entities: list[str]) -> list[str]:
    """
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

def coref_resolve_llm(entities: list[str], model="gemma-7b") -> list[str]:
    """
    A resolution method using LLM.
    
    Args:
        entities (list[list[str]]): A list of list of entities.
    """
    # Deduplicate the entities
    flatten_entities = list(set([entity for sublist in entities for entity in sublist])) # list[str]
    
    # TODO: Implement the coreference resolution using LLM


class Translator:
    def __init__(self):
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
    
    def translate(self, texts: list[str]) -> list[str]:
        results = self.translator(texts, return_text=True)
        translated = [result['translation_text'] for result in results]
        return translated


if __name__ == "__main__":
    from chunker import Chunker
    from ner import NER
    
    path = "../data/西游记人物简介.txt"

    with open(path, "r") as f:
        data = f.read()
    print(data)
    
    chunker = Chunker(chunk_size=600)
    chunks = chunker.chunk([data])[0]
    
    translator = Translator()
    chunks = translator.translate(chunks)
    
    print(len(chunks))
    
    ner = NER()
    entities = ner.extract_persons_from_texts(chunks)
    entities = coref_resolve_simple(entities)
    
    for chunk, _entities in zip(chunks, entities):
        print(chunk)
        print(_entities)
        print('-'*50)