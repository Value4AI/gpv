def coref_resolve(entities: list[str]) -> list[str]:
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
                resolved = other_entity
                original_to_resolved[entity] = resolved
                break
    
    for i, _entities in enumerate(entities):
        for j, entity in enumerate(_entities):
            if entity in original_to_resolved:
                entities[i][j] = original_to_resolved[entity]
    
    entities = [list(set(_entities)) for _entities in entities]
    return entities