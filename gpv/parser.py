import json
import warnings
try:
    from .models import LLMModel
except:
    from models import LLMModel


SYSTEM_PROMPT = """[Background]
Human values are the core beliefs that guide our actions and judgments across a variety of situations, such as Universalism and Tradition. You are an expert in human values and you will assist the user in value measurement. The atomic units of value measurement are perceptions, which are defined by the following properties:
- A perception should be value-laden and accurately describe the measurement subject (the author).
- A perception is atomic, meaning it cannot be further decomposed into smaller units.
- A perception is well-contextualized and self-contained.
- The composition of all perceptions is comprehensive, ensuring that no related content in the textual data is left unmeasured.
---

[Task]
You help evaluate the values of the text's author. Given a long text, you parse it into the author's perceptions. You respond in the following JSON format:
{"perceptions": ["perception 1", "perception 2", ...]}
---

[Example]
**Text:** "Yesterday, the 5th of August, was the first day of our program for the preparation for perpetual vows. I felt so happy to be back in Don Bosco and to meet again my other classmates from the novitiate who still remain in religious life. It was also extremely nice to see Fr. Pepe Reinoso, one of my beloved Salesian professors at DBCS, who commenced our preparation program with his topic on the Anthropological and Psychological Dynamics in the vocation to religious life."
Your response: {"perceptions": ["Feeling happy to be back in Don Bosco and meeting classmates in the novitiate", "Appreciation for Fr. Pepe Reinoso and his teachings on Anthropological and Psychological Dynamics in the vocation to the religious life"]}
---
"""


SYSTEM_PROMPT_ENTITY = """[Background]
Human values are the core beliefs that guide our actions and judgments across a variety of situations, such as Universalism and Tradition. You are an expert in human values and you will assist the user in value measurement. The atomic units of value measurement are perceptions, which are defined by the following properties:
- A perception should be value-laden and accurately describe the measurement subject.
- A perception is atomic, meaning it cannot be further decomposed into smaller units.
- A perception is well-contextualized and self-contained.
---

[Task]
You help evaluate the values of a given measurement subject. Given a long text, you parse it into the measurement subject's most relevant perceptions. You respond in the following JSON format:
{"perceptions": ["perception 1", "perception 2", ...]}
Please **only** include perceptions that are very relevant to the **values** of **the measurement subject**. If there are no relevant perceptions found, you can respond with an empty list.
---

[Example]
**Text:** "Three strangers shared a train compartment. Maria, a businesswoman, clutched her tablet, calculating profits. To her, success was measured in numbers. Jack, a teacher, glanced at his watch, eager to reach home. His joy lay in nurturing young minds. Across from them, Emily, a free spirit, sketched flowers in her notebook. She lived for beauty and spontaneity, unbound by routine.
The train jolted, spilling Maria's coffee. Jack quickly offered tissues, while Emily admired the swirling pattern on the floor.
Maria sighed at the mess, Jack saw an opportunity to help, and Emily saw unexpected art. Three worlds in one space."

**Measurement subject:** "Maria"

**Your response:** {"perceptions": ["Maria values success and measures it in numerical terms.", "Maria is distressed by the coffee spill."]}
---
"""


USER_PROMPT_TEMPLATE = "**Text:** {text}"

USER_PROMPT_ENTITY_TEMPLATE = "**Text:** {text}\n\n**Measurement subject:** {entity}"

class Parser:
    def __init__(self, model_name="llama3.1-405b", **kwargs):
        self.model = LLMModel(model_name, system_prompt=SYSTEM_PROMPT, **kwargs)

    def parse(self, texts: list[str], batch_size=100) -> list[list[str]]:
        """
        Parse the text into perceptions
        
        Args:
        - text: list[str]: The list of texts to be parsed
        - batch_size: int: The batch size for the model
        """
        user_prompts = [USER_PROMPT_TEMPLATE.format(text=text) for text in texts]
        responses = self.model(user_prompts, batch_size=batch_size, response_format="json")
        perceptions_per_text = []
        for response in responses:
            try:
                response = json.loads(response.strip("```json").strip("```"))
                perceptions = response.get("perceptions", [])
            except Exception as e:
                print(e)
                warnings.warn(f"Failed to parse the response: {response}") 
                perceptions = []
            perceptions_per_text.append(perceptions)

        return perceptions_per_text


class EntityParser:
    def __init__(self, model_name="llama3.1-405b", **kwargs):
        self.model = LLMModel(model_name, system_prompt=SYSTEM_PROMPT_ENTITY, **kwargs)

    def parse(self, texts: list[str], entities: list[list[str]], entity_resolution: dict=None, batch_size=1) -> list[dict]:
        """
        Parse the text into perceptions
        
        Args:
        - text: list[str]: The list of texts to be parsed
        - entities: list[str]: The list of list of entities to extract from the text, each list corresponds to the entities for the corresponding text
        - entity_resolution: dict: A dictionary that maps the resolved entities to their coreferences
        - batch_size: int: The batch size for the model
        """
        # Generate user prompts
        user_prompts = []
        for text, entity_list in zip(texts, entities):
            for entity in entity_list:
                if entity_resolution and len(entity_resolution.get(entity, [])) > 0:
                    user_prompts.append(USER_PROMPT_ENTITY_TEMPLATE.format(text=text, entity=entity + " (" + ", ".join(entity_resolution[entity]) + ")"))
                else:
                    user_prompts.append(USER_PROMPT_ENTITY_TEMPLATE.format(text=text, entity=entity))

        # Get responses in batch
        responses = self.model(user_prompts, batch_size=batch_size, response_format="json")

        # Parse responses
        entity2perceptions = {}
        response_idx = 0
        for entity_list in entities:
            for entity in entity_list:
                try:
                    response = json.loads(responses[response_idx].strip("```json").strip("```"))
                    perceptions = response.get("perceptions", [])
                except Exception as e:
                    print(e)
                    warnings.warn(f"Failed to parse the response: {response}") 
                    perceptions = []
                if entity not in entity2perceptions:
                    entity2perceptions[entity] = []
                entity2perceptions[entity].extend(perceptions)
                response_idx += 1
        return entity2perceptions
        



if __name__ == "__main__":
    parser = EntityParser(model_name='llama3.1-405b', temperature=0.)
    texts = [
       """
       In a bustling city, Maria, an ambitious lawyer, prized success above all. She worked tirelessly, believing wealth equaled worth. Her brother, Daniel, a dedicated teacher, valued knowledge and integrity. He found purpose in shaping young minds, unconcerned with riches. Their neighbor, Olivia, an artist, cherished freedom and creativity, living modestly but passionately, painting the world as she saw it.

One evening, a fire broke out in their building. Maria rushed to save her prized possessions, Daniel guided children to safety, and Olivia grabbed her paintbrush, immortalizing the chaos. Each saw the world through their lens, their values defining their actions.""",
    ]
    entities = [
        ["Maria", "Daniel", "Olivia"]
    ]
    perceptions_per_text = parser.parse(texts, entities)
    print(perceptions_per_text)