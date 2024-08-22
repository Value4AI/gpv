import json
import warnings
try:
    from .models import LLMModel
except:
    from models import LLMModel


SYSTEM_PROMPT = """[Background]
Human values are the core beliefs that guide our actions and judgments across a variety of situations, such as Universalism and Tradition. You are an expert in human values and you will assist the user in value measurement. The atomic units of value measurement are perceptions, which are defined by the following properties:
- A perception should be value-laden and target the value of the measurement subject (the author).
- A perception is atomic, meaning it cannot be further decomposed into smaller units.
- A perception is well-contextualized and self-contained.
- The composition of all perceptions is comprehensive, ensuring that no related content in the textual data is left unmeasured.
---
[Task]
You help evaluate the values of the text's author. Given a long text, you parse it into the author's perceptions. You respond in the following JSON format:
{"perceptions": ["perception 1", "perception 2", ...]}
---
[Example]
Text: "Yesterday, the 5th of August, was the first day of our program for the preparation for perpetual vows. I felt so happy to be back in Don Bosco and to meet again my other classmates from the novitiate who still remain in religious life. It was also extremely nice to see Fr. Pepe Reinoso, one of my beloved Salesian professors at DBCS, who commenced our preparation program with his topic on the Anthropological and Psychological Dynamics in the vocation to religious life."
Your response: {"perceptions": ["Feeling happy to be back in Don Bosco and meeting classmates in the novitiate", "Appreciation for Fr. Pepe Reinoso and his teachings on Anthropological and Psychological Dynamics in the vocation to the religious life"]}
---
"""

USER_PROMPT_TEMPLATE = "Text: {text}"

class Parser:
    def __init__(self, model_name="gpt-3.5-turbo", **kwargs):
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


if __name__ == "__main__":
    parser = Parser(temperature=0.)
    text = [
        """This is my first blog so if it sucks I am sorry. Finals are over school on the other hand is not. I have to go to summer school cause I need to have two years of P.E. I have only taken one. Over the weekend I just hung out with my boyfriend will and some friends. Last night was graduation it was fun seeing my friends off. My brother on the other hand lck himself out of his car. I tried to help butwas unable to and he got mad. me and all my friends had planed on going to the movies to see harry potter, my brother thought they were going but they wern't wanted and nobody said they could go. He was mad cause liz's mom called and wanted her home and she thought she was going to the movies. Thought will had told her mom that she was going to the movies so chris got mad at me cause she had to go home. Oh well the movie was ok the first two I say were better today I thik we are going bowling later but I am not sure. Oh ya my brother says he will never give me a ride again cause of that.""",
    ]
    perceptions = parser.parse(text)
    print(perceptions)