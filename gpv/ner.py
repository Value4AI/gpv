from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class NER:
    def __init__(self, model_name="dslim/bert-large-NER"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device_map="auto")
    
    def _extract_persons(self, ner_results):
        persons = []
        current_person = []

        for entity in ner_results:
            if entity['entity'] == 'B-PER':  # Start of a new person entity
                if current_person:
                    persons.append(''.join(current_person))
                current_person = [entity['word'].replace('##', '')]  # Start a new person name
            elif entity['entity'] == 'I-PER':  # Continuation of the person entity
                current_person.append(entity['word'].replace('##', ''))
        
        # Append the last person entity if any
        if current_person:
            persons.append(''.join(current_person))
        
        return persons
    
    def extract_entities(self, texts: list[str]) -> list[list[dict]]:
        if isinstance(texts, str):
            texts = [texts]
        return [self.nlp(text) for text in texts]
    
    def extract_persons_from_texts(self, texts: list[str], batch_size: int = 100) -> list[list[str]]:
        ner_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            ner_results_batch = self.extract_entities(batch)
            ner_results.extend(ner_results_batch)
        return [self._extract_persons(ner_result) for ner_result in ner_results]
    
class NER_ZH:
    def __init__(self) -> None:
        from nerpy import NERModel
        self.model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    
    def extract_persons_from_texts(self, texts: list[str]) -> list[list[dict]]:
        predictions, raw_outputs, entities = self.model.predict(texts, split_on_space=False)
        
        persons_lst = []
        for entity_text in entities:
            persons = []
            for name, type in entity_text:
                if type == "PER":
                    persons.append(name)
            persons_lst.append(persons)
        return persons_lst


if __name__ == "__main__":
    # ner = NER()
    # persons = ner.extract_persons_from_texts(["Fei Zhang was the president.", "Haoran Ye founded SpaceX."])
    # print(persons)

    ner_zh = NER_ZH()
    persons = ner_zh.extract_entities(["张飞和刘备是好朋友。", "猪八戒和沙悟净是西天取经的好朋友。"])
    print(persons)