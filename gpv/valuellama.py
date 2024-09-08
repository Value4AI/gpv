from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


class ValueLlama:
    def __init__(self, model_name="Value4AI/ValueLlama-3-8B", device="auto"):
        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        # templates
        self.valence_template = """[Task] Given a sentence and a value, determine whether the sentence supports or opposes the value. If the sentence supports the value, output "support". If the sentence opposes the value, output "oppose". If you need more context to make a decision, output "either".
Sentence: {sentence}
Value: {value}
Output:\n"""

        self.relevance_template = """[Task] Given a sentence and a value, determine whether the sentence is relevant to the value. If the sentence is relevant to the value, output "yes", otherwise output "no".
Sentence: {sentence}
Value: {value}
Output:\n"""

        self.get_default_batch_sizes()
        print("inference batch size", self.inference_batch_size)

        self.get_token_ids()  


    def get_token_ids(self,):
        key_words = ['yes', 'no', 'support', 'oppose', 'either']
        self.token_ids = {}
        for word in key_words:
            tokens = self.tokenizer.tokenize(word)
            token_id = self.tokenizer.convert_tokens_to_ids(tokens[0])
            self.token_ids[word] = token_id
        self.relevant_ids = [self.token_ids['yes'], self.token_ids['no']]
        self.valence_ids = [self.token_ids['support'], self.token_ids['oppose'], self.token_ids['either']]         
        self.index_to_relevance = {0: 'yes', 1: 'no'}
        self.index_to_valence = {0: 'support', 1: 'oppose', 2: 'either'}


    def get_default_batch_sizes(self):
        '''
        Function to get default batch sizes based on GPU memory
        '''
        if not torch.cuda.is_available():
            self.inference_batch_size = 8
            return
        # get total memory
        # initialize total_memory
        total_memory = 0

        # iterate over all devices
        for i in range(torch.cuda.device_count()):
            total_memory += torch.cuda.get_device_properties(i).total_memory

        # if over 80GB (a100)
        if total_memory > 80_000_000_000:
            self.inference_batch_size = 128
        # else, if over 50GB (a6000)
        elif total_memory > 50_000_000_000:
            self.inference_batch_size = 64
        else:
            self.inference_batch_size = 24


    def get_probs(self, inputs, batch_size=None):
        def prepare_prompts(prompts, tokenizer, batch_size):
            batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
            batches_tok=[]
            tokenizer.padding_side="left"     
            for prompt_batch in batches:
                batches_tok.append(
                    tokenizer(
                        prompt_batch, 
                        return_tensors="pt", 
                        padding='longest', 
                        truncation=False, 
                        pad_to_multiple_of=8,
                        add_special_tokens=False).to("cuda") 
                    )
            tokenizer.padding_side="right"
            return batches_tok
        
        if batch_size is None:
            batch_size = self.inference_batch_size
            
        logits = []
        prompt_batches=prepare_prompts(inputs, self.tokenizer, batch_size=batch_size)
        for prompts_tokenized in prompt_batches:
            outputs_tokenized=self.model.generate(**prompts_tokenized, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            logits_batch = outputs_tokenized.scores[0].detach().cpu() # (batch_size, vocab_size)
            logits.append(logits_batch)

        # concatenate logits
        logits_cat = torch.cat(logits, dim=0)
        # Get probabilities
        probs = torch.softmax(logits_cat, dim=-1)
        return probs


    def get_probs_template(self, perceptions, values, template, token_ids, batch_size=None):
        # templatize
        inputs = [template.format(sentence=s, value=v) for s, v in zip(perceptions, values)]
        # pass through get_probs
        probs = self.get_probs(inputs, batch_size=batch_size)
        probs = probs[:, token_ids]
        # renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs.cpu()


    def get_relevance(self, perceptions, values, batch_size=None):
        # check if str (if single instance, then batch)
        single = False
        if isinstance(perceptions, str):
            perceptions = [perceptions]
            values = [values]
            single = True
        # run through get_probs_template
        probs = self.get_probs_template(perceptions, values, self.relevance_template, self.relevant_ids, batch_size=batch_size)
        if single:
            probs = probs[0]
        return probs


    def get_valence(self, perceptions, values, batch_size=None):
        # check if str (if single instance, then batch)
        single = False
        if isinstance(perceptions, str):
            perceptions = [perceptions]
            values = [values]
            single = True
        # run through get_probs_template
        probs = self.get_probs_template(perceptions, values, self.valence_template, self.valence_ids, batch_size=batch_size)
        if single:
            probs = probs[0]
        return probs
