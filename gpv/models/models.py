import os
from abc import ABC
import concurrent.futures
from tqdm import tqdm
import time
from typing import Optional

import torch
from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
except ImportError:
    print("Mistral is not installed. Related models will not work.")



class LLMBaseModel(ABC):
    """
    Abstract base class for language model interfaces.

    This class provides a common interface for various language models and includes methods for prediction.

    Parameters:
    -----------
    model : str
        The name of the language model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    __call__(input_text, **kwargs)
        Shortcut for predict method.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device='auto'):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
    
    def predict(self, input_text, **kwargs):
        raise NotImplementedError("The predict method must be implemented in the derived class.")

    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)


class InternLMModel(LLMBaseModel):
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(InternLMModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=dtype, device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=dtype, device_map=device)
        self.model = self.model.eval()

    def predict(self, input_text, **kwargs):
        response, history = self.model.chat(self.tokenizer, input_text, history=[])
        return response



class YiModel(LLMBaseModel):
    """
    Language model class for the Yi model.

    Inherits from LLMBaseModel and sets up the Yi language model for use.

    Parameters:
    -----------
    model : str
        The name of the Yi model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype, system_prompt=None):
        super(YiModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device).eval()
        self.system_prompt = system_prompt if system_prompt is not None else "You are a helpful assistant."

    def predict(self, input_text, **kwargs):
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_text}]
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(
            input_ids.to(self.device),
            temperature=self.temperature if self.temperature > 1e-3 else None,
            top_p=0.9 if self.temperature > 1e-3 else None,
            max_new_tokens=self.max_new_tokens,
            do_sample=True if self.temperature > 1e-3 else False,
            )
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Model response: "Hello! How can I assist you today?"
        return response


class MistralModel(LLMBaseModel):
    """
    Language model class for the Mistral model.

    Inherits from LLMBaseModel and sets up the Mistral language model for use.

    Parameters:
    -----------
    model : str
        The name of the Mistral model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        temperature = max(temperature, 0.01)
        super(MistralModel, self).__init__(model_name, max_new_tokens, temperature, device)
        self.tokenizer = MistralTokenizer.v1()
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)
        

    def predict(self, input_text, **kwargs):
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=input_text)])
        
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens

        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        
        generated_ids = self.model.generate(
            tokens,
            max_new_tokens=self.max_new_tokens, 
            temperature=self.temperature if self.temperature > 1e-3 else None,
            top_p=0.9 if self.temperature > 1e-3 else None,
            do_sample=True if self.temperature > 1e-3 else False,
            pad_token_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
            **kwargs,
        )

        # decode with mistral tokenizer
        result = self.tokenizer.decode(generated_ids[0].tolist())
        
        # Return the content after [/INST]
        response = result.split("[/INST]")[1]
        return response


class LlamaModel(LLMBaseModel):
    """
    Language model class for the Llama model.

    Inherits from LLMBaseModel and sets up the Llama language model for use.

    Parameters:
    -----------
    model : str
        The name of the Llama model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    system_prompt : str
        The system prompt to be used (default is 'You are a helpful assistant.').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype, system_prompt):
        super(LlamaModel, self).__init__(model_name, max_new_tokens, temperature, device)
        if system_prompt is None:
            self.system_prompt = "You are a helpful assistant."
        else:
            self.system_prompt = system_prompt
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device, torch_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype)

    def predict(self, input_text, **kwargs):
        input_text = f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>\n{input_text}[/INST]"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.model.generate(input_ids, 
                                     max_new_tokens=self.max_new_tokens, 
                                     temperature=self.temperature if self.temperature > 1e-3 else None,
                                     top_p=0.9 if self.temperature > 1e-3 else None,
                                     do_sample=True if self.temperature > 1e-3 else False,
                                     **kwargs)
        
        out = self.tokenizer.decode(outputs[0], 
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=False)
        
        return out[len(input_text)-1:]


class VicunaModel(LLMBaseModel):
    """
    Language model class for the Vicuna model.
    # https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/huggingface_api.py

    Inherits from LLMBaseModel and sets up the Vicuna language model for use.

    Parameters:
    -----------
    model : str
        The name of the Vicuna model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(VicunaModel, self).__init__(model_name, max_new_tokens, temperature, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device, torch_dtype=dtype, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype)

    def predict(self, input_text, **kwargs):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        print(self.temperature)

        output_ids = self.model.generate(
            input_ids, 
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 1e-3 else None,
            do_sample=True if self.temperature > 1e-3 else False,
            repetition_penalty=1.2, # https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/huggingface_api.py
            **kwargs
            )

        output_ids = output_ids[0][len(input_ids[0]) :]
        
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        return outputs


class OpenAIModel(LLMBaseModel):
    """
    Language model class for interfacing with OpenAI's GPT models or Llama API models.

    Inherits from LLMBaseModel and sets up a model interface for OpenAI GPT models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is 'You are a helpful assistant.').
    openai_key : str
        The OpenAI API key (default is None).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the OpenAI model.
    """
    def __init__(self, model_name, max_new_tokens, temperature, system_prompt=None, openai_key=None):
        super(OpenAIModel, self).__init__(model_name, max_new_tokens, temperature)    
        self.openai_key = openai_key
        self.system_prompt = system_prompt

    def predict(self, input_text, kwargs={}):
        client = OpenAI(api_key=self.openai_key if self.openai_key is not None else os.environ['OPENAI_API_KEY'])
        if self.system_prompt is None:
            system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        else:
            system_messages = {'role': "system", 'content': self.system_prompt}
        
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]
        
        messages.insert(0, system_messages)
    
        # extra parameterss
        n = kwargs['n'] if 'n' in kwargs else 1
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        response_format = kwargs['response_format'] if 'response_format' in kwargs else None
        
        for attempt in range(1000):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    n=n,
                    response_format={"type": "json_object"} if response_format=="json" else None,
                )
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying ({attempt + 1})...")
                time.sleep(1)
            
        if n > 1:
            result = [choice.message.content for choice in response.choices]
        else:
            result = response.choices[0].message.content
            
        return result

    def multi_predict(self, input_texts, **kwargs):
        """
        An example of input_texts:
        input_texts = ["Hello!", "How are you?", "Tell me a joke."]
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            args = [(messages, kwargs) for messages in input_texts]
            contents = executor.map(lambda p: self.predict(*p), args)
        return list(contents)

    def batch_predict(self, input_texts, **kwargs):
        assert "n" not in kwargs or kwargs["n"] == 1, "n > 1 is not supported for batch prediction."
        responses_list = []
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 50
        for start_idx in tqdm(range(0, len(input_texts), batch_size), disable=True):
            end_idx = min(start_idx + batch_size, len(input_texts))
            batch_input_texts = input_texts[start_idx: end_idx]
            batch_results_list = self.multi_predict(batch_input_texts, **kwargs)
            responses_list.extend(batch_results_list)
            # Save responses to file
            # with open(f"temp-file-responses-{self.model_name}.txt", "a") as f:
            #     for response in batch_results_list:
            #         f.write(response + "\n")
        return responses_list


class LlamaAPIModel(OpenAIModel):
    def __init__(self, model_name, max_new_tokens, temperature, system_prompt=None, llama_key=None):
        super(LlamaAPIModel, self).__init__(model_name, max_new_tokens, temperature, system_prompt, llama_key)
        self.system_prompt = system_prompt
        self.llama_key = llama_key
    
    def predict(self, input_text, kwargs={}):
        client = OpenAI(
                    api_key = self.llama_key if self.llama_key is not None else os.environ['LLAMA_API_KEY'],
                    base_url = "https://api.llama-api.com"
                    )
        if self.system_prompt is None:
            system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        else:
            system_messages = {'role': "system", 'content': self.system_prompt}
        
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]
        
        messages.insert(0, system_messages)
    
        # extra parameterss
        n = kwargs['n'] if 'n' in kwargs else 1
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        response_format = kwargs['response_format'] if 'response_format' in kwargs else None
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n,
            response_format={"type": "json_object"} if response_format=="json" else None,
        )
        
        if n > 1:
            result = [choice.message.content for choice in response.choices]
        else:
            result = response.choices[0].message.content
            
        return result

    
if __name__ == "__main__":
    model_name = "llama3.1-405b"

    model = LlamaAPIModel(model_name, max_new_tokens=4096, temperature=0)

    user_prompt = "Hi there"
    response = model.predict(user_prompt)
    
    print(response)
        