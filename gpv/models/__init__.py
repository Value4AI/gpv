from tqdm import tqdm

from .models import *

# A dictionary mapping of model architecture to its supported model names
MODEL_LIST = {
    # InternLMModel: ['internlm/internlm2-chat-7b', 'internlm/internlm2-chat-20b', 'internlm/internlm-chat-7b'],
    LlamaAPIModel: ['gemma-7b', 'gemma-2b', 'llama3.1-405b'] # These are gemma instruct/chat models
        + [f'Qwen1.5-{n}B-Chat' for n in [110, 72, 32, 14, 7, 4, 1.8, 0.5]], # We use LlamaAPI for these models, one can also implement them locally
    # LlamaModel: ['meta-llama/Llama-2-7b-chat-hf'],
    OpenAIModel: ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o-mini', 'gpt-4o'],
    # VicunaModel: ['lmsys/vicuna-7b-v1.5-16k'],
    # MistralModel: ['mistralai/Mistral-7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.2'],
    # YiModel: ['01-ai/Yi-6B-Chat'],
}

SUPPORTED_MODELS = [model for model_class in MODEL_LIST.keys() for model in MODEL_LIST[model_class]]


class LLMModel(object):
    """
    A class providing an interface for various language models.

    This class supports creating and interfacing with different language models, handling prompt engineering, and performing model inference.

    Parameters:
    -----------
    model : str
        The name of the model to be used.
    max_new_tokens : int, optional
        The maximum number of new tokens to be generated (default is 20).
    temperature : float, optional
        The temperature for text generation (default is 0).
    device : str, optional
        The device to be used for inference (default is "cuda").
    dtype : str, optional
        The loaded data type of the language model (default is "auto").
    model_dir : str or None, optional
        The directory containing the model files (default is None).
    system_prompt : str or None, optional
        The system prompt to be used (default is None).
    api_key : str or None, optional
        The API key for API-based models (GPT series and Gemini series), if required (default is None).

    Methods:
    --------
    _create_model(max_new_tokens, temperature, device, dtype, model_dir, system_prompt, api_key)
        Creates and returns the appropriate model instance.
    convert_text_to_prompt(text, role)
        Constructs a prompt based on the text and role.
    concat_prompts(prompt_list)
        Concatenates multiple prompts into a single prompt.
    _gpt_concat_prompts(prompt_list)
        Concatenates prompts for GPT models.
    _other_concat_prompts(prompt_list)
        Concatenates prompts for non-GPT models.
    __call__(input_text, **kwargs)
        Makes a prediction based on the input text using the loaded model.
    """
    
    @staticmethod
    def model_list():
        return SUPPORTED_MODELS

    def __init__(self, model, max_new_tokens=4096, temperature=0, device="cuda", dtype=torch.float16, system_prompt=None, api_key=None):
        self.model_name = model
        self.model = self._create_model(max_new_tokens, temperature, device, dtype, system_prompt, api_key)

    def _create_model(self, max_new_tokens, temperature, device, dtype, system_prompt, api_key):
        """Creates and returns the appropriate model based on the model name."""

        # Dictionary mapping of model names to their respective classes
        model_mapping = {model: model_class for model_class in MODEL_LIST.keys() for model in MODEL_LIST[model_class]}

        # Get the model class based on the model name and instantiate it
        model_class = model_mapping.get(self.model_name)
        if model_class:
            if model_class in [LlamaAPIModel, OpenAIModel]:
                return model_class(self.model_name, max_new_tokens, temperature, system_prompt, api_key)
            elif model_class in [YiModel, LlamaModel]:
                return model_class(self.model_name, max_new_tokens, temperature, device, dtype, system_prompt)
            else:
                return model_class(self.model_name, max_new_tokens, temperature, device, dtype)
        else:
            raise ValueError("The model is not supported!")
    
    def __call__(self, input_texts, **kwargs):
        """Predicts the output based on the given input text using the loaded model."""
        if not isinstance(input_texts, list):
            assert isinstance(input_texts, str)
            input_texts = [input_texts]
        if isinstance(self.model, OpenAIModel) or isinstance(self.model, LlamaAPIModel):
            return self.model.batch_predict(input_texts, **kwargs)
        else:
            responses = []
            for input_text in tqdm(input_texts):
                responses.append(self.model.predict(input_text, **kwargs))
            return responses