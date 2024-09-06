import os
from tqdm import tqdm
from openai import OpenAI
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SentenceEmbedding:
    def __init__(self, model_name_or_path: str='Alibaba-NLP/gte-multilingual-base', device="cuda:0"):
        self.device = device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    

    def get_embedding(self, input_texts: list[str], dimension: int = 768, batch_size: int = 16) -> torch.Tensor:
        """
        Get the sentence embeddings of the input texts.
        
        Args:
            input_texts (list[str]): A list of input texts.
            dimension (int): The output dimension of the output embedding, should be in [128, 768].
            batch_size (int): The number of samples per batch.
            
        Returns:
            torch.Tensor: The embeddings for the input texts.
        """
        embeddings_list = []
        
        for i in tqdm(range(0, len(input_texts), batch_size), desc="Embedding", disable=True):
            # Select the batch
            batch_texts = input_texts[i:i + batch_size]
            
            # Tokenize the input texts
            batch_dict = self.tokenizer(batch_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(self.device)
            
            # Get the output embeddings
            outputs = self.model(**batch_dict)
            batch_embeddings = outputs.last_hidden_state[:, 0][:, :dimension]
            
            # Normalize the embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1) # shape: (batch_size, dimension)
            
            embeddings_list.append(batch_embeddings.detach().cpu())
        
        # Concatenate all batch embeddings
        embeddings = torch.cat(embeddings_list, dim=0)
        
        return embeddings.numpy()


if __name__ == "__main__":
    embd = SentenceEmbedding()

    input_texts = [
        "what is the capital of China?",
        "how to implement quick sort in python?",
        "北京",
        "快排算法介绍"
    ]

    embeddings = embd.get_embedding(input_texts)
    print(embeddings)