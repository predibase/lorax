import torch
from pprint import pprint

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
model.eval()


sentence = "Dog"


encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')


with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings_og = model_output[0][:, 0]


print("Sentence embeddings:", sentence_embeddings_og.norm())
