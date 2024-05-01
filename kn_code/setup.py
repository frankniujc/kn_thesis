import torch
from transformers import (
    BertTokenizer, BertLMHeadModel,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, LlamaForCausalLM,
)
from .knowledge_neurons import KnowledgeNeurons

BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-uncased", "bert-base-cased"]
GPT2_MODELS = ["gpt2", "gpt2-xl"]
ALL_MODELS = BERT_MODELS + GPT2_MODELS


def initialize_model_and_tokenizer(model_name: str, torch_dtype='auto'):
    if model_name in BERT_MODELS:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(model_name)
    elif model_name in GPT2_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, torch_dtype=torch_dtype)
        model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    elif 'llama' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    else:
        raise ValueError("Model {model_name} not supported")

    model.eval()

    return model, tokenizer

def model_type(model_name: str):
    if model_name in BERT_MODELS:
        return "bert"
    elif model_name in GPT2_MODELS:
        return "gpt2"
    elif 'llama' in model_name:
        return 'llama'
    else:
        raise ValueError("Model {model_name} not supported")


def load_model(model_name_or_path, device=None):
    model, tokenizer = initialize_model_and_tokenizer(model_name_or_path)
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(model_name_or_path))
    if kn.model_type in ['gpt2', 'llama']:
        kn.tokenizer.pad_token = kn.tokenizer.eos_token
    return kn