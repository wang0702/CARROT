import math
import tiktoken
import torch
from FlagEmbedding import FlagReranker
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def setup(llm_str):
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", embed_batch_size=16)
    llm = Ollama(model=llm_str, request_timeout=90.0)
    Settings.llm = llm

    jina_model_v1 = AutoModelForSequenceClassification.from_pretrained('jinaai/jina-reranker-v1-turbo-en', num_labels=1,
                                                                       trust_remote_code=True)
    jina_model_v1.eval()
    jina_model_v1.to('cuda')

    jina_model_v2 = AutoModelForSequenceClassification.from_pretrained('jinaai/jina-reranker-v2-base-multilingual',
                                                                       num_labels=1, trust_remote_code=True)
    jina_model_v2.eval()
    jina_model_v2.to('cuda')

    flag_reranker_v2 = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    flag_reranker_large = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
    flag_reranker_base = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)

    gte_model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"
    revision = "4e88bd5dec38b6b9a7e623755029fc124c319d67"

    gte_tokenizer = AutoTokenizer.from_pretrained(gte_model_name_or_path)
    gte_model = AutoModelForSequenceClassification.from_pretrained(gte_model_name_or_path, trust_remote_code=True,
                                                                   torch_dtype=torch.float16, revision=revision)
    gte_model.eval()
    gte_model.to('cuda')

    

    return llm, jina_model_v1, jina_model_v2, flag_reranker_v2, flag_reranker_large, flag_reranker_base, gte_model, gte_tokenizer
