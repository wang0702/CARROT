"""
ModelLoader - Unified Model Loading and Management
Handles loading and configuration of LLM, embedding, and reranker models
"""

import torch
from openai import OpenAI
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core import Settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FlagEmbedding import FlagReranker

from core.config import API_OPENROUTER, API_KEY_1


class ModelLoader:
    """
    ModelLoader handles loading and configuration of all models:
    - LLM (Language Model)
    - Embedding models
    - Reranker models (Jina, BGE, GTE)
    """

    def __init__(self, llm_str='openai/gpt-4o', embed_model='BAAI/bge-large-en-v1.5',
                 use_openai_direct=True):
        """
        Initialize ModelLoader

        Args:
            llm_str: LLM model name (e.g., 'openai/gpt-4o', 'gpt-4o')
            embed_model: Embedding model name
            use_openai_direct: If True, use OpenAI API directly; else use OpenRouter
        """
        self.llm_str = llm_str
        self.embed_model_name = embed_model
        self.use_openai_direct = use_openai_direct

        # Setup all models
        self._setup_llm()
        self._setup_embedding_model()
        self._setup_rerankers()

    def _setup_llm(self):
        """Setup LLM with OpenAI API (primary) and OpenRouter (backup)"""
        if self.use_openai_direct:
            # Primary: OpenAI direct API
            self.openai_client = OpenAI(api_key=API_KEY_1)
            # Backup: OpenRouter
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=API_OPENROUTER,
            )
        else:
            # Primary: OpenRouter
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=API_OPENROUTER,
            )
            # Backup: OpenAI
            self.openai_client = OpenAI(api_key=API_KEY_1)

        # For backward compatibility with LlamaIndex (kept for retrieval components)
        model_name = self.llm_str.split('/')[-1] if '/' in self.llm_str else self.llm_str
        if self.use_openai_direct:
            self.llm = LlamaIndexOpenAI(model=model_name, api_key=API_KEY_1, temperature=0.0)
        else:
            self.llm = LlamaIndexOpenAI(model=model_name, api_key=API_OPENROUTER,
                                       api_base="https://openrouter.ai/api/v1", temperature=0.0)
        Settings.llm = self.llm

    def _setup_embedding_model(self):
        """Setup embedding model for LlamaIndex"""
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
        Settings.embed_model = self.embed_model

    def _setup_rerankers(self):
        """Load all reranker models"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        use_fp16 = (device == 'cuda')

        # Jina Rerankers (use raw models, have compute_score method)
        self.jina_model_v1 = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v1-turbo-en',
            num_labels=1,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            trust_remote_code=True
        ).to(device).eval()

        self.jina_model_v2 = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v2-base-multilingual',
            num_labels=1,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            trust_remote_code=True,
            ignore_mismatched_sizes=True  # Fix position embeddings size mismatch
        ).to(device).eval()

        # BGE Rerankers (use FlagReranker wrapper for compute_score method)
        self.flag_reranker_v2 = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            use_fp16=use_fp16
        )

        self.flag_reranker_large = FlagReranker(
            'BAAI/bge-reranker-large',
            use_fp16=use_fp16
        )

        self.flag_reranker_base = FlagReranker(
            'BAAI/bge-reranker-base',
            use_fp16=use_fp16
        )

        # GTE Reranker
        self.gte_model = AutoModelForSequenceClassification.from_pretrained(
            'Alibaba-NLP/gte-multilingual-reranker-base',
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(device).eval()

        self.gte_tokenizer = AutoTokenizer.from_pretrained(
            'Alibaba-NLP/gte-multilingual-reranker-base',
            trust_remote_code=True
        )

    def get_all_models(self):
        """
        Get all models for evaluation scripts

        Returns:
            Tuple of (llm, jina_v1, jina_v2, flag_v2, flag_large, flag_base,
                     gte_model, gte_tokenizer)
        """
        return (
            self.llm,
            self.jina_model_v1,
            self.jina_model_v2,
            self.flag_reranker_v2,
            self.flag_reranker_large,
            self.flag_reranker_base,
            self.gte_model,
            self.gte_tokenizer,
        )

    def get_reranker_by_name(self, reranker_name):
        """
        Get reranker model and tokenizer by name

        Args:
            reranker_name: Name of the reranker

        Returns:
            (model, tokenizer) or (None, None) if not found
        """
        reranker_map = {
            'jina-reranker-v1-turbo-en': (self.jina_model_v1, None),
            'jina-reranker-v2-base-multilingual': (self.jina_model_v2, None),
            'bge-reranker-v2-m3': (self.flag_reranker_v2, None),
            'bge-reranker-large': (self.flag_reranker_large, None),
            'bge-reranker-base': (self.flag_reranker_base, None),
            'gte-multilingual-reranker-base': (self.gte_model, self.gte_tokenizer),
        }

        return reranker_map.get(reranker_name, (None, None))

    def complete(self, prompt):
        """
        Direct LLM completion with OpenAI (primary) and OpenRouter (fallback)

        Args:
            prompt: Input prompt string

        Returns:
            Completion text
        """
        model_name = self.llm_str.split('/')[-1] if '/' in self.llm_str else self.llm_str

        if self.use_openai_direct:
            try:
                # Try OpenAI API first
                completion = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"⚠ OpenAI API failed: {e}, falling back to OpenRouter...")
                # Fallback to OpenRouter
                completion = self.openrouter_client.chat.completions.create(
                    extra_headers={"HTTP-Referer": "NA", "X-Title": "CARROT-RAG"},
                    model=self.llm_str,
                    messages=[{"role": "user", "content": prompt}]
                )
                return completion.choices[0].message.content
        else:
            try:
                # Try OpenRouter first
                completion = self.openrouter_client.chat.completions.create(
                    extra_headers={"HTTP-Referer": "NA", "X-Title": "CARROT-RAG"},
                    model=self.llm_str,
                    messages=[{"role": "user", "content": prompt}]
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"⚠ OpenRouter API failed: {e}, falling back to OpenAI...")
                # Fallback to OpenAI
                completion = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return completion.choices[0].message.content
