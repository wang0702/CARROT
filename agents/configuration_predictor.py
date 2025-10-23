"""
ConfigurationPredictor - MCTS Configuration Prediction
Predicts optimal reranker and MCTS hyperparameters for queries
"""

import os
import torch
import joblib
from FlagEmbedding import BGEM3FlagModel

from agents.network_enhanced import SiameseNetwork


class ConfigurationPredictor:
    """
    ConfigurationPredictor uses a trained SiameseNetwork to predict:
    1. Optimal reranker for a given query
    2. Optimal MCTS hyperparameters (C, iterations, lambda)
    """

    def __init__(self, model_path='models/siamese_model.pth',
                 label_encoder_path='models/label_encoder.pkl'):
        """
        Initialize ConfigurationPredictor

        Args:
            model_path: Path to trained predictor model
            label_encoder_path: Path to label encoder
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load label encoder
        if not os.path.isabs(label_encoder_path) and not os.path.exists(label_encoder_path):
            alt_path = os.path.join('models', os.path.basename(label_encoder_path))
            if os.path.exists(alt_path):
                label_encoder_path = alt_path

        self.label_encoder = joblib.load(label_encoder_path)

        # Load predictor model
        self.predictor_model = SiameseNetwork(embed_dim=1024, num_classes=6).to(self.device)

        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            alt_path = os.path.join('models', os.path.basename(model_path))
            if os.path.exists(alt_path):
                model_path = alt_path

        self.predictor_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.predictor_model.eval()

        # Setup embedding model for query encoding
        self.query_embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def predict(self, query, retrieved_nodes=None):
        """
        Predict optimal reranker and MCTS parameters for a given query

        Args:
            query: Input query string
            retrieved_nodes: List of retrieved nodes (optional, for chunk-aware prediction)

        Returns:
            predicted_reranker: Name of the predicted reranker
            predicted_params: Array of [C_param, iteration, lambda]
        """
        # Generate embedding for query
        query_embedding = self.query_embedding_model.encode(
            query, batch_size=12, max_length=8192
        )['dense_vecs']

        # Convert to tensor
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).to(self.device)
        query_tensor = query_tensor.unsqueeze(0)

        # If chunk-aware prediction is enabled and chunks are provided
        chunk_tensor = None
        if retrieved_nodes and hasattr(self.predictor_model, 'use_chunk_fusion') and self.predictor_model.use_chunk_fusion:
            # Generate chunk embeddings (average pooling of top chunks)
            chunk_texts = [node.node.text for node in retrieved_nodes[:10]]  # Use top 10 chunks
            if chunk_texts:
                chunk_embeddings = self.query_embedding_model.encode(
                    chunk_texts, batch_size=len(chunk_texts), max_length=8192
                )['dense_vecs']

                # Average pooling
                import numpy as np
                chunk_embedding = np.mean(chunk_embeddings, axis=0)
                chunk_tensor = torch.tensor(chunk_embedding, dtype=torch.float32).to(self.device)
                chunk_tensor = chunk_tensor.unsqueeze(0)

        # Predict
        with torch.no_grad():
            if chunk_tensor is not None:
                class_output, param_output = self.predictor_model(query_tensor, chunk_tensor)
            else:
                class_output, param_output = self.predictor_model(query_tensor)
            _, predicted = torch.max(class_output, dim=1)

        # Decode predicted reranker
        predicted_label = self.label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
        predicted_params = param_output.cpu().numpy()[0]

        return predicted_label, predicted_params
