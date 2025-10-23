import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import os
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from FlagEmbedding import BGEM3FlagModel
import joblib  # For saving and loading the label encoder
from agents.train import SiameseNetworkTrainer

if __name__ == "__main__":
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).parent.absolute()

    # Define paths relative to script directory
    file_path = SCRIPT_DIR / 'data' / 'dummy_training.pkl'
    model_path = SCRIPT_DIR / 'siamese_model.pth'  # Use newly trained model in current directory
    USE_CHUNK_FUSION = True

    file_path = str(file_path)
    model_path = str(model_path)

    print(f"Loading training data from {file_path}")
    df = pd.read_pickle(file_path)

    print(f"\nDataset info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    has_chunk_embed = 'nodes_query_embedding' in df.columns

    if USE_CHUNK_FUSION and not has_chunk_embed:
        print("\nWarning: USE_CHUNK_FUSION=True but 'nodes_query_embedding' column not found!")
        print("Falling back to query-only mode")
        USE_CHUNK_FUSION = False

    print(f"\nTraining mode: {'Query + Chunk Fusion' if USE_CHUNK_FUSION else 'Query Only'}")

    param_df = df[['C_param', 'iteration', 'lambda']]

    trainer = SiameseNetworkTrainer(
        df,
        embed_column='question_embedding',
        label_column='max_reranker',
        param_df=param_df,
        chunk_embed_column='nodes_query_embedding' if USE_CHUNK_FUSION else None,
        use_chunk_fusion=USE_CHUNK_FUSION
    )

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    trainer.train()

    print("\n" + "=" * 80)
    print("Testing Prediction")
    print("=" * 80)

    # Create new trainer instance for prediction
    trainer = SiameseNetworkTrainer()
    # Load model with chunk fusion mode matching training
    trainer.load_model(model_path, use_chunk_fusion=USE_CHUNK_FUSION)

    query = "What is the capital of France?"
    chunk_text = "France is a country in Western Europe. Paris is the capital and largest city of France."

    if trainer.use_chunk_fusion:
        print(f"\nUsing chunk fusion mode for prediction")
        predicted_reranker, predicted_params = trainer.predict(query, chunk_text=chunk_text)
    else:
        print(f"\nUsing query-only mode for prediction")
        predicted_reranker, predicted_params = trainer.predict(query)

    print(f"\nQuery: {query}")
    if trainer.use_chunk_fusion:
        print(f"Chunk: {chunk_text}")
    print(f"Predicted Reranker: {predicted_reranker}")
    print(f"Predicted Parameters: C_param={predicted_params[0]:.2f}, "
          f"iteration={predicted_params[1]:.0f}, lambda={predicted_params[2]:.2f}")
