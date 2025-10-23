import torch
from pathlib import Path
from agents.train import SiameseNetworkTrainer

if __name__ == "__main__":
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).parent.absolute()

    # Define paths relative to script directory
    model_path = str(SCRIPT_DIR / 'siamese_model.pth')
    USE_CHUNK_FUSION = True

    print("=" * 80)
    print("Testing Prediction with Chunk Fusion")
    print("=" * 80)

    # Create new trainer instance for prediction
    trainer = SiameseNetworkTrainer()
    # Load model with chunk fusion mode
    trainer.load_model(model_path, use_chunk_fusion=USE_CHUNK_FUSION)

    print(f"\nModel loaded successfully!")
    print(f"Chunk fusion enabled: {trainer.use_chunk_fusion}")
    print(f"Model type: {type(trainer.model).__name__}")

    query = "What is the capital of France?"
    chunk_text = "France is a country in Western Europe. Paris is the capital and largest city of France."

    if trainer.use_chunk_fusion:
        print(f"\nUsing chunk fusion mode for prediction")
        predicted_reranker, predicted_params = trainer.predict(query, chunk_text=chunk_text)
        print(f"\nQuery: {query}")
        print(f"Chunk: {chunk_text}")
    else:
        print(f"\nUsing query-only mode for prediction")
        predicted_reranker, predicted_params = trainer.predict(query)
        print(f"\nQuery: {query}")

    print(f"Predicted Reranker: {predicted_reranker}")
    print(f"Predicted Parameters: C_param={predicted_params[0]:.2f}, "
          f"iteration={predicted_params[1]:.0f}, lambda={predicted_params[2]:.2f}")
