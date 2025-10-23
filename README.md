# CARROT: Cost-Constrained Retrieval Optimization

**CARROT** (**C**ost-constr**A**ined **R**et**R**ieval **O**p**T**imization) is a rank-based RAG framework that optimizes chunk combination order using Monte Carlo Tree Search (MCTS) with cost constraints.

## Overview

CARROT addresses three key challenges in RAG systems:

1. **Chunk Correlation**: Finding optimal chunk combinations considering relationships and ordering
2. **Non-monotonic Utility**: Handling cases where adding more chunks decreases response quality
3. **Query Diversity**: Adapting reranker selection and parameters to different query types


## Installation

```bash
# Create environment
conda create -n rag_new python=3.10
conda activate rag_new

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `core/config.py` with your API keys:

```python
API_KEY_1 = "sk-your-openai-api-key-here"          # OpenAI API
API_OPENROUTER = "sk-or-v1-your-openrouter-key"   # OpenRouter API (fallback)
```

Get keys from: [OpenAI](https://platform.openai.com/api-keys) | [OpenRouter](https://openrouter.ai/keys)

## Quick Start

### 1. Train Configuration Agent (Optional)

```bash
# Train on provided dummy training data
python main_train.py
```

This trains the configuration agent to predict optimal reranker and MCTS parameters. The trained model will be saved to `models/siamese_model.pth`.

### 2. Run Demo

```bash
# With fixed configuration (Jina-v2 reranker)
python main.py

# With configuration agent (auto-selects reranker)
python main.py --agent
```

The demo queries a built-in document collection about MCTS and chunk retrieval, demonstrating the full pipeline.

## Advanced: Train on Your Own Dataset

To train the configuration agent on your own data, follow these steps (using WikiPassage as example):

### Step 1: Prepare Your Data

Organize your dataset in a similar structure to `data/wikipassage/`:
- `document_passages.json`: Document chunks in JSON format
  ```json
  {
    "doc_id_1": {
      "passage_1": "text content...",
      "passage_2": "text content..."
    }
  }
  ```
- `test.tsv`: Questions with columns: `Question`, `DocumentID`, `RelevantPassages`

### Step 2: Build FAISS Index

```bash
python data/build_faiss_index.py --chunk-size 256
```

Creates a vector index for efficient chunk retrieval using BGE-M3 embeddings.

### Step 3: Generate Training Data

```bash
python data/data_preprocess_wikipassage_smart.py --max-queries 1000
```

Performs grid search over rerankers and MCTS parameters to find optimal configurations for each query. Saves results to `{dataset}_training_data.pkl`.

**Note**: This step is computationally expensive and may take hours.

### Step 4: Train Configuration Agent

Update `main_train.py` to point to your training data, then run:

```bash
python main_train.py
```

The trained model will be saved to `models/siamese_model.pth`.

## How It Works

### MCTS Search Process

1. **Selection**: Use UCB to traverse tree and find promising node
2. **Expansion**: Generate child nodes by adding each remaining chunk
3. **Simulation**: Batch evaluate all children with reranker
4. **Backpropagation**: Update statistics along path to root

After iterations, select the best node across all depths within budget.

### UCB Formula

```
U(node) = avg_reward + C × sqrt(ln(N) / visits) - λ × (cost / budget)
```

- **C**: Exploration coefficient
- **λ**: Cost penalty coefficient
- **cost/budget**: Token usage ratio

### Configuration Agent

Optional neural predictor that selects optimal reranker and parameters per query:
- Takes query + retrieved chunk embeddings as input
- Outputs: reranker choice + [C, iterations, λ] parameters
- Uses Siamese Network with query-chunk fusion

## Supported Rerankers

- Jina Reranker v1/v2
- BGE Reranker (base/large/v2-m3)
- GTE Multilingual Reranker

## Project Structure

```
CARROT/
├── main.py                     # Demo script
├── main_train.py              # Training script
├── agents/                    # Configuration agent
├── core/                      # MCTS implementation
├── data/                      # Dataset scripts
└── models/                    # Trained models
```
