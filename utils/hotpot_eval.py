from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import Document
import json
from eval import calculate_f1
import config
import time
from typing import Dict, List, Tuple, Any

def setup_llama_index():
    # Initialize settings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5", 
        embed_batch_size=16
    )
    Settings.llm = Ollama(model="llama3", request_timeout=90.0)

def query_with_vector_search(document: str, query: str) -> Tuple[str, float]:
    """
    Query documents using vector search method
    
    Args:
        document: Document content
        query: Query question
    
    Returns:
        Answer result and query latency time
    """
    start_time = time.time()
    
    # Create document and index
    doc = Document(text=document)
    index = VectorStoreIndex.from_documents([doc])
    query_engine = index.as_query_engine()
    
    # Get answer for the question
    response = query_engine.query(query)
    
    end_time = time.time()
    latency = end_time - start_time
    
    return str(response), latency

def process_single_qa(qa_item: Dict[str, Any], retrieval_method: callable) -> Dict[str, float]:
    """
    Process a single QA pair
    
    Args:
        qa_item: QA pair data
        retrieval_method: Retrieval method function
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create document from all context
    context = ""
    for title, sentences in qa_item.get('context', []):
        for sentence in sentences:
            context += f"{title}: {sentence}\n"
    
    # Get answer using specified retrieval method
    question = qa_item['question']
    response, latency = retrieval_method(context, question)
    
    # Calculate F1 score
    metrics = calculate_f1(response, qa_item['answer'])
    metrics['latency'] = latency
    
    return metrics

def evaluate_hotpot_qa(file_path: str, retrieval_method: callable = query_with_vector_search):
    """
    Evaluate HotpotQA dataset
    
    Args:
        file_path: Dataset file path
        retrieval_method: Retrieval method function
    
    Returns:
        Average evaluation metrics
    """
    # Setup LlamaIndex
    setup_llama_index()
    
    # Read the HotpotQA dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = data[:100]  # Only process first 10 questions
    
    total_metrics = {
        'em': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'latency': 0.0
    }
    
    # Process each QA pair
    for i, qa_item in enumerate(data):
        print(f"Processing question {i+1}/{len(data)}")
        metrics = process_single_qa(qa_item, retrieval_method)
        
        # Accumulate metrics
        for key in total_metrics:
            total_metrics[key] += metrics[key]
            
        # Print intermediate results
        print(f"Current metrics for question {i+1}:")
        print(json.dumps(metrics, indent=2))
    
    # Calculate averages
    num_questions = len(data)
    avg_metrics = {
        key: value/num_questions 
        for key, value in total_metrics.items()
    }
    
    return avg_metrics

if __name__ == "__main__":
    file_path = ''
    results = evaluate_hotpot_qa(file_path)
    print("\nFinal Average Metrics:")
    print(json.dumps(results, indent=2))