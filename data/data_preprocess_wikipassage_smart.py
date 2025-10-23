#!/usr/bin/env python3
"""
Smart Grid Search for WikiPassage with ROUGE-L scoring.
Generates training data for configuration agent.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from rouge_score import rouge_scorer
from scipy.stats.qmc import LatinHypercube

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from FlagEmbedding import BGEM3FlagModel
from agents.model_loader import ModelLoader
from core.mcts import MCTS

import logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def calculate_rouge_l(prediction: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure


def generate_stratified_samples(c_values: list, lambda_values: list, iter_values: list) -> list:
    samples = []

    c_key = [c_values[0], c_values[len(c_values)//2], c_values[-1]]
    lambda_key = [lambda_values[0], lambda_values[len(lambda_values)//2], lambda_values[-1]]
    iter_key = [iter_values[0], iter_values[len(iter_values)//2], iter_values[-1]]

    for c in c_key:
        for lam in lambda_key:
            for it in iter_key:
                samples.append((c, lam, it, 'must-test'))

    quartile_samples = []

    c_quartiles = [v for v in c_values if v not in c_key]
    lambda_quartiles = [v for v in lambda_values if v not in lambda_key]
    iter_quartiles = [v for v in iter_values if v not in iter_key]

    mid_lambda = lambda_key[1]
    mid_iter = iter_key[1]
    for c in c_quartiles:
        quartile_samples.append((c, mid_lambda, mid_iter, 'quartile'))

    mid_c = c_key[1]
    for lam in lambda_quartiles:
        quartile_samples.append((mid_c, lam, mid_iter, 'quartile'))

    for it in iter_quartiles:
        quartile_samples.append((mid_c, mid_lambda, it, 'quartile'))

    return samples, quartile_samples


class SmartGridSearch:
    def __init__(self, logger, llm, passages_dict):
        self.logger = logger
        self.llm = llm
        self.passages_dict = passages_dict
        self.best_configs = {}

    def get_ground_truth_answer(self, question: str, doc_id: str, relevant_passages: str) -> str:
        ground_truth_texts = []
        if relevant_passages and doc_id in self.passages_dict:
            passage_ids = [p.strip() for p in relevant_passages.split(',') if p.strip()]
            for passage_id in passage_ids:
                if passage_id in self.passages_dict[doc_id]:
                    ground_truth_texts.append(self.passages_dict[doc_id][passage_id])

        if not ground_truth_texts:
            self.logger.warning(f"No ground truth passages found for doc {doc_id}, passages {relevant_passages}")
            return ""

        ground_truth_context = " ".join(ground_truth_texts)
        prompt = f"""Provide a precise answer to the question based on the given context.

Question: {question}
Context: {ground_truth_context}

Answer:"""

        try:
            ground_truth_answer = self.llm.complete(prompt).text
            self.logger.info(f"Generated ground truth answer: {ground_truth_answer[:100]}...")
            return ground_truth_answer
        except Exception as e:
            self.logger.error(f"Error generating ground truth answer: {e}")
            return ""

    def evaluate_config(self, question: str, context: str, ground_truth: str) -> float:
        prompt = f"""Provide a precise answer to the question based on the given context.

Question: {question}
Context: {context}

Answer:"""

        try:
            answer = self.llm.complete(prompt).text
            rouge_score = calculate_rouge_l(answer, ground_truth)
            return rouge_score, answer
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return 0.0, ""

    def search(self, question: str, nodes: list, rerankers: dict, doc_id: str,
               relevant_passages: str, budget: int = 1024) -> dict:
        ground_truth = self.get_ground_truth_answer(question, doc_id, relevant_passages)
        if not ground_truth:
            self.logger.warning(f"Failed to generate ground truth answer for doc {doc_id}, passages {relevant_passages}")
            return {'best_config': None, 'reranker_scores': {}, 'all_results': [], 'ground_truth': ""}

        all_results = []

        self.logger.info("Phase 1: Stratified sampling with ROUGE-L scoring")
        self.logger.info("  Strategy: Priority samples (boundary+midpoint) + adaptive quartile supplements")

        coarse_params = {
            'C_values': [1.5, 2.0, 2.5, 3.0],
            'lambda_values': [0.15, 0.2, 0.25, 0.3],
            'iteration_values': [5, 10, 15, 20, 25]
        }

        priority_samples, quartile_samples = generate_stratified_samples(
            coarse_params['C_values'],
            coarse_params['lambda_values'],
            coarse_params['iteration_values']
        )

        self.logger.info(f"  Generated {len(priority_samples)} priority samples + {len(quartile_samples)} quartile samples")

        priority_results_by_reranker = {}
        for reranker_name, (model, tokenizer) in rerankers.items():
            self.logger.info(f"  Evaluating {reranker_name} with priority samples")
            reranker_results = []

            for C, lambda_val, iterations, sample_type in priority_samples:
                start_time = time.time()

                mcts = MCTS(
                    nodes_query=nodes,
                    model=model,
                    tokenizer=tokenizer,
                    C=C,
                    lambda_=lambda_val,
                    budget=budget
                )
                best_node = mcts.search(question, max_iterations=iterations)
                context = best_node.concat_text

                rouge_score, answer = self.evaluate_config(question, context, ground_truth)

                elapsed = time.time() - start_time

                result = {
                    'reranker': reranker_name,
                    'C': C,
                    'lambda': lambda_val,
                    'iterations': iterations,
                    'score': rouge_score,
                    'context': context,
                    'answer': answer,
                    'time': elapsed,
                    'sample_type': sample_type
                }
                reranker_results.append(result)

                self.logger.info(f"    C={C:.1f}, λ={lambda_val:.2f}, iter={iterations}: ROUGE-L={rouge_score:.3f}")

            all_results.extend(reranker_results)
            priority_results_by_reranker[reranker_name] = reranker_results

        best_priority_result = max(all_results, key=lambda x: x['score'])
        best_reranker_name = best_priority_result['reranker']

        self.logger.info(f"Best priority result: {best_reranker_name} with C={best_priority_result['C']:.1f}, "
                        f"λ={best_priority_result['lambda']:.2f}, iter={best_priority_result['iterations']}, "
                        f"ROUGE-L={best_priority_result['score']:.3f}")

        self.logger.info(f"  Refining {best_reranker_name} with quartile supplements")
        model, tokenizer = rerankers[best_reranker_name]

        for C, lambda_val, iterations, sample_type in quartile_samples:
            start_time = time.time()

            mcts = MCTS(
                nodes_query=nodes,
                model=model,
                tokenizer=tokenizer,
                C=C,
                lambda_=lambda_val,
                budget=budget
            )
            best_node = mcts.search(question, max_iterations=iterations)
            context = best_node.concat_text

            rouge_score, answer = self.evaluate_config(question, context, ground_truth)
            elapsed = time.time() - start_time

            result = {
                'reranker': best_reranker_name,
                'C': C,
                'lambda': lambda_val,
                'iterations': iterations,
                'score': rouge_score,
                'context': context,
                'answer': answer,
                'time': elapsed,
                'sample_type': sample_type
            }
            all_results.append(result)

            self.logger.info(f"    [Quartile] C={C:.1f}, λ={lambda_val:.2f}, iter={iterations}: ROUGE-L={rouge_score:.3f}")

        # Find best coarse result
        best_coarse = max(all_results, key=lambda x: x['score'])
        self.logger.info(f"Best coarse: {best_coarse['reranker']} with C={best_coarse['C']}, "
                        f"λ={best_coarse['lambda']}, ROUGE-L={best_coarse['score']:.3f}")

        # Phase 2: Fine grid around best coarse result
        self.logger.info("Phase 2: Fine grid search around best configuration")
        fine_results = []

        best_reranker = best_coarse['reranker']
        model, tokenizer = rerankers[best_reranker]
        c_range = np.linspace(
            max(2.0, best_coarse['C'] - 0.1),
            min(3.0, best_coarse['C'] + 0.1),
            5
        )
        lambda_range = np.linspace(
            max(0.15, best_coarse['lambda'] - 0.02),
            min(0.3, best_coarse['lambda'] + 0.02),
            5
        )
        iter_range = range(
            max(5, best_coarse['iterations'] - 5),
            min(25, best_coarse['iterations'] + 5),
            5
        )

        for C in c_range:
            for lambda_val in lambda_range:
                for iterations in iter_range:
                    start_time = time.time()

                    mcts = MCTS(
                        nodes_query=nodes,
                        model=model,
                        tokenizer=tokenizer,
                        C=float(C),
                        lambda_=float(lambda_val),
                        budget=budget
                    )
                    best_node = mcts.search(question, max_iterations=int(iterations))
                    context = best_node.concat_text

                    # Evaluate with ROUGE-L
                    rouge_score, answer = self.evaluate_config(question, context, ground_truth)

                    elapsed = time.time() - start_time

                    result = {
                        'reranker': best_reranker,
                        'C': float(C),
                        'lambda': float(lambda_val),
                        'iterations': int(iterations),
                        'score': rouge_score,
                        'context': context,
                        'answer': answer,
                        'time': elapsed
                    }
                    fine_results.append(result)

                    self.logger.info(f"  Fine: C={C:.2f}, λ={lambda_val:.3f}, iter={iterations}: ROUGE-L={rouge_score:.3f}")

        all_results.extend(fine_results)
        best_result = max(all_results, key=lambda x: x['score'])

        self.logger.info(f"Best overall: {best_result['reranker']} with C={best_result['C']:.2f}, "
                        f"λ={best_result['lambda']:.3f}, iter={best_result['iterations']}, "
                        f"ROUGE-L={best_result['score']:.3f}")
        reranker_scores = {}
        best_c = best_result['C']
        best_lambda = best_result['lambda']
        best_iter = best_result['iterations']

        for reranker_name, (model, tokenizer) in rerankers.items():
            mcts = MCTS(
                nodes_query=nodes,
                model=model,
                tokenizer=tokenizer,
                C=best_c,
                lambda_=best_lambda,
                budget=budget
            )
            best_node = mcts.search(question, max_iterations=best_iter)
            context = best_node.concat_text
            rouge_score, _ = self.evaluate_config(question, context, ground_truth)
            reranker_scores[reranker_name] = rouge_score

        return {
            'best_config': best_result,
            'reranker_scores': reranker_scores,
            'all_results': all_results,
            'ground_truth': ground_truth
        }


def get_dense_embeddings(texts: List[str], model: BGEM3FlagModel) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    outputs = model.encode(
        texts,
        batch_size=len(texts),
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )

    embeddings = outputs['dense_vecs']

    if len(texts) == 1:
        return embeddings[0]
    return embeddings


def main():
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'wikipassage_smart_grid.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('wikipassage_smart_grid')
    logger.info("Starting smart grid search for WikiPassage with dense embeddings and ROUGE-L scoring")

    # Load data
    passages_file = PROJECT_ROOT / 'data' / 'wikipassage' / 'document_passages.json'
    test_file = PROJECT_ROOT / 'data' / 'wikipassage' / 'test.tsv'
    faiss_dir = PROJECT_ROOT / 'data' / 'wikipassage' / 'faiss_index_backup'

    with open(passages_file, 'r') as f:
        passages_dict = json.load(f)

    parser = argparse.ArgumentParser(description='WikiPassage smart grid search')
    parser.add_argument('--max-queries', type=int, default=1,
                        help='Maximum number of queries to process (default: 1)')
    args = parser.parse_args()

    wiki_df = pd.read_csv(test_file, sep='\t')

    if args.max_queries:
        wiki_df = wiki_df.head(args.max_queries)
        logger.info(f"Processing {len(wiki_df)} queries (limited by --max-queries={args.max_queries})")
    else:
        logger.info(f"Processing all {len(wiki_df)} queries")

    logger.info("Loading models...")
    model_loader = ModelLoader(llm_str='openai/gpt-4o')
    (
        llm,
        jina_v1, jina_v2,
        flag_v2, flag_large, flag_base,
        gte_model, gte_tokenizer
    ) = model_loader.get_all_models()

    rerankers = {
        'jina-reranker-v1-turbo-en': (jina_v1, None),
        'jina-reranker-v2-base-multilingual': (jina_v2, None),
        'bge-reranker-v2-m3': (flag_v2, None),
        'bge-reranker-large': (flag_large, None),
        'bge-reranker-base': (flag_base, None),
        'gte-multilingual-reranker-base': (gte_model, gte_tokenizer),
    }

    logger.info("Loading BGE-M3 for dense embeddings...")
    embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    logger.info("Loading FAISS index...")
    index_file = faiss_dir / 'faiss_index.bin'
    metadata_file = PROJECT_ROOT / 'data' / 'wikipassage' / 'faiss_index' / 'chunk_metadata.pkl'

    if not index_file.exists():
        logger.error(f"FAISS index not found at {index_file}")
        logger.error("Please run: python data/build_faiss_index.py")
        return

    if not metadata_file.exists():
        logger.error(f"Metadata file not found at {metadata_file}")
        return

    faiss_index = faiss.read_index(str(index_file))
    with open(metadata_file, 'rb') as f:
        chunk_metadata = pickle.load(f)

    grid_search = SmartGridSearch(logger, llm, passages_dict)
    training_records = []

    for idx, row in wiki_df.iterrows():
        question = row['Question']
        doc_id = str(row['DocumentID'])
        doc_name = row.get('DocumentName', '')
        relevant_passages = str(row.get('RelevantPassages', ''))

        logger.info(f"\nProcessing query {idx+1}/{len(wiki_df)}: {question[:50]}...")

        logger.info("Generating dense embedding for question...")
        question_embedding = get_dense_embeddings(question, embed_model)

        logger.info("Retrieving top-60 chunks using FAISS index...")
        query_vec = question_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)

        top_k = 60
        scores, indices = faiss_index.search(query_vec, top_k)

        from types import SimpleNamespace
        nodes = []
        chunk_texts = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunk_metadata):
                continue
            meta = chunk_metadata[idx]
            chunk_text = meta['text']
            chunk_texts.append(chunk_text)

            node = SimpleNamespace(text=chunk_text)
            nodes.append(SimpleNamespace(node=node))

        logger.info(f"Retrieved {len(nodes)} chunks")

        logger.info("Generating dense embeddings for retrieved chunks...")
        if chunk_texts:
            batch_size = 10
            chunk_embeddings = []
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i+batch_size]
                batch_embs = get_dense_embeddings(batch, embed_model)
                if batch_embs.ndim == 1:
                    batch_embs = batch_embs.reshape(1, -1)
                chunk_embeddings.append(batch_embs)

            all_chunk_embeddings = np.vstack(chunk_embeddings)
            nodes_query_embedding = np.mean(all_chunk_embeddings, axis=0)
        else:
            nodes_query_embedding = np.zeros_like(question_embedding)

        logger.info("Running smart grid search with ROUGE-L evaluation...")
        search_results = grid_search.search(
            question=question,
            nodes=nodes,
            rerankers=rerankers,
            doc_id=doc_id,
            relevant_passages=relevant_passages,
            budget=1024
        )

        best_config = search_results['best_config']
        reranker_scores = search_results['reranker_scores']

        record = {
            'QID': idx,
            'Question': question,
            'DocumentID': int(doc_id) if doc_id.isdigit() else 0,
            'DocumentName': doc_name,
            'RelevantPassages': relevant_passages,
            'nodes_query': nodes,
            'jina-reranker-v1-turbo-en': reranker_scores.get('jina-reranker-v1-turbo-en', 0.0),
            'bge-reranker-v2-m3': reranker_scores.get('bge-reranker-v2-m3', 0.0),
            'jina-reranker-v2-base-multilingual': reranker_scores.get('jina-reranker-v2-base-multilingual', 0.0),
            'bge-reranker-large': reranker_scores.get('bge-reranker-large', 0.0),
            'bge-reranker-base': reranker_scores.get('bge-reranker-base', 0.0),
            'gte-multilingual-reranker-base': reranker_scores.get('gte-multilingual-reranker-base', 0.0),
            'question_embedding': question_embedding.tolist(),
            'nodes_query_embedding': nodes_query_embedding.tolist(),
            'max_reranker': best_config['reranker'],
            'C_param': best_config['C'],
            'iteration': best_config['iterations'],
            'lambda': best_config['lambda']
        }

        training_records.append(record)

        logger.info(f"Best configuration for this query:")
        logger.info(f"  Reranker: {best_config['reranker']}")
        logger.info(f"  C: {best_config['C']:.2f}")
        logger.info(f"  Lambda: {best_config['lambda']:.3f}")
        logger.info(f"  Iterations: {best_config['iterations']}")
        logger.info(f"  ROUGE-L Score: {best_config['score']:.3f}")
        logger.info(f"  Best Answer Preview: {best_config['answer'][:100]}...")

    training_df = pd.DataFrame(training_records)

    output_file = PROJECT_ROOT / 'data' / 'wikipassage' / 'wikipassage_training_data.pkl'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(training_df, f)

    logger.info(f"\nSaved training data to {output_file}")
    logger.info(f"Total records: {len(training_df)}")

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Processed {len(training_df)} queries")
    logger.info(f"Best rerankers distribution:")
    logger.info(training_df['max_reranker'].value_counts())
    logger.info(f"\nAverage best parameters:")
    logger.info(f"  C: {training_df['C_param'].mean():.2f} (±{training_df['C_param'].std():.2f})")
    logger.info(f"  Lambda: {training_df['lambda'].mean():.3f} (±{training_df['lambda'].std():.3f})")
    logger.info(f"  Iterations: {training_df['iteration'].mean():.1f} (±{training_df['iteration'].std():.1f})")

    logger.info(f"\nAverage ROUGE-L scores by reranker:")
    for reranker in rerankers.keys():
        if reranker in training_df.columns:
            avg_score = training_df[reranker].mean()
            logger.info(f"  {reranker}: {avg_score:.3f}")

    logger.info("\nTraining data saved successfully with dense embeddings and ROUGE-L scores!")


if __name__ == '__main__':
    main()