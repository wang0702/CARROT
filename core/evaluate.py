import ast
import concurrent
import itertools
import json
import logging
import os
import uuid
from statistics import mean

import numpy as np
import pandas as pd
import requests
import torch
from FlagEmbedding import FlagReranker
from datasets import load_dataset
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core import load_index_from_storage, StorageContext, QueryBundle
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from rouge import Rouge
from rouge_score import rouge_scorer
from tqdm import tqdm

from core import config
# Optional imports - comment out if not needed
try:
    from NodeAgent import initialize_query_engine
    from NodeAgent3 import NodeClassifier
    from NodeSelector import NodeSelector
    from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, SBertEmbeddingModel
    from raptor.QAModels import GroqQAModelV2
    from raptor.SummarizationModels import GroqSummarizationModel
    from raptor.utils import HTTPHandler, calculate_max_rouge_scores, calculate_rouge_scores
except ImportError:
    pass  # These are optional for the main evaluation scripts

os.environ["OPENAI_API_KEY"] = config.API_KEY_1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def summarize_documents(selector, question, nodes):
    summarize_topk_base = f"Question: {question}\n"
    for item in nodes:
        item_str = selector.get_formatted_sources_without_question([item])
        item_summarized = selector.summarize_text(text=item_str)
        summarize_topk_base += f"Document: {item_summarized}\n"
    return summarize_topk_base


def evaluate_model_base(filepath, logger):
    df = pd.read_csv(filepath).iloc[1520:1530]
    df['nodes_ranking'] = df['nodes_ranking'].apply(ast.literal_eval)
    topk_rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    selector = NodeSelector()

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="QA rows"):
        question = row['question']
        answer = row['answer']
        nodes = row['nodes_ranking'][:7]

        summarize_topk_base = summarize_documents(selector, question, nodes)
        best_choice_res_base = selector.send_request_to_openai(prompt=summarize_topk_base)
        rouge_score_base = scorer.score(answer, best_choice_res_base)
        for rouge_metric in topk_rouge_scores.keys():
            if rouge_score_base[rouge_metric] != 1:
                topk_rouge_scores[rouge_metric].append(rouge_score_base[rouge_metric].fmeasure)

        logger.info(f'Best choice base score: {rouge_score_base}')

    average_best_choice_base = {k: mean(v) for k, v in topk_rouge_scores.items()}
    logger.info(f'Average base topk retrieval Rouge-1 score: {average_best_choice_base["rouge1"]}')
    logger.info(f'Average base topk retrieval Rouge-2 score: {average_best_choice_base["rouge2"]}')
    logger.info(f'Average base topk retrieval Rouge-L score: {average_best_choice_base["rougeL"]}')


def process_combinations(question, nodes, classifier):
    combinations = NodeSelector.generate_combinations(nodes=nodes)
    top_choices = []

    for combination in tqdm(combinations, desc="Processing Combinations"):
        formatted_sources = NodeSelector.get_formatted_sources(question=question, node_combination=combination)
        rouge_score = classifier.predict_rouge(formatted_sources)
        rouge_score = float(rouge_score)
        top_choices.append((combination, rouge_score))
        top_choices = sorted(top_choices, key=lambda x: x[1], reverse=True)[:5]

    return top_choices


def evaluate_model_t5(filepath, model_path, logger):
    df = pd.read_csv(filepath).iloc[1520:1530]
    df['nodes_ranking'] = df['nodes_ranking'].apply(ast.literal_eval)
    topk_rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    classifier = NodeClassifier(filepath)
    classifier.load_model(model_path)

    selector = NodeSelector()

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="QA rows"):
        question = row['question']
        answer = row['answer']
        nodes = row['nodes_ranking'][:7]

        best_choices = process_combinations(question, nodes, classifier)

        summarize_topk_base = f"Question: {question}\n"
        for best_choice, _ in best_choices:
            for item in best_choice:
                item_str = selector.get_formatted_sources_without_question([item])
                item_summarized = selector.summarize_text(text=item_str)
                summarize_topk_base += f"Document: {item_summarized}\n"

        best_choice_res_base = selector.send_request_to_openai(prompt=summarize_topk_base)
        rouge_score_base = scorer.score(answer, best_choice_res_base)
        for rouge_metric in topk_rouge_scores.keys():
            if rouge_score_base[rouge_metric] != 1:
                topk_rouge_scores[rouge_metric].append(rouge_score_base[rouge_metric].fmeasure)

        logger.info(f'Best choice T5 score: {rouge_score_base}')

    average_best_choice_base = {k: mean(v) for k, v in topk_rouge_scores.items()}
    logger.info(f'Average T5 model retrieval Rouge-1 score: {average_best_choice_base["rouge1"]}')
    logger.info(f'Average T5 model retrieval Rouge-2 score: {average_best_choice_base["rouge2"]}')
    logger.info(f'Average T5 model retrieval Rouge-L score: {average_best_choice_base["rougeL"]}')


def evaluate_model_gru(filepath, logger):
    # Similar implementation to T5 but using a GRU model
    # This is a placeholder for your actual GRU-based method
    pass


def load_data_and_embedding():
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    df = pd.read_csv(
        '/home/wang/Downloads/auto_tuning4rag/raptor/tree_index_llama3/llama3_train_set/evaluation_results_without_embeddings.csv')
    filtered_df = df[df['rougeL'] != 0].copy()  # 使用 .copy() 方法避免 SettingWithCopyWarning

    context_embeddings = []

    for index, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0], desc="Encoding contexts"):
        context = row['context']
        embedding = model.encode([context],
                                 batch_size=1,
                                 max_length=8192,
                                 )['dense_vecs']
        context_embeddings.append(embedding[0])  # Assuming 'dense_vecs' is a list of embeddings

    filtered_df['context_embedding'] = context_embeddings

    # 保存新的 DataFrame 到 pickle 文件
    filtered_df.to_pickle(
        '/home/wang/Downloads/auto_tuning4rag/raptor/tree_index_llama3/llama3_train_set/evaluation_results_with_embeddings.pkl')

    print("Embeddings added and DataFrame saved.")


def evaluate_tree(start_index=0, end_index=50, logger=None):
    url = 'http://www.pushplus.plus/send'

    df = pd.read_pickle('narrativeqa_data.pkl')
    df = df[start_index:end_index]

    embedding_model = SBertEmbeddingModel()

    RAC = RetrievalAugmentationConfig(
        summarization_model=GroqSummarizationModel(),
        qa_model=GroqQAModelV2(),
        embedding_model=embedding_model
    )

    results = []
    baseline_rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    ours_rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="QA rows"):
        actual_index = index + start_index
        print(f'Processing index: {actual_index}')

        id_value = row['id']
        pkl_path = f"/home_nfs/ziting/auto_tuning4rag/raptor/tree_index_llama3/{id_value}.pkl"

        if not os.path.exists(pkl_path):
            print(f"File {pkl_path} does not exist. Skipping.")
            continue

        # Initialize RetrievalAugmentation with the specified tree path
        RA = RetrievalAugmentation(tree=pkl_path, config=RAC)

        question = row['question']
        answer_gt = row['answer']

        best_answer, best_base_rouge_scores, best_rouge_scores = RA.answer_question_with_combinations(
            question=question['text'],
            answer_ground_truth=answer_gt,
            logger=logger
        )

        if best_answer is None:
            continue

        for key in baseline_rouge_scores.keys():
            baseline_rouge_scores[key].append(best_base_rouge_scores[key])
            ours_rouge_scores[key].append(best_rouge_scores[key])

        results.append({
            'index': actual_index,
            'question': question,
            'ground_truth': answer_gt,
            'best_answer': best_answer
        })

    average_baseline_rouge = {k: np.mean(v) for k, v in baseline_rouge_scores.items()}
    average_ours_rouge = {k: np.mean(v) for k, v in ours_rouge_scores.items()}

    if logger:
        logger.info(f"Average Baseline ROUGE scores: {average_baseline_rouge}")
        logger.info(f"Average Ours ROUGE scores: {average_ours_rouge}")

    print(f"Average Baseline ROUGE scores: {average_baseline_rouge}")
    print(f"Average Ours ROUGE scores: {average_ours_rouge}")

    return results


def save_combinations_only(logger, start_index, end_index):
    try:
        df = load_df_base_id()
        df = df[start_index:end_index]
        embedding_model = SBertEmbeddingModel()

        RAC = RetrievalAugmentationConfig(
            summarization_model=GroqSummarizationModel(),
            qa_model=GroqQAModelV2(),
            embedding_model=embedding_model
        )
        results = []

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        for index, (i, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"),
                                         start=start_index):
            try:
                print('Processing index:', i)
                question = row['question']
                answer_gt = row['answer']  # Ground truth answer should be a list of dictionaries
                doc_id = row['document']['id']
                reference = row['document']['text']

                # Load the saved tree for the current index
                RA = RetrievalAugmentation(
                    tree=f'/home/wang/Downloads/auto_tuning4rag/raptor/tree_index_llama3_id/{doc_id}.pkl',
                    config=RAC
                )
                res_baseline = RA.answer_question(question=question['text'], top_k=8)

                # Use the abstracted function to calculate max ROUGE scores
                baseline_rouge1, baseline_rouge2, baseline_rougeL = calculate_max_rouge_scores(res_baseline, answer_gt,
                                                                                               scorer)

                # Get train combinations
                res_df = RA.get_train_combinations(question=question['text'], answer_ground_truth=answer_gt, top_k=10,
                                                   logger=logger)

                # Find the row with the maximum rougeL score
                max_rougeL_row = res_df.loc[res_df['rougeL'].idxmax()]

                # Extract the max ROUGE scores from the row with the highest rougeL
                max_rouge1 = max_rougeL_row['rouge1']
                max_rouge2 = max_rougeL_row['rouge2']
                max_rougeL = max_rougeL_row['rougeL']

                # Log the baseline and max ROUGE scores
                log_messages = (
                    f"Baseline ROUGE scores for index {i}: rouge1={baseline_rouge1}, rouge2={baseline_rouge2}, rougeL={baseline_rougeL}\n"
                    f"Max rougeL combination row for index {i}: rouge1={max_rouge1}, rouge2={max_rouge2}, rougeL={max_rougeL}\n"
                )
                logger.info(log_messages)

                # Add a unique batch_id to the DataFrame
                batch_id = str(uuid.uuid4())
                res_df['batch_id'] = batch_id

                # Add the reference text only to the first row of the DataFrame
                res_df.loc[0, 'reference'] = reference

                results.append(res_df)
            except Exception as e:
                logger.error(f"Error processing index {i}: {e}")

        # Save any remaining results
        if results:
            results_df = pd.concat(results, ignore_index=True)
            results_df.to_csv(
                f'/home/wang/Downloads/auto_tuning4rag/raptor/tree_index_llama3/llama3_train_set/new_train_set_with_embeddings_{start_index}_{end_index}.csv',
                index=False
            )

    except Exception as e:
        logger.error(f"Error in save_combinations_only function: {e}")


def save_combinations_only(logger, start_index, end_index):
    try:
        df = load_df_base_id()
        df = df[start_index:end_index]
        embedding_model = SBertEmbeddingModel()

        RAC = RetrievalAugmentationConfig(
            summarization_model=GroqSummarizationModel(),
            qa_model=GroqQAModelV2(),
            embedding_model=embedding_model
        )
        results = []

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        for index, (i, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"),
                                         start=start_index):
            try:
                print('Processing index:', i)
                question = row['question']
                answer_gt = row['answer']  # Ground truth answer should be a list of dictionaries
                doc_id = row['document']['id']
                reference = row['document']['text']

                # Load the saved tree for the current index
                RA = RetrievalAugmentation(
                    tree=f'/home/wang/Downloads/auto_tuning4rag/raptor/tree_index_llama3_id/{doc_id}.pkl',
                    config=RAC
                )
                res_baseline = RA.answer_question(question=question['text'], top_k=8)

                # Use the abstracted function to calculate max ROUGE scores
                baseline_rouge1, baseline_rouge2, baseline_rougeL = calculate_max_rouge_scores(res_baseline, answer_gt,
                                                                                               scorer)

                # Get train combinations
                res_df = RA.get_train_combinations(question=question['text'], answer_ground_truth=answer_gt, top_k=10,
                                                   logger=logger)

                # Find the row with the maximum rougeL score
                max_rougeL_row = res_df.loc[res_df['rougeL'].idxmax()]

                # Extract the max ROUGE scores from the row with the highest rougeL
                max_rouge1 = max_rougeL_row['rouge1']
                max_rouge2 = max_rougeL_row['rouge2']
                max_rougeL = max_rougeL_row['rougeL']

                # Log the baseline and max ROUGE scores
                log_messages = (
                    f"Baseline ROUGE scores for index {i}: rouge1={baseline_rouge1}, rouge2={baseline_rouge2}, rougeL={baseline_rougeL}\n"
                    f"Max rougeL combination row for index {i}: rouge1={max_rouge1}, rouge2={max_rouge2}, rougeL={max_rougeL}\n"
                )
                logger.info(log_messages)

                # Add a unique batch_id to the DataFrame
                batch_id = str(uuid.uuid4())
                res_df['batch_id'] = batch_id

                # Add the reference text only to the first row of the DataFrame
                res_df.loc[0, 'reference'] = reference

                results.append(res_df)
            except Exception as e:
                logger.error(f"Error processing index {i}: {e}")

        # Save any remaining results
        if results:
            results_df = pd.concat(results, ignore_index=True)
            results_df.to_csv(
                f'/home/wang/Downloads/auto_tuning4rag/raptor/tree_index_llama3/llama3_train_set/new_train_set_with_embeddings_{index - (index % 1000)}_{index}.csv',
                index=False
            )

    except Exception as e:
        logger.error(f"Error in save_combinations_only function: {e}")


def load_tree_and_evaluate_subset(logger, enable_combinations=False):
    dataset = load_dataset("deepmind/narrativeqa")
    df_test = dataset['test'].to_pandas()
    df_test = df_test[230:250]

    embedding_model = SBertEmbeddingModel()

    RAC = RetrievalAugmentationConfig(
        summarization_model=GroqSummarizationModel(),
        qa_model=GroqQAModelV2(),
        embedding_model=embedding_model
    )

    results = []
    baseline_rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    ours_rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    hyde_rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    combinations_rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}

    for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="QA rows"):
        llm = Groq(model="llama3-70b-8192", api_key=config.API_KEY_GROQ)

        document_id = row['document']['id']
        print(f'Processing document ID: {document_id}')
        document = row['document']['text']
        question = row['question']
        answer_gt = row['answers'].tolist()

        tree_path = f'/home/wang/Downloads/auto_tuning4rag/raptor/tree_llama3_test_narr/{document_id}.pkl'
        RA = RetrievalAugmentation(
            tree=tree_path,
            config=RAC
        )

        best_answer_mlp, best_base_rouge_scores, best_rouge_scores_mlp, best_rouge_scores_hyde, best_rouge_scores_combinations = RA.answer_question_with_combinations(
            question=question['text'],
            answer_ground_truth=answer_gt,
            logger=logger
        )

        if best_answer_mlp is None:
            continue

        # Skip if either best_base_rouge_scores or best_rouge_scores has a RougeL score of 0
        if best_base_rouge_scores['rouge-l'] == 0 or best_rouge_scores_mlp['rouge-l'] == 0:
            continue

        for key in baseline_rouge_scores.keys():
            baseline_rouge_scores[key].append(best_base_rouge_scores[key])
            ours_rouge_scores[key].append(best_rouge_scores_mlp[key])
            hyde_rouge_scores[key].append(best_rouge_scores_hyde[key])
            if enable_combinations:
                combinations_rouge_scores[key].append(best_rouge_scores_combinations[key])

        results.append({
            'index': document_id,
            'question': question,
            'ground_truth': answer_gt,
            'best_answer': best_answer_mlp
        })

    average_baseline_rouge = {k: np.mean(v) for k, v in baseline_rouge_scores.items()}
    average_ours_rouge = {k: np.mean(v) for k, v in ours_rouge_scores.items()}
    average_hyde_rouge = {k: np.mean(v) for k, v in hyde_rouge_scores.items()}
    average_combinations_rouge = {k: np.mean(v) for k, v in
                                  combinations_rouge_scores.items()} if enable_combinations else None

    logger.info(f"Average Baseline ROUGE scores: {average_baseline_rouge}")
    logger.info(f"Average Ours ROUGE scores: {average_ours_rouge}")
    logger.info(f"Average HYDE ROUGE scores: {average_hyde_rouge}")
    if enable_combinations:
        logger.info(f"Average Combinations ROUGE scores: {average_combinations_rouge}")

    print(f"Average Baseline ROUGE scores: {average_baseline_rouge}")
    print(f"Average Ours ROUGE scores: {average_ours_rouge}")
    print(f"Average HYDE ROUGE scores: {average_hyde_rouge}")
    if enable_combinations:
        print(f"Average Combinations ROUGE scores: {average_combinations_rouge}")

    return results


class MCTS:
    def __init__(self, reranker, iterations=1000):
        self.reranker = reranker
        self.iterations = iterations

    def selection(self, node):
        while node.is_fully_expanded():
            node = node.best_child()
        return node

    def expansion(self, node, possible_combinations):
        expanded_nodes = []
        if node.state is None:
            # Initialize the root node with the first combination
            expanded_nodes.append(node.expand(possible_combinations[0]))
        else:
            for combination in possible_combinations:
                if combination not in [child.state for child in node.children]:
                    expanded_nodes.append(node.expand(combination))
                    if len(expanded_nodes) >= 3:  # Example: expand up to 3 combinations
                        break
        return expanded_nodes

    def simulation(self, combination, question):
        scores = []
        for state in combination:
            score = self.reranker.compute_score([[question, state]])
            scores.append(score)
        total_score = np.sum(scores)
        return total_score

    def backtracking(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self, root, question, possible_combinations):
        for _ in range(self.iterations):
            node = self.selection(root)
            expanded_nodes = self.expansion(node, possible_combinations)
            for expanded_node in expanded_nodes:
                if expanded_node.state is not None:
                    reward = self.simulation(expanded_node.state, question)
                    self.backtracking(expanded_node, reward)

        best_node = root.best_child(c_param=0)
        return best_node


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        if self.state is None:
            return False
        return len(self.children) == len(self.state)

    def best_child(self, c_param=1.2):
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, state):
        child_node = MCTSNode(state, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward


def process_node(best_node, question, llm, use_llama3, gt_answer, rouge):
    combined_text = " ".join(best_node.state)
    if use_llama3:
        res_vector = llm.answer_question(context=combined_text, question=question)
    else:
        res_vector = llm.complete(
            f"Provide a short and precise answer to the question: {question} Based on given Context: {combined_text}")
        res_vector = res_vector.text

    scores = rouge.get_scores(res_vector, gt_answer)
    return {
        "combined_text": combined_text,
        "res_vector": res_vector,
        "scores": scores
    }


def process_question(row, passages, query_engine_vector, llm, use_llama3, reranker, rouge, c_param):
    question = row['Question']
    relevant_passages = str(row['RelevantPassages']).split(',')

    gt_doc = " ".join([passages[str(row['DocumentID'])][passage_id.strip()] for passage_id in relevant_passages])

    if use_llama3:
        gt_answer = llm.answer_question(context=gt_doc, question=question)
    else:
        gt_answer_com = llm.complete(
            f"Provide a short and precise answer to the question: {question} Based on given Context: {gt_doc} ")
        gt_answer = gt_answer_com.text

    question_bundle = QueryBundle(question)
    nodes = query_engine_vector.retrieve(question_bundle)

    # Initialize MCTS root node
    root = MCTSNode(state=None)
    possible_states = [node.text for node in nodes]
    possible_combinations = list(itertools.combinations(possible_states, 3))

    # Perform MCTS search
    mcts = MCTS(reranker)
    best_node = mcts.search(root, question, possible_combinations)

    result = process_node(best_node, question, llm, use_llama3, gt_answer, rouge)
    combined_text = result["combined_text"]
    res_vector = result["res_vector"]
    scores = result["scores"]

    return {
        "question": question,
        "combined_text": combined_text,
        "res_vector": res_vector,
        "gt_answer": gt_answer,
        "scores": scores
    }


def evaluate_wiki(logger, build_index=False, use_llama3=False):
    # Load the dataset
    wiki_df = pd.read_csv('./WikiPassageQA/test.tsv', sep='\t')
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", embed_batch_size=16)

    with open('./WikiPassageQA/document_passages.json', 'r', encoding='utf-8') as file:
        passages = json.load(file)

    if build_index:
        # Initialize document processing
        documents = SimpleDirectoryReader('/home/wang/Downloads/auto_tuning4rag/scripts').load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        llm = Ollama(model="llama2", request_timeout=90.0)
        Settings.llm = llm
        reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

        index = VectorStoreIndex.from_documents(documents=documents)
        index.storage_context.persist(persist_dir='./WikiPassageQA/vector_index_bge')
    else:
        storage_context = StorageContext.from_defaults(persist_dir='./WikiPassageQA/vector_index_bge')
        index = load_index_from_storage(storage_context)

    query_engine_vector = initialize_query_engine(index)

    if use_llama3:
        llm = GroqQAModelV2()
    else:
        llm = Ollama(model="llama2", request_timeout=90.0)

    rouge = Rouge()
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    def process_row(row, c_param, passages, query_engine_vector, llm, use_llama3, reranker, rouge):
        result = process_question(row, passages, query_engine_vector, llm, use_llama3, reranker, rouge, c_param)
        question = result["question"]
        combined_text = result["combined_text"]
        res_vector = result["res_vector"]
        gt_answer = result["gt_answer"]
        scores = result["scores"]

        return {
            "question": question,
            "combined_text": combined_text,
            "res_vector": res_vector,
            "gt_answer": gt_answer,
            "scores": scores
        }

    c_params = [0.5, 1.0, 1.5, 2.0]  # Example c_param values

    for c_param in c_params:
        rouge1_f_scores = []
        rouge2_f_scores = []
        rougeL_f_scores = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_row, row, c_param, passages, query_engine_vector, llm, use_llama3, reranker,
                                rouge) for idx, row in wiki_df.iterrows()]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc=f"QA rows (c_param={c_param})"):
                result = future.result()
                question = result["question"]
                combined_text = result["combined_text"]
                res_vector = result["res_vector"]
                gt_answer = result["gt_answer"]
                scores = result["scores"]

                rouge1_f_scores.append(scores[0]['rouge-1']['f'])
                rouge2_f_scores.append(scores[0]['rouge-2']['f'])
                rougeL_f_scores.append(scores[0]['rouge-l']['f'])

                # Log the results
                logger.info(f"c_param: {c_param}")
                logger.info(f"Question: {question}")
                logger.info(f"Combined Text: {combined_text}")
                logger.info(f"LLM Response: {res_vector}")
                logger.info(f"Ground Truth Answer: {gt_answer}")
                logger.info(f"ROUGE Scores: {scores}")

        logger.info(f"ROUGE-1 F1 Score (c_param={c_param}): {np.mean(rouge1_f_scores)}")
        send_wechat(f"ROUGE-1 F1 Score (c_param={c_param}): {np.mean(rouge1_f_scores)}")

        logger.info(f"ROUGE-2 F1 Score (c_param={c_param}): {np.mean(rouge2_f_scores)}")
        send_wechat(f"ROUGE-2 F1 Score (c_param={c_param}): {np.mean(rouge2_f_scores)}")

        logger.info(f"ROUGE-L F1 Score (c_param={c_param}): {np.mean(rougeL_f_scores)}")
        send_wechat(f"ROUGE-L F1 Score (c_param={c_param}): {np.mean(rougeL_f_scores)}")


def evaluate_wiki_baseline(logger, build_index=False, use_llama3=False):
    # Load the dataset
    wiki_df = pd.read_csv('./WikiPassageQA/test.tsv', sep='\t')
    wiki_df = wiki_df[70:80]
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", embed_batch_size=16)

    with open('./WikiPassageQA/document_passages.json', 'r', encoding='utf-8') as file:
        passages = json.load(file)

    if build_index:
        # Initialize document processing
        documents = SimpleDirectoryReader('/home/wang/Downloads/auto_tuning4rag/scripts').load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        llm = Ollama(model="llama2", request_timeout=90.0)
        Settings.llm = llm

        index = VectorStoreIndex.from_documents(documents=documents)
        index.storage_context.persist(persist_dir='./WikiPassageQA/vector_index_bge')
    else:
        storage_context = StorageContext.from_defaults(persist_dir='./WikiPassageQA/vector_index_bge')
        index = load_index_from_storage(storage_context)

    query_engine_vector = initialize_query_engine(index)

    if use_llama3:
        llm = GroqQAModelV2()
    else:
        llm = Ollama(model="llama2", request_timeout=90.0)

    rouge = Rouge()

    rouge1_f_scores = []
    rouge2_f_scores = []
    rougeL_f_scores = []
    rouge1_f_scores_baseline = []
    rouge2_f_scores_baseline = []
    rougeL_f_scores_baseline = []
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    # reranker_layerWise = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise',
    #                                     use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    for idx, row in tqdm(wiki_df.iterrows(), total=wiki_df.shape[0], desc="QA rows"):
        question = row['Question']
        relevant_passages = str(row['RelevantPassages']).split(',')

        gt_doc = " ".join([passages[str(row['DocumentID'])][passage_id.strip()] for passage_id in relevant_passages])

        if use_llama3:
            gt_answer = llm.answer_question(context=gt_doc, question=question)
        else:
            gt_answer_com = llm.complete(
                f"Provide precise answer to the question: {question} Based on given Context: {gt_doc} ")
            gt_answer = gt_answer_com.text

        question_bundle = QueryBundle(question)
        nodes = query_engine_vector.retrieve(question_bundle)

        # Rerank nodes
        scores = reranker.compute_score([[question, node.text] for node in nodes])
        # scores = reranker_layerWise.compute_score([[question, node.text] for node in nodes], cutoff_layers=[28])

        # Select top 3 nodes
        top_3_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        top_3_nodes = [nodes[i] for i in top_3_indices]

        # Generate prediction using the top 3 nodes
        combined_text = "  ".join([node.text for node in top_3_nodes])
        if use_llama3:
            res_vector = llm.answer_question(context=combined_text, question=question)
        else:
            res_vector = llm.complete(
                f"Provide a comprehensive and precise answer to the question: {question} Based on given Context: {combined_text}")
            res_vector = res_vector.text
            res_vector_baseline = query_engine_vector.query(question)

        scores = rouge.get_scores(res_vector, gt_answer, avg=True)
        scores_baseline = rouge.get_scores(res_vector_baseline.response, gt_answer, avg=True)
        rouge1_f_scores.append(scores['rouge-1']['f'])
        rouge2_f_scores.append(scores['rouge-2']['f'])
        rougeL_f_scores.append(scores['rouge-l']['f'])
        rouge1_f_scores_baseline.append(scores_baseline['rouge-1']['f'])
        rouge2_f_scores_baseline.append(scores_baseline['rouge-2']['f'])
        rougeL_f_scores_baseline.append(scores_baseline['rouge-l']['f'])

        logger.info(f"Question: {question}")
        logger.info(f"ROUGE Scores: {scores}")
        logger.info(f"Baseline ROUGE Scores: {scores_baseline}")

    avg_rouge1_f = sum(rouge1_f_scores) / len(rouge1_f_scores)
    avg_rouge2_f = sum(rouge2_f_scores) / len(rouge2_f_scores)
    avg_rougeL_f = sum(rougeL_f_scores) / len(rougeL_f_scores)

    avg_rouge1_f_baseline = sum(rouge1_f_scores_baseline) / len(rouge1_f_scores_baseline)
    avg_rouge2_f_baseline = sum(rouge2_f_scores_baseline) / len(rouge2_f_scores_baseline)
    avg_rougeL_f_baseline = sum(rougeL_f_scores_baseline) / len(rougeL_f_scores_baseline)

    logger.info("Average ROUGE scores:")
    logger.info(f"ROUGE-1 F1 Score: {avg_rouge1_f:.4f}")
    logger.info(f"ROUGE-2 F1 Score: {avg_rouge2_f:.4f}")
    logger.info(f"ROUGE-L F1 Score: {avg_rougeL_f:.4f}")

    logger.info("Average Baseline ROUGE scores:")
    logger.info(f"Baseline ROUGE-1 F1 Score: {avg_rouge1_f_baseline:.4f}")
    logger.info(f"Baseline ROUGE-2 F1 Score: {avg_rouge2_f_baseline:.4f}")
    logger.info(f"Baseline ROUGE-L F1 Score: {avg_rougeL_f_baseline:.4f}")



def evaluate_llama():
    wiki_df = pd.read_csv('./WikiPassageQA/train.tsv', sep='\t')
    with open('./WikiPassageQA/document_passages.json', 'r', encoding='utf-8') as file:
        passages = json.load(file)
    dataset = load_dataset("deepmind/narrativeqa")
    df_test = dataset['test'].to_pandas()
    df_test = df_test[0:20]

    llm = Groq(model="llama3-70b-8192", api_key=config.API_KEY_GROQ)
    rouge = Rouge()

    # Dictionaries to store rouge scores for vector and bm25 methods
    rouge_scores_vector = {
        'rouge-1': [],
        'rouge-2': [],
        'rouge-l': []
    }
    rouge_scores_bm25 = {
        'rouge-1': [],
        'rouge-2': [],
        'rouge-l': []
    }

    for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="QA rows"):
        document = row['document']['text']
        document_id = row['document']['id']
        question = row['question']['text']
        gt_answers = row['answers']
        gt_texts = [answer['text'] for answer in gt_answers]

        # Save document to file
        doc_path = f'/home/wang/Downloads/auto_tuning4rag/raptor/narr-txt/{document_id}.txt'
        with open(doc_path, 'w') as f:
            f.write(document)

        documents = SimpleDirectoryReader(
            '/home/wang/Downloads/auto_tuning4rag/raptor/narr-txt').load_data()
        splitter = SentenceSplitter(chunk_size=512)
        nodes = splitter.get_nodes_from_documents(documents)
        index = VectorStoreIndex.from_documents(documents)

        # Vector method
        query_engine_vector = initialize_query_engine(index)
        res_vector = query_engine_vector.query(question)
        res_vector = res_vector.response
        vector_rouge = calculate_rouge_scores(res_vector, gt_answers, rouge)
        rouge_scores_vector['rouge-1'].append(vector_rouge['rouge-1'])
        rouge_scores_vector['rouge-2'].append(vector_rouge['rouge-2'])
        rouge_scores_vector['rouge-l'].append(vector_rouge['rouge-l'])

        # BM25 method
        retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)
        reranker = SentenceTransformerRerank(top_n=10, model="BAAI/bge-reranker-base")

        query_engine_bm25 = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
            llm=llm
        )

        res_bm25 = query_engine_bm25.query(question)
        answer_bm25 = res_bm25.response

        bm25_rouge = calculate_rouge_scores(answer_bm25, gt_answers, rouge)
        rouge_scores_bm25['rouge-1'].append(bm25_rouge['rouge-1'])
        rouge_scores_bm25['rouge-2'].append(bm25_rouge['rouge-2'])
        rouge_scores_bm25['rouge-l'].append(bm25_rouge['rouge-l'])

        print(f"Question: {question}")
        print(f"Vector Method Answer: {res_vector}")
        print(f"BM25 Method Answer: {answer_bm25}")
        print(f"Ground Truths: {gt_texts}")
        print(f"Vector ROUGE Scores: {vector_rouge}")
        print(f"BM25 ROUGE Scores: {bm25_rouge}")

    # Display average ROUGE scores for vector method
    avg_rouge_vector_1 = sum(rouge_scores_vector['rouge-1']) / len(rouge_scores_vector['rouge-1'])
    avg_rouge_vector_2 = sum(rouge_scores_vector['rouge-2']) / len(rouge_scores_vector['rouge-2'])
    avg_rouge_vector_l = sum(rouge_scores_vector['rouge-l']) / len(rouge_scores_vector['rouge-l'])

    print(f"Average Vector ROUGE-1 F1 Score: {avg_rouge_vector_1}")
    print(f"Average Vector ROUGE-2 F1 Score: {avg_rouge_vector_2}")
    print(f"Average Vector ROUGE-L F1 Score: {avg_rouge_vector_l}")

    # Display average ROUGE scores for BM25 method
    avg_rouge_bm25_1 = sum(rouge_scores_bm25['rouge-1']) / len(rouge_scores_bm25['rouge-1'])
    avg_rouge_bm25_2 = sum(rouge_scores_bm25['rouge-2']) / len(rouge_scores_bm25['rouge-2'])
    avg_rouge_bm25_l = sum(rouge_scores_bm25['rouge-l']) / len(rouge_scores_bm25['rouge-l'])

    print(f"Average BM25 ROUGE-1 F1 Score: {avg_rouge_bm25_1}")
    print(f"Average BM25 ROUGE-2 F1 Score: {avg_rouge_bm25_2}")
    print(f"Average BM25 ROUGE-L F1 Score: {avg_rouge_bm25_l}")


# Make sure to define the missing functions and import necessary modules for the above script to work.

def raptor_evaluate():
    dataset = load_dataset("deepmind/narrativeqa")
    df_train = dataset['train'].to_pandas()
    document_id = '01aa10d75658840a478ede17631dba875651c370'
    # filter all rows with the document_id
    df_train = df_train[df_train['document']['id'] == document_id]
    RA = RetrievalAugmentation(tree='./raptor/tree_index_llama3_id/01aa10d75658840a478ede17631dba875651c370.pkl')
    answer = RA.answer_question(question=question)

def save_trees_raptor_id():

    dataset = load_dataset("deepmind/narrativeqa")
    df_train = dataset['train'].to_pandas()

    save_dir = "./raptor/tree_index_llama3_id"
    # / home / wang / Downloads / auto_tuning4rag / raptor / tree_llama3_test_narr
    os.makedirs(save_dir, exist_ok=True)

    for index, row in tqdm(df_train.iterrows(), total=df_train.shape[0], desc="Processing rows"):
        save_path = os.path.join(save_dir, f"{row['document']['id']}.pkl")
        if os.path.exists(save_path) or row['document']['file_size'] > 140000:
            continue  # Skip if the file already exists

        RAC = RetrievalAugmentationConfig(
            summarization_model=GroqSummarizationModel(),
            qa_model=GroqQAModelV2(),
            embedding_model=SBertEmbeddingModel()
        )

        # Initialize RetrievalAugmentation with the specified config
        RA = RetrievalAugmentation(config=RAC)
        RA.tree_builder.summarization_model = GroqSummarizationModel()

        document = row['document']

        document_final = f"""
        Document:
        =========
        {document['text']}xxx
        """

        # Use the RetrievalAugmentation instance as needed
        # For example, you might want to add the documents to the tree and answer the question
        RA.add_documents(document_final)
        res = RA.answer_question(question="What is the document about?")

        RA.save(save_path)
        print(f"Saved tree to {save_path}")


def load_df_base_id():
    # dataset = load_dataset("deepmind/narrativeqa")
    # df_train = dataset['train'].to_pandas()
    df_train = pd.read_pickle('narrativeqa_data.pkl')

    save_dir = "./raptor/tree_index_llama3_id"

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        logger.error(f"Save directory {save_dir} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if the directory does not exist

    # Function to check if the file exists
    def file_exists(row):
        file_path = os.path.join(save_dir, f"{row['document']['id']}.pkl")
        return os.path.exists(file_path)

    df_filtered = df_train[df_train.apply(file_exists, axis=1)]

    return df_filtered


def save_trees_raptor(start_index=412):
    # temp = pd.read_pickle('/home/wang/Downloads/auto_tuning4rag/raptor/tree_index_llama3/412.pkl')
    df = pd.read_pickle('narrativeqa_data.pkl')

    document_final = 'Documents:'
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="QA rows"):
        if index + 1 < start_index:
            continue  # Skip until the starting index is reached

        RAC = RetrievalAugmentationConfig(summarization_model=GroqSummarizationModel(), qa_model=GroqQAModelV2(),
                                          embedding_model=SBertEmbeddingModel())

        # Initialize RetrievalAugmentation with the specified config
        RA = RetrievalAugmentation(config=RAC)
        RA.tree_builder.summarization_model = GroqSummarizationModel()

        question = row['question']
        answer = row['answer']
        document = row['document']
        document_final = 'Documents: ' + document['text']

        # Use the RetrievalAugmentation instance as needed
        # For example, you might want to add the documents to the tree and answer the question
        RA.add_documents(document_final)

        SAVE_PATH = f"/home/wang/Downloads/auto_tuning4rag/raptor/tree_index_llama3/{index + 1}.pkl"
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        RA.save(SAVE_PATH)
        print(SAVE_PATH)


def evaluate_model_raptor(filepath, logger):
    topk_rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    df = pd.read_csv(filepath).iloc[1520:1530]
    df['nodes_ranking'] = df['nodes_ranking'].apply(ast.literal_eval)

    selector = NodeSelector()

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="QA rows"):
        question = row['question']
        answer = row['answer']
        nodes = row['nodes_ranking']
        formatted_sources = "\n".join([selector.get_formatted_sources_without_question([item]) for item in nodes])
        RA = RetrievalAugmentation()
        # RA.retriever = TreeRetrieverV2(self.tree_retriever_config, self.tree)
        RA.add_documents(formatted_sources)
        raptor_answer_list = RA.answer_question_with_combinations(question=question, answer_ground_truth=answer)
        raptor_answer = RA.answer_question(question=question)
        rouge_score_base = scorer.score(answer, raptor_answer)
        for rouge_metric in topk_rouge_scores.keys():
            if rouge_score_base[rouge_metric] != 1:
                topk_rouge_scores[rouge_metric].append(rouge_score_base[rouge_metric].fmeasure)

        logger.info(f'Best choice base score: {rouge_score_base}')

    average_best_choice_base = {k: mean(v) for k, v in topk_rouge_scores.items()}
    logger.info(f'Average Raptor retrieval Rouge-1 score: {average_best_choice_base["rouge1"]}')
    logger.info(f'Average Raptor retrieval Rouge-2 score: {average_best_choice_base["rouge2"]}')
    logger.info(f'Average Raptor retrieval Rouge-L score: {average_best_choice_base["rougeL"]}')


if __name__ == '__main__':
    df = pd.read_csv('/home/wang/Downloads/auto_tuning4rag/mcts_reranker_results_test.csv')
    df_jina = pd.read_csv('/home/wang/Downloads/auto_tuning4rag/mcts_reranker_jina_test.csv')
    df = pd.concat([df, df_jina], ignore_index=True)
    grouped = df.groupby('Question')
    specific_question = "How were the Olympic games broadcasted?"
    question_records = grouped.get_group(specific_question)
    best_records = df.loc[df.groupby('Question')['ROUGE-L'].idxmax()]

    # 计算平均 ROUGE 分数
    avg_rouge_1 = best_records['ROUGE-1'].mean()
    avg_rouge_2 = best_records['ROUGE-2'].mean()
    avg_rouge_l = best_records['ROUGE-L'].mean()

    print(f"Average ROUGE-1: {avg_rouge_1}")
    print(f"Average ROUGE-2: {avg_rouge_2}")
    print(f"Average ROUGE-L: {avg_rouge_l}")

    filepath = 'new_1024_20_20.csv'
    model_path = 'models/long_t5_epoch_1_combined.pth'
    logger = setup_logger('evaluation_logger', 'evaluation.log')
    # evaluate_wiki_baseline(logger=logger)
    # evaluate_wiki(logger)
    # evaluate_llama()
    # save_combinations_only(logger=logger, start_index=1000, end_index=2000)
    # save_trees_raptor_id()
    raptor_evaluate()
    # load_tree_and_evaluate_subset(logger=logger)
