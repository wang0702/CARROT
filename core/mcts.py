import math
import tiktoken
import torch
from FlagEmbedding import FlagReranker
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from core.node import Node


class MCTS:
    def __init__(self, nodes_query, model, tokenizer=None, C=1.41, lambda_=1, budget=8192):
        self.nodes_query = nodes_query
        self.model = model
        self.tokenizer = tokenizer
        self.total_trials = 0
        self.C = C  # Exploration parameter
        self.lambda_ = lambda_  # Cost penalization parameter
        self.budget = budget  # Budget in terms of tokens

    def select_best_node(self, nodes, parent_visits):
        valid_nodes = self._filter_by_budget(nodes)
        if not valid_nodes:
            valid_nodes = nodes
        return max(valid_nodes, key=lambda node: node.ucb1(parent_visits, self.C, self.lambda_, self.budget))

    def _filter_by_budget(self, nodes):
        if not self.tokenizer:
            return nodes
        enc = tiktoken.get_encoding("cl100k_base")
        return [node for node in nodes if len(enc.encode(node.concat_text)) <= self.budget]

    def _is_fully_expanded(self, node):
        available_chunks = [n for n in self.nodes_query if n not in node.state]
        return len(node.children) >= len(available_chunks)

    def expand(self, parent_node):
        expanded_nodes = []
        available_nodes = [node for node in self.nodes_query if node not in parent_node.state]

        for next_node in available_nodes:
            new_state = parent_node.state + [next_node]
            if any(child.state == new_state for child in parent_node.children):
                continue

            new_concat_text = parent_node.concat_text + " " + next_node.node.text
            new_node = Node(state=new_state, layer=parent_node.layer + 1, concat_text=new_concat_text,
                            parent=parent_node)
            parent_node.add_child(new_node)
            expanded_nodes.append(new_node)

        return expanded_nodes

    def simulate(self, query, nodes):
        all_states = [node.state for node in nodes]
        all_documents = [[node.node.text for node in state] for state in all_states]
        sentence_pairs = [[query, " ".join(docs)] for docs in all_documents]

        if self.tokenizer:
            with torch.no_grad():
                inputs = self.tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors='pt',
                                        max_length=8192)
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        else:
            scores = self.model.compute_score(sentence_pairs)

        if isinstance(scores, list):
            rewards = scores
        elif isinstance(scores, (int, float)):
            rewards = [scores]
        elif hasattr(scores, 'dim') and scores.dim() == 0:
            rewards = [scores.item()]
        else:
            rewards = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        return rewards

    def backpropagate(self, nodes, rewards):
        for node, reward in zip(nodes, rewards):
            self._backpropagate_single(node, reward)

    def _backpropagate_single(self, node, reward):
        current = node
        is_first_node = True

        while current is not None:
            current.visits += 1
            current.total_reward += reward

            if is_first_node and current.reranker_score is None:
                current.reranker_score = reward

            current = current.parent
            is_first_node = False

    def search(self, query, max_iterations=100):
        root_node = Node(state=[], layer=0, concat_text="")

        for iteration in range(max_iterations):
            self.total_trials += 1
            node = root_node

            while self._is_fully_expanded(node) and node.children:
                node = self.select_best_node(node.children, node.visits)

            if node.layer < len(self.nodes_query):
                expanded_nodes = self.expand(node)

                if expanded_nodes:
                    rewards = self.simulate(query, expanded_nodes)
                    self.backpropagate(expanded_nodes, rewards)

        best_node = self.find_best_node_recursive(root_node)
        return best_node

    def find_best_node_recursive(self, node):
        """
        Find the node with highest reranker score across all layers.
        Uses reranker_score (not avg_reward) to avoid penalizing nodes with poor children.
        Enforces budget constraint before selection.
        """
        all_nodes = []

        def collect(n):
            if n.visits > 0 and len(n.state) > 0 and n.reranker_score is not None:
                all_nodes.append(n)
            for child in n.children:
                collect(child)

        collect(node)

        if self.tokenizer and all_nodes:
            enc = tiktoken.get_encoding("cl100k_base")
            valid_nodes = [n for n in all_nodes
                          if len(enc.encode(n.concat_text)) <= self.budget]
            all_nodes = valid_nodes if valid_nodes else all_nodes

        return max(all_nodes, key=lambda n: (
            n.reranker_score,
            n.visits,
            n.layer
        )) if all_nodes else node
