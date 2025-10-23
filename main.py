import sys
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.readers import StringIterableReader

from agents.model_loader import ModelLoader
from agents.configuration_predictor import ConfigurationPredictor
from core.mcts import MCTS


def main(use_agent=False):
    print("Initializing models...")
    model_loader = ModelLoader(llm_str='openai/gpt-4o')
    llm, jina_model_v1, jina_model_v2, flag_reranker_v2, flag_reranker_large, \
        flag_reranker_base, gte_model, gte_tokenizer = model_loader.get_all_models()

    query = "How does MCTS optimize chunk selection in cost-constrained retrieval?"

    documents = StringIterableReader().load_data(texts=[
        "Chunk Combination Retrieval achieves a good trade-off between efficiency and accuracy in searching for the optimal chunk combination order problem. We model the problem as searching the optimal node within a policy tree and propose an MCTS-based algorithm to address this node search problem.",
        "The Policy Tree represents all potential orders of chunk combinations sourced from the vector database. The root node symbolizes the initial state without any chunk, with each subsequent node depicting a selected chunk. A child node emerges from its parent by selecting the next available chunk and incorporating it into the sequence established by the ancestor node.",
        "Node Utility comprises two components: the benefit derived from selecting the chunk combination and the cost associated with using it as a prompt in LLMs. The benefit is measured by semantic similarity between chunk combination and input query using a lightweight reranker. Cost is measured as the proportion of tokens required relative to the total available token budget.",
        "The Upper Confidence Bound (UCB) design provides a principled approach to node selection while preventing premature convergence to suboptimal solutions. We maintain two key statistics for each node: cumulative reward tracking aggregate reward across multiple visits, and visit count tracking exploration frequency. These statistics enable accurate estimation of expected node values through iterative refinement.",
        "The node utility is defined as U(v_i) = V(v_i)/N(v_i) + c*sqrt(ln N/N(v_i)) - lambda*cost(v_i)/B, where V(v_i) is cumulative reward, N(v_i) is visit count, N is total visits, cost(v_i) is token cost, and c, lambda are tuning parameters. This formulation extends standard UCB to cost-constrained retrieval settings.",
        "The task of selecting an optimal chunk combination order is reformulated as optimal node selection within the policy tree. Given a budget constraint B, the objective is to identify the node v_i that maximizes utility U(v_i) while ensuring total cost does not exceed B. This enables selection of chunks that maximize utility within the given budget.",
        "Enumerating all nodes in the policy tree will locate the optimal node but results in high computational costs. A greedy strategy navigating the tree from the root iteratively selects the child with highest benefit, but this leads to suboptimal results. For example, benefit of chi_1 may be slightly higher than chi_2, but benefit of chi_2 + chi_3 could greatly exceed chi_1 + chi_3.",
        "The MCTS-based strategy explores the space of potential chunk orders iteratively, optimizing a given query within specified budget constraints. We begin by initializing the root node of the policy tree using the input query. While computational budget is not exhausted, we iteratively perform Node Selection and Utility Update, then traverse the entire tree to identify the node with highest average reward within budget constraint.",
        "Node Selection recursively chooses the node with highest utility to expand. Upon the selected node, we build a set of child nodes representing different subsequent chunk combinations and evaluate both benefit and cost metrics. The Selection phase traverses the tree by recursively selecting nodes with maximum utility value. Expansion generates all potential child nodes representing new possible chunk combination orders within budget.",
        "For newly expanded child nodes, we initialize statistics required for future utility computations by computing cumulative reward and cost values. A lightweight reranker model efficiently evaluates reward for each child node. The reranker processes multiple chunk combinations in parallel yielding reward scores. Cumulative reward V is initialized with R and visit number N is set as 1.",
        "Utility Update identifies the child node with highest reranker reward among all newly expanded children. We then propagate this maximum reward upward along the path from best child to root, updating cumulative reward and visit count for each ancestor node. Only the path containing highest-reward child receives updated statistics, allowing UCB to balance exploration with exploitation.",
        "After completing all iterations, we perform global selection across all explored nodes to identify optimal chunk combination. Crucially, we enforce budget constraint by filtering nodes whose token cost exceeds B before selecting the node with highest average reward. Time complexity is O(I*D) where I is iterations and D is maximum depth. Space complexity is O(I*k) where k is expanded child nodes."
    ])
    splitter = SentenceSplitter(chunk_size=256)
    nodes = splitter.get_nodes_from_documents(documents)

    vector_index = VectorStoreIndex(nodes)
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=60)
    query_engine_vector = RetrieverQueryEngine(retriever=retriever)

    retrieved_nodes = query_engine_vector.retrieve(query)

    if use_agent:
        print("Using configuration agent...")
        configuration_predictor = ConfigurationPredictor()
        predicted_reranker, predicted_params = configuration_predictor.predict(query=query, retrieved_nodes=retrieved_nodes)

        C = float(predicted_params[0])
        max_iterations = int(predicted_params[1])
        lambda_ = float(predicted_params[2])

        if predicted_reranker == 'jina-reranker-v1-turbo-en':
            mcts = MCTS(nodes_query=retrieved_nodes, model=jina_model_v1, C=C, lambda_=lambda_, budget=8192)
        elif predicted_reranker == 'jina-reranker-v2-base-multilingual':
            mcts = MCTS(nodes_query=retrieved_nodes, model=jina_model_v2, C=C, lambda_=lambda_, budget=8192)
        elif predicted_reranker == 'bge-reranker-v2-m3':
            mcts = MCTS(nodes_query=retrieved_nodes, model=flag_reranker_v2, C=C, lambda_=lambda_, budget=8192)
        elif predicted_reranker == 'bge-reranker-large':
            mcts = MCTS(nodes_query=retrieved_nodes, model=flag_reranker_large, C=C, lambda_=lambda_, budget=8192)
        elif predicted_reranker == 'bge-reranker-base':
            mcts = MCTS(nodes_query=retrieved_nodes, model=flag_reranker_base, C=C, lambda_=lambda_, budget=8192)
        elif predicted_reranker == 'gte-multilingual-reranker-base':
            mcts = MCTS(nodes_query=retrieved_nodes, model=gte_model, tokenizer=gte_tokenizer, C=C, lambda_=lambda_,
                        budget=8192)

        best_node = mcts.search(query, max_iterations=max_iterations)
    else:
        print("Using fixed configuration...")
        mcts = MCTS(nodes_query=retrieved_nodes, model=jina_model_v2, tokenizer=None, C=2.4, lambda_=0.1,
                    budget=8192)
        best_node = mcts.search(query, max_iterations=10)

    best_context = best_node.concat_text
    answer = llm.complete(
        f"Provide precise answer to the question: {query} Based on given Context: {best_context}").text

    print("\nAnswer:")
    print(answer)


if __name__ == '__main__':
    use_agent = len(sys.argv) > 1 and sys.argv[1] == '--agent'
    main(use_agent=use_agent)
