import math


class Node:
    def __init__(self, state, layer, concat_text, parent=None):
        self.state = state
        self.layer = layer
        self.concat_text = concat_text
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.reranker_score = None

    def add_child(self, child_node):
        self.children.append(child_node)

    def token_cost(self):
        return len(self.concat_text.split())

    def ucb1(self, parent_visits, C=2, lambda_=1, budget=1024):
        if self.visits == 0:
            return float('inf')
        average_reward = self.total_reward / self.visits
        exploration_term = C * math.sqrt(math.log(parent_visits) / self.visits)
        cost_term = lambda_ * (self.token_cost() / budget)
        return average_reward + exploration_term - cost_term
