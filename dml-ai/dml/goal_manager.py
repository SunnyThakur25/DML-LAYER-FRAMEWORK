import random
from collections import defaultdict

class GoalManager:
    def __init__(self):
        self.task_rewards = defaultdict(list)
        self.task_types = ["story", "math", "code", "poem", "essay"]

    def select_task(self):
        scores = {
            t: sum(self.task_rewards[t]) / len(self.task_rewards[t])
            if self.task_rewards[t] else 0.5
            for t in self.task_types
        }
        total = sum(scores.values())
        probs = [scores[t] / total for t in self.task_types]
        return random.choices(self.task_types, weights=probs, k=1)[0]

    def register_reward(self, task_type, reward):
        self.task_rewards[task_type].append(reward)