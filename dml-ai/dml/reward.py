import torch
from collections import defaultdict

class RewardEvaluator:
    def __init__(self):
        self.task_stats = defaultdict(list)

    def evaluate_output(self, task_type, output_text):
        score = 0.0
        if task_type == "story":
            score = 1.0 if "once upon a time" in output_text.lower() else 0.7
        elif task_type == "math":
            score = 1.0 if "=" in output_text or output_text.strip().isdigit() else 0.6
        elif task_type == "code":
            score = 1.0 if "def" in output_text or "import" in output_text else 0.5
        elif task_type == "poem":
            score = 0.8 if output_text.count("\n") > 3 else 0.5
        else:
            score = 0.6
        self.task_stats[task_type].append(score)
        return score

    def update_memory(self, dml_model, reward_score):
        with torch.no_grad():
            norm_reward = torch.tensor(reward_score).clamp(0.0, 1.0)
            dml_model.memory.data *= 0.95
            dml_model.memory.data += norm_reward * 0.05

    def log_feedback(self, task_type, reward_score):
        print(f"📊 Task: {task_type}, Reward Score: {reward_score:.2f}")