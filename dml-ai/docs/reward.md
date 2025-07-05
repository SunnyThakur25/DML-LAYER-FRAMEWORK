


Reinforces creativity and task success through heuristic scoring.

## Features

- Task-specific scoring (story/math/code/poem/essay)
- Memory reinforcement
- Feedback logging

## Example


class RewardEvaluator:
    def evaluate_output(self, task_type, output_text):
        ...
    def update_memory(self, dml_model, reward_score):
        ...