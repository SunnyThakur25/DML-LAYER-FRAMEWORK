from dml.goal_manager import GoalManager
from dml.reward import RewardEvaluator
from dml.selfmodify import SafeSelfModifyingDML

model = SafeSelfModifyingDML()
goal_manager = GoalManager()
evaluator = RewardEvaluator()

task_templates = {
    "story": "Once upon a time, AI evolved...",
    "math": "2 + 3 = 5",
    "code": "def hello(): print('Hello World')",
    "poem": "Dreams flow deep\nCode runs in sleep\n...",
    "essay": "The future of AI is bright and uncertain."
}

print("🔁 Starting AGI Loop...")
for i in range(5):
    task = goal_manager.select_task()
    output = task_templates[task]
    reward = evaluator.evaluate_output(task, output)
    evaluator.update_memory(model, reward)
    goal_manager.register_reward(task, reward)
    evaluator.log_feedback(task, reward)