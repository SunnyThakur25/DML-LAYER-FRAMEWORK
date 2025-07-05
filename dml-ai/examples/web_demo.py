import gradio as gr
from dml.goal_manager import GoalManager
from dml.reward import RewardEvaluator
from dml.selfmodify import SafeSelfModifyingDML

goal_manager = GoalManager()
evaluator = RewardEvaluator()
dml = SafeSelfModifyingDML()

task_templates = {
    "story": "Once upon a time, AI evolved...",
    "math": "2 + 3 = 5",
    "code": "def hello(): print('Hello World')",
    "poem": "Dreams flow deep\nCode runs in sleep\n...",
    "essay": "The future of AI is bright and uncertain."
}

def run_cycle():
    task = goal_manager.select_task()
    output = task_templates[task]
    reward = evaluator.evaluate_output(task, output)
    evaluator.update_memory(dml, reward)
    goal_manager.register_reward(task, reward)
    evaluator.log_feedback(task, reward)
    return f"🧠 Selected Task: {task}\n📊 Reward Score: {reward:.2f}", output

with gr.Blocks(title="🧠 DML-AI Demo") as demo:
    gr.Markdown("## 🧠 DML-AI: Dynamic Meta-Learning Layer Demo")
    btn = gr.Button("🔄 Run AGI Cycle")
    status = gr.Textbox(label="Status")
    output = gr.Textbox(label="Generated Output")
    btn.click(fn=run_cycle, inputs=[], outputs=[status, output])

demo.launch()