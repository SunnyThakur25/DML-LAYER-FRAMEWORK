
import unittest
import torch
from dml.core import DMLLayer
from dml.memory import MemoryDML
from dml.selfmodify import SafeSelfModifyingDML
from dml.swarm import SecureSwarmNode
from dml.goal_manager import GoalManager
from dml.reward import RewardEvaluator

class TestDMLComponents(unittest.TestCase):

    def test_dml_layer(self):
        layer = DMLLayer(d_model=128)
        x = torch.randn(2, 10, 128)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_memory_dml(self):
        layer = MemoryDML(d_model=128)
        x = torch.randn(2, 10, 128)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_selfmodify_dml(self):
        layer = SafeSelfModifyingDML(d_model=128)
        x = torch.randn(1, 10, 128)
        code = layer.generate_safe_code(x)
        result = layer.apply_code(code)
        self.assertTrue(result)

    def test_swarm_node(self):
        node = SecureSwarmNode(node_id=1, secret_key="testkey")
        other_node = SecureSwarmNode(node_id=2, secret_key="testkey")
        node.peers = [other_node]
        node.sync_memory()
        self.assertTrue(True)  # No error means it passed

    def test_goal_manager(self):
        manager = GoalManager()
        task = manager.select_task()
        self.assertIn(task, ["story", "math", "code", "poem", "essay"])

    def test_reward_evaluator(self):
        evaluator = RewardEvaluator()
        score = evaluator.evaluate_output("story", "Once upon a time...")
        self.assertGreaterEqual(score, 0.7)

if __name__ == "__main__":
    unittest.main()