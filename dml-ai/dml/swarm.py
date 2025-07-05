import torch
import hmac
import hashlib
from collections import defaultdict

class SecureSwarmNode:
    def __init__(self, node_id, secret_key):
        self.id = node_id
        self.dml = SafeSelfModifyingDML()
        self.peers = []
        self.memory_clock = 0
        self.key = secret_key.encode()

    def _sign(self, data):
        return hmac.new(self.key, data, hashlib.sha256).hexdigest()

    def sync_memory(self):
        for peer in self.peers:
            peer_mem = peer.get_memory()
            if self.verify_signature(peer_mem['data'], peer_mem['sig']):
                self.dml.memory.data = 0.6 * self.dml.memory.data + 0.4 * peer_mem['data']
                self.memory_clock = max(self.memory_clock, peer_mem['clock']) + 1

    def get_memory(self):
        return {
            'data': self.dml.memory.data.clone(),
            'sig': self._sign(self.dml.memory.data.numpy().tobytes()),
            'clock': self.memory_clock
        }

    def verify_signature(self, data, signature):
        correct_sig = hmac.new(self.key, data.numpy().tobytes(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(correct_sig, signature)