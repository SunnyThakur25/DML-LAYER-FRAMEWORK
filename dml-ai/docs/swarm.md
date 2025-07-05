




# Swarm Learning

Secure peer-to-peer synchronization mechanism using HMAC signatures.

## Features

- HMAC-signed memory sharing
- Adaptive mixing based on novelty and trust
- Security hardening against malicious peers

## Example


class SecureSwarmNode:
    def __init__(self, node_id, secret_key):
        self.id = node_id
        self.dml = SafeSelfModifyingDML()
        self.peers = []
        self.key = secret_key.encode()