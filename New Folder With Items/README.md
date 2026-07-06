Hybrid Chain Prototype

This prototype demonstrates a minimal blockchain integrating:
- PoW (low difficulty)
- VRF (simple ECDSA-based proof-of-concept)
- Registry Repository (JSON) with Merkle root and proofs
- Small NP-hard problem family (TSP instances) and brute-force solver for n<=7
- Simple proof module (commitment + reveal) for verification in prototype

Run:
- Create & activate a Python venv
- pip install -r requirements.txt
- python main.py
