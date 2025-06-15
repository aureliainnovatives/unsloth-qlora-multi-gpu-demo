#!/usr/bin/env python3

"""
Test script to verify the analysis fix works
"""

import os
import json
from pathlib import Path

# Create test session structure
test_session = Path("sessions/test_session")
test_session.mkdir(parents=True, exist_ok=True)

single_gpu_dir = test_session / "single_gpu"
multi_gpu_dir = test_session / "multi_gpu"

single_gpu_dir.mkdir(exist_ok=True)
multi_gpu_dir.mkdir(exist_ok=True)

# Create test results files
single_results = {
    "mode": "single-gpu",
    "session": "test_session",
    "config": "small",
    "duration_seconds": 100.0,
    "final_loss": 5.5,
    "steps_per_second": 1.0,
    "total_steps": 50,
    "model_name": "test-model",
    "optimizations": "test-opt"
}

multi_results = {
    "mode": "multi-gpu",
    "session": "test_session", 
    "config": "small",
    "duration_seconds": 60.0,
    "final_loss": 4.5,
    "steps_per_second": 1.67,
    "total_steps": 50,
    "num_gpus": 2,
    "model_name": "test-model",
    "optimizations": "test-opt"
}

with open(single_gpu_dir / "results.json", "w") as f:
    json.dump(single_results, f)

with open(multi_gpu_dir / "results.json", "w") as f:
    json.dump(multi_results, f)

print("Test session structure created!")
print("Now run: python analyze_training_results.py --session test_session")
print("\nTo clean up test files:")
print("rm -rf sessions/test_session")