#!/usr/bin/env python3

"""
Training Results Analyzer

PURPOSE:
This script analyzes and compares the performance of single GPU vs multi-GPU training
runs. It provides comprehensive insights into training efficiency, speed improvements,
loss convergence, and resource utilization.

WHAT IT DOES:
1. Scans for training session directories containing single_gpu and multi_gpu outputs
2. Extracts performance metrics like training duration, loss, steps per second
3. Compares single GPU vs multi-GPU performance with speedup calculations
4. Shows detailed file listings and model outputs for each training run
5. Provides actionable insights on GPU scaling efficiency

KEY METRICS ANALYZED:
- Training Duration: Total time taken for training completion
- Steps Per Second: Training throughput (higher is better)
- Final Loss: Model convergence quality (lower is better)
- GPU Utilization: Number of GPUs used and scaling efficiency
- Memory Usage: Model and checkpoint file sizes
- Speedup Factor: Performance improvement from using multiple GPUs

OUTPUT STRUCTURE:
The script expects training outputs in this structure:
sessions/
  session_name/
    single_gpu/
      results.json, trainer_state.json, model files
    multi_gpu/
      results.json, trainer_state.json, model files
"""

import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path

def find_training_sessions():
    """Find all training session directories"""
    sessions_dir = Path("./sessions")
    sessions = {}
    
    if not sessions_dir.exists():
        # Fallback to old structure
        current_dir = Path(".")
        patterns = ["*gpu_output", "*_output", "output*", "results*"]
        
        for pattern in patterns:
            for path in current_dir.glob(pattern):
                if path.is_dir():
                    sessions[path.name] = {"path": path, "type": "legacy"}
        return sessions
    
    # New session-based structure
    for session_path in sessions_dir.iterdir():
        if session_path.is_dir():
            single_gpu_path = session_path / "single_gpu"
            multi_gpu_path = session_path / "multi_gpu"
            
            sessions[session_path.name] = {
                "path": session_path,
                "type": "session",
                "single_gpu": single_gpu_path if single_gpu_path.exists() else None,
                "multi_gpu": multi_gpu_path if multi_gpu_path.exists() else None
            }
    
    return sessions

def analyze_results_json(results_path):
    """Analyze results.json file"""
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"❌ Error reading {results_path}: {e}")
        return None

def analyze_training_logs(output_dir):
    """Analyze training logs and checkpoints"""
    analysis = {
        "directory": str(output_dir),
        "files": [],
        "checkpoints": [],
        "logs": [],
        "model_files": []
    }
    
    # List all files
    for item in output_dir.rglob("*"):
        if item.is_file():
            analysis["files"].append({
                "name": item.name,
                "path": str(item.relative_to(output_dir)),
                "size_mb": item.stat().st_size / (1024*1024),
                "modified": datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Categorize files
            if "checkpoint" in item.name.lower():
                analysis["checkpoints"].append(str(item.relative_to(output_dir)))
            elif item.suffix in [".log", ".txt"]:
                analysis["logs"].append(str(item.relative_to(output_dir)))
            elif item.suffix in [".bin", ".pt", ".pth", ".safetensors"]:
                analysis["model_files"].append(str(item.relative_to(output_dir)))
    
    return analysis

def display_comparison(single_gpu_results, multi_gpu_results):
    """Display side-by-side comparison"""
    print("="*80)
    print("📊 TRAINING RESULTS COMPARISON")
    print("="*80)
    
    if single_gpu_results and multi_gpu_results:
        print(f"{'Metric':<25} {'Single GPU':<20} {'Multi GPU':<20} {'Improvement':<15}")
        print("-"*80)
        
        # Duration comparison
        single_duration = single_gpu_results.get('duration_seconds', 0)
        multi_duration = multi_gpu_results.get('duration_seconds', 0)
        if single_duration > 0 and multi_duration > 0:
            speedup = single_duration / multi_duration
            print(f"{'Duration (seconds)':<25} {single_duration:<20.1f} {multi_duration:<20.1f} {speedup:.2f}x")
        
        # Steps per second
        single_sps = single_gpu_results.get('steps_per_second', 0)
        multi_sps = multi_gpu_results.get('steps_per_second', 0)
        if single_sps > 0 and multi_sps > 0:
            sps_improvement = multi_sps / single_sps
            print(f"{'Steps/second':<25} {single_sps:<20.2f} {multi_sps:<20.2f} {sps_improvement:.2f}x")
        
        # Final loss
        single_loss = single_gpu_results.get('final_loss', 0)
        multi_loss = multi_gpu_results.get('final_loss', 0)
        if single_loss > 0 and multi_loss > 0:
            loss_diff = (single_loss - multi_loss) / single_loss * 100
            print(f"{'Final Loss':<25} {single_loss:<20.4f} {multi_loss:<20.4f} {loss_diff:+.1f}%")
        
        # Total steps
        single_steps = single_gpu_results.get('total_steps', 0)
        multi_steps = multi_gpu_results.get('total_steps', 0)
        print(f"{'Total Steps':<25} {single_steps:<20} {multi_steps:<20}")
        
        # GPU count
        single_gpus = 1
        multi_gpus = multi_gpu_results.get('num_gpus', 1)
        print(f"{'GPU Count':<25} {single_gpus:<20} {multi_gpus:<20}")
        
        # Model info
        single_model = single_gpu_results.get('model_name', 'Unknown')
        multi_model = multi_gpu_results.get('model_name', 'Unknown')
        print(f"{'Model':<25} {single_model[:19]:<20} {multi_model[:19]:<20}")
        
        # Optimizations
        single_opt = single_gpu_results.get('optimizations', 'Unknown')
        multi_opt = multi_gpu_results.get('optimizations', 'Unknown')
        print(f"{'Optimizations':<25} {single_opt[:19]:<20} {multi_opt[:19]:<20}")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument("--session", type=str, help="Specific session to analyze")
    args = parser.parse_args()
    
    print("TRAINING RESULTS ANALYZER")
    print("="*50)
    print("This tool analyzes your multi-GPU training performance and provides")
    print("detailed comparisons between single GPU and multi-GPU training runs.")
    print("")
    
    # Find sessions
    sessions = find_training_sessions()
    
    if not sessions:
        print("No training output directories found!")
        print("Make sure you're running this from the training directory")
        print("Expected structure: sessions/session_name/single_gpu/ and sessions/session_name/multi_gpu/")
        return
    
    if args.session:
        if args.session not in sessions:
            print(f"Session '{args.session}' not found!")
            print(f"Available sessions: {list(sessions.keys())}")
            return
        sessions_to_analyze = {args.session: sessions[args.session]}
    else:
        sessions_to_analyze = sessions
    
    print(f"Found {len(sessions_to_analyze)} training session(s):")
    for name, session_info in sessions_to_analyze.items():
        if session_info["type"] == "session":
            single_exists = "YES" if session_info["single_gpu"] else "NO"
            multi_exists = "YES" if session_info["multi_gpu"] else "NO"
            print(f"  - {name}: Single GPU [{single_exists}], Multi GPU [{multi_exists}]")
        else:
            print(f"  - {name}: Legacy format")
    
    print("\n" + "="*50)
    
    # Analyze each session
    for session_name, session_info in sessions_to_analyze.items():
        print(f"\nAnalyzing Session: {session_name}")
        print("-"*40)
        
        single_result = None
        multi_result = None
        
        if session_info["type"] == "session":
            # New session structure
            if session_info["single_gpu"]:
                results_json = session_info["single_gpu"] / "results.json"
                if results_json.exists():
                    single_result = analyze_results_json(results_json)
                    print("Single GPU Training Results:")
                    if single_result:
                        for key, value in single_result.items():
                            print(f"  {key}: {value}")
                    
                    # File analysis
                    single_analysis = analyze_training_logs(session_info["single_gpu"])
                    file_count = len(single_analysis["files"])
                    total_size = sum(f["size_mb"] for f in single_analysis["files"])
                    print(f"  Files: {file_count} (Total: {total_size:.1f} MB)")
            
            if session_info["multi_gpu"]:
                results_json = session_info["multi_gpu"] / "results.json"
                if results_json.exists():
                    multi_result = analyze_results_json(results_json)
                    print("\nMulti GPU Training Results:")
                    if multi_result:
                        for key, value in multi_result.items():
                            print(f"  {key}: {value}")
                    
                    # File analysis
                    multi_analysis = analyze_training_logs(session_info["multi_gpu"])
                    file_count = len(multi_analysis["files"])
                    total_size = sum(f["size_mb"] for f in multi_analysis["files"])
                    print(f"  Files: {file_count} (Total: {total_size:.1f} MB)")
        
        # Performance comparison
        if single_result and multi_result:
            print("\n")
            display_comparison(single_result, multi_result)
            
            # Performance insights
            print("\nPERFORMANCE INSIGHTS:")
            print("-"*30)
            
            duration_improvement = single_result.get('duration_seconds', 0) / multi_result.get('duration_seconds', 1)
            if duration_improvement > 1.5:
                print(f"EXCELLENT: Multi-GPU achieved {duration_improvement:.1f}x speedup")
            elif duration_improvement > 1.1:
                print(f"GOOD: Multi-GPU achieved {duration_improvement:.1f}x speedup")
            else:
                print(f"POOR: Multi-GPU scaling inefficient ({duration_improvement:.1f}x)")
            
            loss_improvement = (single_result.get('final_loss', 0) - multi_result.get('final_loss', 0)) / single_result.get('final_loss', 1) * 100
            if abs(loss_improvement) < 5:
                print("Loss convergence is similar between single and multi-GPU")
            elif loss_improvement > 0:
                print(f"Multi-GPU achieved {loss_improvement:.1f}% better loss convergence")
            else:
                print(f"Single-GPU achieved {abs(loss_improvement):.1f}% better loss convergence")
        elif single_result:
            print("Only single GPU results available")
        elif multi_result:
            print("Only multi GPU results available")
        else:
            print("No training results found for this session")
    
    # Generate summary
    print(f"\nSUMMARY")
    print("="*50)
    print(f"Total training sessions analyzed: {len(sessions_to_analyze)}")
    
    # Count sessions with results
    sessions_with_results = 0
    for session_name, session_info in sessions_to_analyze.items():
        if session_info["type"] == "session":
            if session_info.get("single_gpu") or session_info.get("multi_gpu"):
                sessions_with_results += 1
        else:
            sessions_with_results += 1
    
    print(f"Sessions with training results: {sessions_with_results}")
    
    print("\nTo view detailed logs and files:")
    for session_name, session_info in sessions_to_analyze.items():
        if session_info["type"] == "session":
            print(f"Session {session_name}:")
            if session_info.get("single_gpu"):
                print(f"  Single GPU: sessions/{session_name}/single_gpu/")
            if session_info.get("multi_gpu"):
                print(f"  Multi GPU: sessions/{session_name}/multi_gpu/")
    
    print("\nReal-time log monitoring commands:")
    for session_name in sessions_to_analyze.keys():
        print(f"  tail -f sessions/{session_name}/single_gpu/trainer_state.json")
        print(f"  tail -f sessions/{session_name}/multi_gpu/trainer_state.json")

if __name__ == "__main__":
    main()