#!/usr/bin/env python3

"""
Training Results Analyzer - Compare Single GPU vs Multi-GPU Performance
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path

def find_training_outputs():
    """Find all training output directories"""
    current_dir = Path(".")
    outputs = {}
    
    # Look for output directories
    patterns = ["*gpu_output", "*_output", "output*", "results*"]
    
    for pattern in patterns:
        for path in current_dir.glob(pattern):
            if path.is_dir():
                outputs[path.name] = path
    
    return outputs

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
    print("🔍 TRAINING RESULTS ANALYZER")
    print("="*50)
    
    # Find output directories
    outputs = find_training_outputs()
    
    if not outputs:
        print("❌ No training output directories found!")
        print("💡 Make sure you're running this from the training directory")
        return
    
    print(f"📁 Found {len(outputs)} output directories:")
    for name, path in outputs.items():
        print(f"  - {name}: {path}")
    
    print("\n" + "="*50)
    
    # Analyze each output directory
    results = {}
    detailed_analysis = {}
    
    for name, path in outputs.items():
        print(f"\n📊 Analyzing: {name}")
        print("-"*30)
        
        # Look for results.json
        results_json = path / "results.json"
        if results_json.exists():
            results[name] = analyze_results_json(results_json)
            if results[name]:
                print(f"✅ Found results.json")
                for key, value in results[name].items():
                    print(f"  {key}: {value}")
        else:
            print(f"❌ No results.json found")
        
        # Detailed file analysis
        detailed_analysis[name] = analyze_training_logs(path)
        file_count = len(detailed_analysis[name]["files"])
        total_size = sum(f["size_mb"] for f in detailed_analysis[name]["files"])
        
        print(f"📁 Files: {file_count} (Total: {total_size:.1f} MB)")
        print(f"📁 Checkpoints: {len(detailed_analysis[name]['checkpoints'])}")
        print(f"📁 Model files: {len(detailed_analysis[name]['model_files'])}")
        print(f"📁 Log files: {len(detailed_analysis[name]['logs'])}")
    
    # Compare single vs multi GPU if both exist
    single_result = results.get('single_gpu_output')
    multi_result = results.get('multi_gpu_output')
    
    if single_result and multi_result:
        print("\n")
        display_comparison(single_result, multi_result)
    
    # Display detailed file listing
    print("\n📋 DETAILED FILE LISTING")
    print("="*50)
    
    for name, analysis in detailed_analysis.items():
        print(f"\n📁 {name.upper()}")
        print("-"*30)
        
        if analysis["files"]:
            print(f"{'File':<30} {'Size (MB)':<10} {'Modified':<20}")
            print("-"*60)
            for file_info in sorted(analysis["files"], key=lambda x: x["size_mb"], reverse=True):
                print(f"{file_info['name'][:29]:<30} {file_info['size_mb']:<10.2f} {file_info['modified']:<20}")
        else:
            print("No files found")
    
    # Generate summary
    print(f"\n🎯 SUMMARY")
    print("="*50)
    print(f"✅ Total output directories analyzed: {len(outputs)}")
    print(f"✅ Results files found: {len([r for r in results.values() if r])}")
    
    if single_result and multi_result:
        print(f"✅ Performance comparison available")
        
        # Key insights
        if 'duration_seconds' in single_result and 'duration_seconds' in multi_result:
            speedup = single_result['duration_seconds'] / multi_result['duration_seconds']
            if speedup > 1.1:
                print(f"🚀 Multi-GPU training was {speedup:.1f}x faster!")
            elif speedup < 0.9:
                print(f"⚠️  Single-GPU training was {1/speedup:.1f}x faster")
            else:
                print(f"⚖️  Similar performance between single and multi-GPU")
    
    print("\n💡 To view logs in real-time during training, use:")
    print("   tail -f single_gpu_output/trainer_state.json")
    print("   tail -f multi_gpu_output/trainer_state.json")

if __name__ == "__main__":
    main()