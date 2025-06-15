#!/usr/bin/env python3

"""
Training Log Viewer - Real-time and Historical Log Analysis
"""

import os
import json
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def watch_logs(log_path, follow=False):
    """Watch training logs in real-time"""
    if not Path(log_path).exists():
        print(f"❌ Log file not found: {log_path}")
        return
    
    print(f"👀 Watching: {log_path}")
    print("="*50)
    
    if follow:
        # Follow mode (like tail -f)
        with open(log_path, 'r') as f:
            # Go to end of file
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    print(line.strip())
                else:
                    time.sleep(0.1)
    else:
        # Read entire file
        with open(log_path, 'r') as f:
            print(f.read())

def parse_trainer_state(state_file):
    """Parse trainer_state.json for training metrics"""
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        metrics = {
            'steps': [],
            'loss': [],
            'learning_rate': [],
            'epoch': []
        }
        
        # Extract log history
        if 'log_history' in state:
            for entry in state['log_history']:
                if 'train_loss' in entry:
                    metrics['steps'].append(entry.get('step', 0))
                    metrics['loss'].append(entry['train_loss'])
                    metrics['learning_rate'].append(entry.get('learning_rate', 0))
                    metrics['epoch'].append(entry.get('epoch', 0))
        
        return state, metrics
    except Exception as e:
        print(f"❌ Error parsing trainer state: {e}")
        return None, None

def plot_training_metrics(single_metrics, multi_metrics, save_path="training_comparison.png"):
    """Create training metrics visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Comparison: Single GPU vs Multi-GPU', fontsize=16)
    
    # Loss comparison
    if single_metrics and single_metrics['loss']:
        axes[0,0].plot(single_metrics['steps'], single_metrics['loss'], 
                      label='Single GPU', color='blue', linewidth=2)
    if multi_metrics and multi_metrics['loss']:
        axes[0,0].plot(multi_metrics['steps'], multi_metrics['loss'], 
                      label='Multi GPU', color='red', linewidth=2)
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Steps')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Learning rate
    if single_metrics and single_metrics['learning_rate']:
        axes[0,1].plot(single_metrics['steps'], single_metrics['learning_rate'], 
                      label='Single GPU', color='blue', linewidth=2)
    if multi_metrics and multi_metrics['learning_rate']:
        axes[0,1].plot(multi_metrics['steps'], multi_metrics['learning_rate'], 
                      label='Multi GPU', color='red', linewidth=2)
    axes[0,1].set_title('Learning Rate Schedule')
    axes[0,1].set_xlabel('Steps')
    axes[0,1].set_ylabel('Learning Rate')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Steps per epoch
    if single_metrics and single_metrics['epoch']:
        axes[1,0].plot(single_metrics['steps'], single_metrics['epoch'], 
                      label='Single GPU', color='blue', linewidth=2)
    if multi_metrics and multi_metrics['epoch']:
        axes[1,0].plot(multi_metrics['steps'], multi_metrics['epoch'], 
                      label='Multi GPU', color='red', linewidth=2)
    axes[1,0].set_title('Training Progress (Epochs)')
    axes[1,0].set_xlabel('Steps')
    axes[1,0].set_ylabel('Epoch')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Loss improvement rate
    if single_metrics and len(single_metrics['loss']) > 1:
        single_improvement = np.diff(single_metrics['loss'])
        axes[1,1].plot(single_metrics['steps'][1:], single_improvement, 
                      label='Single GPU', color='blue', alpha=0.7)
    if multi_metrics and len(multi_metrics['loss']) > 1:
        multi_improvement = np.diff(multi_metrics['loss'])
        axes[1,1].plot(multi_metrics['steps'][1:], multi_improvement, 
                      label='Multi GPU', color='red', alpha=0.7)
    axes[1,1].set_title('Loss Change Rate')
    axes[1,1].set_xlabel('Steps')
    axes[1,1].set_ylabel('Loss Delta')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {save_path}")
    return save_path

def display_training_summary(state, metrics, training_type):
    """Display comprehensive training summary"""
    print(f"\n📋 {training_type.upper()} TRAINING SUMMARY")
    print("="*50)
    
    if state:
        print(f"✅ Total Steps: {state.get('global_step', 'Unknown')}")
        print(f"✅ Current Epoch: {state.get('epoch', 'Unknown')}")
        print(f"✅ Best Metric: {state.get('best_metric', 'Unknown')}")
        print(f"✅ Best Model Checkpoint: {state.get('best_model_checkpoint', 'None')}")
        
        if 'train_batch_size' in state:
            print(f"✅ Training Batch Size: {state['train_batch_size']}")
        
        # Timing information
        if 'log_history' in state and state['log_history']:
            first_log = state['log_history'][0]
            last_log = state['log_history'][-1]
            
            if 'train_runtime' in last_log:
                runtime = last_log['train_runtime']
                print(f"✅ Training Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
            
            if 'train_samples_per_second' in last_log:
                sps = last_log['train_samples_per_second']
                print(f"✅ Samples/Second: {sps:.2f}")
            
            if 'train_steps_per_second' in last_log:
                steps_ps = last_log['train_steps_per_second']
                print(f"✅ Steps/Second: {steps_ps:.2f}")
    
    if metrics and metrics['loss']:
        initial_loss = metrics['loss'][0]
        final_loss = metrics['loss'][-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"✅ Initial Loss: {initial_loss:.4f}")
        print(f"✅ Final Loss: {final_loss:.4f}")
        print(f"✅ Loss Reduction: {loss_reduction:.1f}%")
        
        if len(metrics['loss']) > 1:
            min_loss = min(metrics['loss'])
            max_loss = max(metrics['loss'])
            print(f"✅ Loss Range: {min_loss:.4f} - {max_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Training Log Viewer and Analyzer")
    parser.add_argument("--output-dir", type=str, help="Output directory to analyze")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow logs in real-time")
    parser.add_argument("--compare", action="store_true", help="Compare single vs multi GPU results")
    parser.add_argument("--plot", action="store_true", help="Generate training plots")
    parser.add_argument("--watch", type=str, help="Watch specific log file")
    
    args = parser.parse_args()
    
    if args.watch:
        watch_logs(args.watch, args.follow)
        return
    
    if args.compare or args.plot:
        print("🔍 TRAINING LOG ANALYZER")
        print("="*50)
        
        # Look for training outputs
        single_dir = Path("single_gpu_output")
        multi_dir = Path("multi_gpu_output")
        
        single_state, single_metrics = None, None
        multi_state, multi_metrics = None, None
        
        # Analyze single GPU
        if single_dir.exists():
            trainer_state = single_dir / "trainer_state.json"
            if trainer_state.exists():
                single_state, single_metrics = parse_trainer_state(trainer_state)
                display_training_summary(single_state, single_metrics, "Single GPU")
        
        # Analyze multi GPU
        if multi_dir.exists():
            trainer_state = multi_dir / "trainer_state.json"
            if trainer_state.exists():
                multi_state, multi_metrics = parse_trainer_state(trainer_state)
                display_training_summary(multi_state, multi_metrics, "Multi GPU")
        
        # Generate plots if requested
        if args.plot:
            try:
                plot_path = plot_training_metrics(single_metrics, multi_metrics)
                print(f"\n📊 Training visualization created: {plot_path}")
            except ImportError:
                print("❌ matplotlib not available. Install with: pip install matplotlib")
            except Exception as e:
                print(f"❌ Error creating plots: {e}")
    
    elif args.output_dir:
        # Analyze specific directory
        output_path = Path(args.output_dir)
        if not output_path.exists():
            print(f"❌ Directory not found: {args.output_dir}")
            return
        
        trainer_state = output_path / "trainer_state.json"
        if trainer_state.exists():
            state, metrics = parse_trainer_state(trainer_state)
            display_training_summary(state, metrics, args.output_dir)
        else:
            print(f"❌ No trainer_state.json found in {args.output_dir}")
    
    else:
        # Default: show available options
        print("🔍 TRAINING LOG VIEWER")
        print("="*50)
        print("Available commands:")
        print("  --compare              Compare single vs multi GPU training")
        print("  --plot                 Generate training visualization plots")
        print("  --output-dir DIR       Analyze specific output directory")
        print("  --watch FILE --follow  Watch log file in real-time")
        print("")
        print("Examples:")
        print("  python log_viewer.py --compare --plot")
        print("  python log_viewer.py --output-dir single_gpu_output")
        print("  python log_viewer.py --watch single_gpu_output/trainer_state.json -f")

if __name__ == "__main__":
    main()