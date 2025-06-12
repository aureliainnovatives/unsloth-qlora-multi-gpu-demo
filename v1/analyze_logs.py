#!/usr/bin/env python3

import os
import json
import argparse
import glob
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd


class TrainingLogAnalyzer:
    """Analyze training logs and generate insights"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.sessions = []
        self.load_sessions()
    
    def load_sessions(self):
        """Load all training sessions from log directory"""
        
        # Find all session directories
        session_dirs = glob.glob(os.path.join(self.log_dir, "training_*"))
        
        for session_dir in session_dirs:
            summary_file = os.path.join(session_dir, "*_summary.json")
            summary_files = glob.glob(summary_file)
            
            if summary_files:
                try:
                    with open(summary_files[0], 'r') as f:
                        session_data = json.load(f)
                    
                    session_data['log_dir'] = session_dir
                    self.sessions.append(session_data)
                except Exception as e:
                    print(f"Error loading session {session_dir}: {e}")
    
    def compare_single_vs_multi_gpu(self):
        """Compare single GPU vs multi-GPU training sessions"""
        
        single_gpu_sessions = [s for s in self.sessions if s['session']['mode'] == 'single-gpu']
        multi_gpu_sessions = [s for s in self.sessions if s['session']['mode'] == 'multi-gpu']
        
        print("\n" + "="*60)
        print("SINGLE GPU vs MULTI-GPU COMPARISON")
        print("="*60)
        
        if single_gpu_sessions:
            print(f"\n📱 SINGLE GPU SESSIONS ({len(single_gpu_sessions)}):")
            for session in single_gpu_sessions:
                self.print_session_summary(session)
        
        if multi_gpu_sessions:
            print(f"\n🔗 MULTI-GPU SESSIONS ({len(multi_gpu_sessions)}):")
            for session in multi_gpu_sessions:
                self.print_session_summary(session)
        
        # Performance comparison
        if single_gpu_sessions and multi_gpu_sessions:
            self.compare_performance(single_gpu_sessions, multi_gpu_sessions)
    
    def print_session_summary(self, session: Dict):
        """Print a readable session summary"""
        
        session_info = session['session']
        duration = session_info['duration_seconds']
        
        print(f"\n  Session: {session_info['id']}")
        print(f"  Duration: {duration:.1f}s ({duration/60:.1f}min)")
        print(f"  GPUs: {session_info['gpu_count']}")
        
        # Training metrics
        if session['training_metrics']:
            metrics = session['training_metrics']
            steps = len(metrics)
            final_loss = metrics[-1]['loss']
            initial_loss = metrics[0]['loss']
            
            print(f"  Steps: {steps}")
            print(f"  Final Loss: {final_loss:.4f}")
            print(f"  Loss Reduction: {initial_loss - final_loss:.4f}")
            print(f"  Steps/sec: {steps / duration:.2f}")
        
        # GPU utilization
        if session['gpu_snapshots']:
            self.print_gpu_utilization(session['gpu_snapshots'])
    
    def print_gpu_utilization(self, snapshots: List[Dict]):
        """Print GPU utilization summary"""
        
        if not snapshots:
            return
        
        gpu_count = len(snapshots[0]['gpus'])
        
        for gpu_id in range(gpu_count):
            utilizations = [s['gpus'][gpu_id]['utilization'] for s in snapshots]
            memory_usage = [s['gpus'][gpu_id]['memory_percent'] for s in snapshots]
            
            avg_util = sum(utilizations) / len(utilizations)
            max_util = max(utilizations)
            avg_mem = sum(memory_usage) / len(memory_usage)
            
            print(f"  GPU {gpu_id}: {avg_util:.1f}% avg util, {max_util:.1f}% peak, {avg_mem:.1f}% avg memory")
    
    def compare_performance(self, single_gpu_sessions: List[Dict], multi_gpu_sessions: List[Dict]):
        """Compare performance between single and multi-GPU sessions"""
        
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        print("-" * 30)
        
        # Calculate averages
        single_gpu_stats = self.calculate_performance_stats(single_gpu_sessions)
        multi_gpu_stats = self.calculate_performance_stats(multi_gpu_sessions)
        
        print(f"Single GPU Average:")
        print(f"  Steps/sec: {single_gpu_stats['steps_per_sec']:.2f}")
        print(f"  Loss reduction: {single_gpu_stats['loss_reduction']:.4f}")
        print(f"  GPU utilization: {single_gpu_stats['gpu_utilization']:.1f}%")
        
        print(f"\nMulti-GPU Average:")
        print(f"  Steps/sec: {multi_gpu_stats['steps_per_sec']:.2f}")
        print(f"  Loss reduction: {multi_gpu_stats['loss_reduction']:.4f}")
        print(f"  GPU utilization: {multi_gpu_stats['gpu_utilization']:.1f}%")
        
        # Calculate speedup
        if single_gpu_stats['steps_per_sec'] > 0:
            speedup = multi_gpu_stats['steps_per_sec'] / single_gpu_stats['steps_per_sec']
            print(f"\n🚀 Multi-GPU Speedup: {speedup:.2f}x")
        
        # GPU efficiency
        if multi_gpu_stats['gpu_count'] > 1:
            efficiency = speedup / multi_gpu_stats['gpu_count'] * 100
            print(f"📊 GPU Efficiency: {efficiency:.1f}%")
    
    def calculate_performance_stats(self, sessions: List[Dict]) -> Dict:
        """Calculate performance statistics for a list of sessions"""
        
        stats = {
            'steps_per_sec': 0,
            'loss_reduction': 0,
            'gpu_utilization': 0,
            'gpu_count': 0
        }
        
        if not sessions:
            return stats
        
        total_sessions = len(sessions)
        
        for session in sessions:
            # Steps per second
            duration = session['session']['duration_seconds']
            steps = len(session['training_metrics']) if session['training_metrics'] else 0
            if duration > 0:
                stats['steps_per_sec'] += steps / duration
            
            # Loss reduction
            if session['training_metrics']:
                metrics = session['training_metrics']
                initial_loss = metrics[0]['loss']
                final_loss = metrics[-1]['loss']
                stats['loss_reduction'] += initial_loss - final_loss
            
            # GPU utilization
            if session['gpu_snapshots']:
                snapshots = session['gpu_snapshots']
                gpu_count = len(snapshots[0]['gpus'])
                total_util = 0
                
                for gpu_id in range(gpu_count):
                    utilizations = [s['gpus'][gpu_id]['utilization'] for s in snapshots]
                    avg_util = sum(utilizations) / len(utilizations)
                    total_util += avg_util
                
                stats['gpu_utilization'] += total_util / gpu_count
                stats['gpu_count'] = gpu_count
        
        # Calculate averages
        for key in ['steps_per_sec', 'loss_reduction', 'gpu_utilization']:
            stats[key] /= total_sessions
        
        return stats
    
    def analyze_multi_gpu_communication(self):
        """Analyze multi-GPU communication patterns"""
        
        multi_gpu_sessions = [s for s in self.sessions if s['session']['mode'] == 'multi-gpu']
        
        if not multi_gpu_sessions:
            print("No multi-GPU sessions found for communication analysis.")
            return
        
        print("\n" + "="*60)
        print("MULTI-GPU COMMUNICATION ANALYSIS")
        print("="*60)
        
        for session in multi_gpu_sessions:
            print(f"\n📡 Session: {session['session']['id']}")
            print(f"  GPUs: {session['session']['gpu_count']}")
            
            # Analyze communication logs
            if session.get('communication_logs'):
                comm_logs = session['communication_logs']
                
                # Group by operation type
                operations = {}
                for log in comm_logs:
                    op_type = log['operation']
                    if op_type not in operations:
                        operations[op_type] = []
                    operations[op_type].append(log)
                
                print(f"  Communication Operations:")
                for op_type, logs in operations.items():
                    print(f"    {op_type}: {len(logs)} occurrences")
                    
                    # Calculate average data transfer
                    data_sizes = [log['data_size_mb'] for log in logs if log.get('data_size_mb')]
                    if data_sizes:
                        avg_size = sum(data_sizes) / len(data_sizes)
                        total_size = sum(data_sizes)
                        print(f"      Avg data size: {avg_size:.2f}MB")
                        print(f"      Total data: {total_size:.2f}MB")
    
    def generate_training_report(self, output_file: str = None):
        """Generate a comprehensive training report"""
        
        if not output_file:
            output_file = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Sessions: {len(self.sessions)}\n\n")
            
            # Session breakdown
            single_gpu = [s for s in self.sessions if s['session']['mode'] == 'single-gpu']
            multi_gpu = [s for s in self.sessions if s['session']['mode'] == 'multi-gpu']
            
            f.write(f"Single GPU Sessions: {len(single_gpu)}\n")
            f.write(f"Multi-GPU Sessions: {len(multi_gpu)}\n\n")
            
            # Detailed analysis
            for session in self.sessions:
                f.write(f"Session: {session['session']['id']}\n")
                f.write(f"Mode: {session['session']['mode']}\n")
                f.write(f"Duration: {session['session']['duration_seconds']:.1f}s\n")
                f.write(f"GPUs: {session['session']['gpu_count']}\n")
                
                if session['training_metrics']:
                    metrics = session['training_metrics']
                    f.write(f"Training Steps: {len(metrics)}\n")
                    f.write(f"Final Loss: {metrics[-1]['loss']:.4f}\n")
                
                f.write("\n" + "-"*40 + "\n\n")
        
        print(f"Training report saved to: {output_file}")
    
    def plot_training_curves(self, session_id: str = None):
        """Plot training curves for analysis"""
        
        sessions_to_plot = self.sessions
        if session_id:
            sessions_to_plot = [s for s in self.sessions if session_id in s['session']['id']]
        
        if not sessions_to_plot:
            print("No sessions found for plotting.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Loss curves
        plt.subplot(2, 2, 1)
        for session in sessions_to_plot:
            if session['training_metrics']:
                metrics = session['training_metrics']
                steps = [m['step'] for m in metrics]
                losses = [m['loss'] for m in metrics]
                
                label = f"{session['session']['id']} ({session['session']['mode']})"
                plt.plot(steps, losses, label=label)
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: GPU Utilization
        plt.subplot(2, 2, 2)
        for session in sessions_to_plot:
            if session['gpu_snapshots']:
                snapshots = session['gpu_snapshots']
                times = range(len(snapshots))
                
                # Average GPU utilization across all GPUs
                avg_utils = []
                for snapshot in snapshots:
                    total_util = sum(gpu['utilization'] for gpu in snapshot['gpus'])
                    avg_util = total_util / len(snapshot['gpus'])
                    avg_utils.append(avg_util)
                
                label = f"{session['session']['id']} ({session['session']['gpu_count']} GPUs)"
                plt.plot(times, avg_utils, label=label)
        
        plt.xlabel('Time (snapshots)')
        plt.ylabel('GPU Utilization (%)')
        plt.title('GPU Utilization Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Learning Rate
        plt.subplot(2, 2, 3)
        for session in sessions_to_plot:
            if session['training_metrics']:
                metrics = session['training_metrics']
                steps = [m['step'] for m in metrics]
                lrs = [m['learning_rate'] for m in metrics]
                
                label = f"{session['session']['id']}"
                plt.plot(steps, lrs, label=label)
        
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Training Speed
        plt.subplot(2, 2, 4)
        session_names = []
        speeds = []
        
        for session in sessions_to_plot:
            if session['training_metrics']:
                duration = session['session']['duration_seconds']
                steps = len(session['training_metrics'])
                speed = steps / duration if duration > 0 else 0
                
                session_names.append(f"{session['session']['mode']}\n({session['session']['gpu_count']} GPUs)")
                speeds.append(speed)
        
        if session_names and speeds:
            plt.bar(session_names, speeds)
            plt.ylabel('Steps per Second')
            plt.title('Training Speed Comparison')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"training_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {plot_file}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("--log-dir", default="./logs", help="Directory containing training logs")
    parser.add_argument("--compare", action="store_true", help="Compare single vs multi-GPU sessions")
    parser.add_argument("--communication", action="store_true", help="Analyze multi-GPU communication")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--plot", action="store_true", help="Generate training plots")
    parser.add_argument("--session-id", type=str, help="Analyze specific session")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"Log directory not found: {args.log_dir}")
        return
    
    analyzer = TrainingLogAnalyzer(args.log_dir)
    
    if not analyzer.sessions:
        print("No training sessions found in log directory.")
        return
    
    print(f"Found {len(analyzer.sessions)} training sessions")
    
    if args.compare:
        analyzer.compare_single_vs_multi_gpu()
    
    if args.communication:
        analyzer.analyze_multi_gpu_communication()
    
    if args.report:
        analyzer.generate_training_report()
    
    if args.plot:
        analyzer.plot_training_curves(args.session_id)
    
    # Default: show basic comparison
    if not any([args.compare, args.communication, args.report, args.plot]):
        analyzer.compare_single_vs_multi_gpu()


if __name__ == "__main__":
    main()