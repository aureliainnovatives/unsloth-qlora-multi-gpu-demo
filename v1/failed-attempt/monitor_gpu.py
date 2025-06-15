#!/usr/bin/env python3

import os
import time
import json
import argparse
import subprocess
import threading
from datetime import datetime
from typing import Dict, List
import GPUtil
import psutil


class GPUMonitor:
    def __init__(self, log_interval: int = 2, output_file: str = None):
        self.log_interval = log_interval
        self.output_file = output_file or f"gpu_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.monitoring = False
        self.data = []
        self.start_time = None
    
    def get_gpu_info(self) -> List[Dict]:
        """Get current GPU information"""
        gpus = GPUtil.getGPUs()
        gpu_info = []
        
        for i, gpu in enumerate(gpus):
            info = {
                "gpu_id": i,
                "name": gpu.name,
                "memory_used_mb": gpu.memoryUsed,
                "memory_total_mb": gpu.memoryTotal,
                "memory_util_percent": gpu.memoryUtil * 100,
                "gpu_util_percent": gpu.load * 100,
                "temperature": gpu.temperature,
            }
            gpu_info.append(info)
        
        return gpu_info
    
    def get_system_info(self) -> Dict:
        """Get system resource information"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_nvidia_smi_info(self) -> Dict:
        """Get detailed info from nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                nvidia_info = []
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        nvidia_info.append({
                            "gpu_id": int(parts[0]),
                            "name": parts[1],
                            "memory_used_mb": int(parts[2]),
                            "memory_total_mb": int(parts[3]),
                            "gpu_util_percent": int(parts[4]),
                            "temperature": int(parts[5]),
                            "power_draw_w": float(parts[6]) if parts[6] != '[N/A]' else 0.0,
                        })
                return {"nvidia_smi": nvidia_info}
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")
        
        return {"nvidia_smi": []}
    
    def log_snapshot(self):
        """Log a single snapshot of GPU/system state"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - self.start_time if self.start_time else 0,
            "gpus": self.get_gpu_info(),
            "system": self.get_system_info(),
        }
        
        # Add nvidia-smi info
        snapshot.update(self.get_nvidia_smi_info())
        
        self.data.append(snapshot)
        
        # Print current status
        self.print_current_status(snapshot)
        
        return snapshot
    
    def print_current_status(self, snapshot: Dict):
        """Print current GPU status to console"""
        print(f"\n{'='*60}")
        print(f"Time: {snapshot['timestamp']}")
        print(f"Elapsed: {snapshot['elapsed_seconds']:.1f}s")
        
        for gpu in snapshot['gpus']:
            print(f"GPU {gpu['gpu_id']} ({gpu['name']}):")
            print(f"  Utilization: {gpu['gpu_util_percent']:.1f}%")
            print(f"  Memory: {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB ({gpu['memory_util_percent']:.1f}%)")
            print(f"  Temperature: {gpu['temperature']}°C")
        
        print(f"CPU: {snapshot['system']['cpu_percent']:.1f}%")
        print(f"RAM: {snapshot['system']['memory_percent']:.1f}%")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        print(f"Starting GPU monitoring (interval: {self.log_interval}s)")
        print(f"Output file: {self.output_file}")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while self.monitoring:
                self.log_snapshot()
                time.sleep(self.log_interval)
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring and save data"""
        self.monitoring = False
        
        if self.data:
            with open(self.output_file, 'w') as f:
                json.dump({
                    "monitoring_config": {
                        "log_interval": self.log_interval,
                        "start_time": self.start_time,
                        "total_snapshots": len(self.data),
                        "duration_seconds": time.time() - self.start_time if self.start_time else 0,
                    },
                    "snapshots": self.data
                }, f, indent=2)
            
            print(f"\nMonitoring data saved to: {self.output_file}")
            self.print_summary()
    
    def print_summary(self):
        """Print monitoring summary"""
        if not self.data:
            return
        
        print(f"\n{'='*60}")
        print("MONITORING SUMMARY")
        print(f"{'='*60}")
        print(f"Total snapshots: {len(self.data)}")
        print(f"Duration: {self.data[-1]['elapsed_seconds']:.1f} seconds")
        
        # Calculate averages
        for gpu_id in range(len(self.data[0]['gpus'])):
            gpu_utils = [snap['gpus'][gpu_id]['gpu_util_percent'] for snap in self.data]
            mem_utils = [snap['gpus'][gpu_id]['memory_util_percent'] for snap in self.data]
            
            print(f"\nGPU {gpu_id} Summary:")
            print(f"  Avg Utilization: {sum(gpu_utils)/len(gpu_utils):.1f}%")
            print(f"  Max Utilization: {max(gpu_utils):.1f}%")
            print(f"  Avg Memory: {sum(mem_utils)/len(mem_utils):.1f}%")
            print(f"  Max Memory: {max(mem_utils):.1f}%")


def run_training_with_monitoring(training_command: str, monitor_file: str = None):
    """Run training command while monitoring GPU usage"""
    
    if not monitor_file:
        monitor_file = f"training_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    monitor = GPUMonitor(log_interval=2, output_file=monitor_file)
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=monitor.start_monitoring)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        print(f"Starting training: {training_command}")
        result = subprocess.run(training_command, shell=True)
        print(f"Training completed with exit code: {result.returncode}")
    except Exception as e:
        print(f"Error running training: {e}")
    finally:
        monitor.stop_monitoring()


def compare_single_vs_multi_gpu():
    """Generate comparison commands for single vs multi-GPU"""
    print("GPU Training Comparison Commands:")
    print("="*50)
    
    print("\n1. SINGLE GPU MODE:")
    print("   CUDA_VISIBLE_DEVICES=0 python train.py --force-single-gpu")
    print("   # Monitor: python monitor_gpu.py --output single_gpu_monitor.json")
    
    print("\n2. MULTI GPU MODE:")
    print("   accelerate launch train.py")
    print("   # Monitor: python monitor_gpu.py --output multi_gpu_monitor.json")
    
    print("\n3. MONITOR DURING TRAINING:")
    print("   python monitor_gpu.py --monitor-training 'accelerate launch train.py'")
    
    print("\n4. QUICK GPU CHECK:")
    print("   nvidia-smi")
    print("   python monitor_gpu.py --snapshot")


def main():
    parser = argparse.ArgumentParser(description="GPU Monitoring Tool")
    parser.add_argument("--interval", type=int, default=2, help="Monitoring interval in seconds")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--snapshot", action="store_true", help="Take single snapshot and exit")
    parser.add_argument("--monitor-training", type=str, help="Monitor during training command")
    parser.add_argument("--compare", action="store_true", help="Show comparison commands")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_single_vs_multi_gpu()
        return
    
    monitor = GPUMonitor(log_interval=args.interval, output_file=args.output)
    
    if args.snapshot:
        print("Taking GPU snapshot...")
        snapshot = monitor.log_snapshot()
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(snapshot, f, indent=2)
            print(f"Snapshot saved to: {args.output}")
    elif args.monitor_training:
        run_training_with_monitoring(args.monitor_training, args.output)
    else:
        monitor.start_monitoring()


if __name__ == "__main__":
    main()