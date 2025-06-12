#!/usr/bin/env python3

import os
import time
import json
import argparse
import threading
import signal
import sys
from datetime import datetime
from typing import Dict, List
import GPUtil
import psutil
import curses
import subprocess


class RealTimeGPUMonitor:
    """Real-time GPU monitoring with live dashboard"""
    
    def __init__(self):
        self.monitoring = False
        self.data_history = []
        self.start_time = None
        self.training_process = None
        
    def get_gpu_stats(self) -> List[Dict]:
        """Get current GPU statistics"""
        try:
            gpus = GPUtil.getGPUs()
            gpu_stats = []
            
            for gpu in gpus:
                stats = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': gpu.memoryUtil * 100,
                    'temperature': gpu.temperature,
                }
                gpu_stats.append(stats)
            
            return gpu_stats
        except Exception as e:
            return []
    
    def get_training_processes(self) -> List[Dict]:
        """Find Python training processes"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if any(keyword in cmdline.lower() for keyword in ['train.py', 'accelerate', 'torch']):
                            processes.append({
                                'pid': proc.info['pid'],
                                'cmd': cmdline[:80] + '...' if len(cmdline) > 80 else cmdline,
                                'cpu': proc.info['cpu_percent'],
                                'memory': proc.info['memory_percent']
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass
        
        return processes
    
    def display_dashboard(self, stdscr):
        """Display live monitoring dashboard using curses"""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(1000)  # 1 second timeout
        
        # Color pairs
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        
        while self.monitoring:
            try:
                stdscr.clear()
                height, width = stdscr.getmaxyx()
                
                # Header
                title = "🚀 REAL-TIME GPU MONITORING DASHBOARD"
                stdscr.addstr(0, (width - len(title)) // 2, title, curses.color_pair(4) | curses.A_BOLD)
                
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elapsed = time.time() - self.start_time if self.start_time else 0
                stdscr.addstr(1, 2, f"Time: {current_time} | Elapsed: {elapsed:.1f}s", curses.color_pair(1))
                stdscr.addstr(2, 2, "Press 'q' to quit, 's' to save snapshot", curses.color_pair(2))
                
                # GPU Stats
                gpu_stats = self.get_gpu_stats()
                row = 4
                
                if gpu_stats:
                    stdscr.addstr(row, 2, "GPU STATISTICS:", curses.color_pair(4) | curses.A_BOLD)
                    row += 1
                    stdscr.addstr(row, 2, "="*70, curses.color_pair(4))
                    row += 1
                    
                    for gpu in gpu_stats:
                        # GPU name and ID
                        gpu_line = f"GPU {gpu['id']}: {gpu['name']}"
                        stdscr.addstr(row, 2, gpu_line, curses.color_pair(5) | curses.A_BOLD)
                        row += 1
                        
                        # Utilization
                        util_color = curses.color_pair(1) if gpu['utilization'] > 70 else curses.color_pair(2) if gpu['utilization'] > 30 else curses.color_pair(3)
                        util_bar = self.create_progress_bar(gpu['utilization'], 40)
                        stdscr.addstr(row, 4, f"Utilization: {gpu['utilization']:5.1f}% {util_bar}", util_color)
                        row += 1
                        
                        # Memory
                        memory_color = curses.color_pair(1) if gpu['memory_percent'] < 80 else curses.color_pair(2) if gpu['memory_percent'] < 90 else curses.color_pair(3)
                        memory_bar = self.create_progress_bar(gpu['memory_percent'], 40)
                        stdscr.addstr(row, 4, f"Memory:      {gpu['memory_percent']:5.1f}% {memory_bar}", memory_color)
                        stdscr.addstr(row+1, 4, f"             {gpu['memory_used']:,}MB / {gpu['memory_total']:,}MB")
                        row += 2
                        
                        # Temperature
                        temp_color = curses.color_pair(1) if gpu['temperature'] < 70 else curses.color_pair(2) if gpu['temperature'] < 80 else curses.color_pair(3)
                        stdscr.addstr(row, 4, f"Temperature: {gpu['temperature']:3.0f}°C", temp_color)
                        row += 2
                
                # Training Processes
                processes = self.get_training_processes()
                if processes and row < height - 10:
                    stdscr.addstr(row, 2, "TRAINING PROCESSES:", curses.color_pair(4) | curses.A_BOLD)
                    row += 1
                    stdscr.addstr(row, 2, "="*70, curses.color_pair(4))
                    row += 1
                    
                    for proc in processes[:5]:  # Show max 5 processes
                        if row >= height - 2:
                            break
                        proc_line = f"PID {proc['pid']}: {proc['cmd'][:50]}"
                        stdscr.addstr(row, 2, proc_line, curses.color_pair(1))
                        row += 1
                        stdscr.addstr(row, 4, f"CPU: {proc['cpu']:5.1f}% | Memory: {proc['memory']:5.1f}%")
                        row += 1
                
                # System Stats
                if row < height - 6:
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    
                    stdscr.addstr(row, 2, "SYSTEM RESOURCES:", curses.color_pair(4) | curses.A_BOLD)
                    row += 1
                    stdscr.addstr(row, 2, "="*70, curses.color_pair(4))
                    row += 1
                    
                    cpu_color = curses.color_pair(1) if cpu_percent < 70 else curses.color_pair(2) if cpu_percent < 90 else curses.color_pair(3)
                    cpu_bar = self.create_progress_bar(cpu_percent, 30)
                    stdscr.addstr(row, 4, f"CPU:    {cpu_percent:5.1f}% {cpu_bar}", cpu_color)
                    row += 1
                    
                    mem_color = curses.color_pair(1) if memory.percent < 70 else curses.color_pair(2) if memory.percent < 90 else curses.color_pair(3)
                    mem_bar = self.create_progress_bar(memory.percent, 30)
                    stdscr.addstr(row, 4, f"Memory: {memory.percent:5.1f}% {mem_bar}", mem_color)
                    row += 1
                
                # Instructions
                if row < height - 2:
                    stdscr.addstr(height-2, 2, "Commands: [q]uit | [s]napshot | [r]eset | [SPACE]pause", curses.color_pair(2))
                
                stdscr.refresh()
                
                # Handle input
                key = stdscr.getch()
                if key == ord('q'):
                    self.monitoring = False
                elif key == ord('s'):
                    self.save_snapshot()
                elif key == ord('r'):
                    self.start_time = time.time()
                    self.data_history.clear()
                
                # Store data point
                self.data_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'elapsed': time.time() - self.start_time if self.start_time else 0,
                    'gpus': gpu_stats,
                    'processes': processes,
                    'system': {
                        'cpu': psutil.cpu_percent(),
                        'memory': psutil.virtual_memory().percent
                    }
                })
                
                # Keep only last 1000 data points
                if len(self.data_history) > 1000:
                    self.data_history = self.data_history[-1000:]
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.monitoring = False
            except Exception as e:
                # Handle terminal resize or other errors
                pass
    
    def create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """Create a text progress bar"""
        filled = int(percentage / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def save_snapshot(self):
        """Save current monitoring data"""
        if self.data_history:
            filename = f"gpu_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'snapshot_time': datetime.now().isoformat(),
                    'monitoring_duration': time.time() - self.start_time if self.start_time else 0,
                    'data_points': len(self.data_history),
                    'current_stats': self.data_history[-1] if self.data_history else None,
                    'history': self.data_history[-100:]  # Last 100 points
                }, f, indent=2)
            
            # Show message (this won't work in curses mode, but data is saved)
            pass
    
    def start_monitoring(self):
        """Start the monitoring dashboard"""
        self.monitoring = True
        self.start_time = time.time()
        
        try:
            curses.wrapper(self.display_dashboard)
        except KeyboardInterrupt:
            self.monitoring = False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup when stopping"""
        if self.data_history:
            print(f"\n📊 Monitoring completed. Captured {len(self.data_history)} data points.")
            
            # Save final data
            filename = f"realtime_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'session_info': {
                        'start_time': self.start_time,
                        'duration': time.time() - self.start_time if self.start_time else 0,
                        'data_points': len(self.data_history)
                    },
                    'final_stats': self.data_history[-1] if self.data_history else None,
                    'history': self.data_history
                }, f, indent=2)
            
            print(f"💾 Monitoring data saved to: {filename}")
            
            # Show summary
            if self.data_history:
                final_stats = self.data_history[-1]
                print(f"\n📈 Final GPU Status:")
                for gpu in final_stats.get('gpus', []):
                    print(f"  GPU {gpu['id']}: {gpu['utilization']:.1f}% util, {gpu['memory_percent']:.1f}% memory")


class SimpleMonitor:
    """Simple non-interactive monitoring for headless systems"""
    
    def __init__(self, interval: int = 2, duration: int = None):
        self.interval = interval
        self.duration = duration
        self.monitoring = False
        self.data = []
    
    def start_monitoring(self):
        """Start simple monitoring without curses"""
        self.monitoring = True
        start_time = time.time()
        
        print("🖥️  Starting simple GPU monitoring...")
        print("Press Ctrl+C to stop")
        print("="*60)
        
        try:
            while self.monitoring:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check duration limit
                if self.duration and elapsed > self.duration:
                    break
                
                # Get GPU stats
                try:
                    gpus = GPUtil.getGPUs()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    print(f"\n[{timestamp}] Elapsed: {elapsed:.1f}s")
                    
                    for gpu in gpus:
                        util_bar = "█" * int(gpu.load * 20) + "░" * (20 - int(gpu.load * 20))
                        mem_bar = "█" * int(gpu.memoryUtil * 20) + "░" * (20 - int(gpu.memoryUtil * 20))
                        
                        print(f"GPU {gpu.id}: {gpu.name}")
                        print(f"  Util: {gpu.load*100:5.1f}% [{util_bar}]")
                        print(f"  Mem:  {gpu.memoryUtil*100:5.1f}% [{mem_bar}] ({gpu.memoryUsed:,}/{gpu.memoryTotal:,}MB)")
                        print(f"  Temp: {gpu.temperature}°C")
                    
                    # Store data
                    self.data.append({
                        'timestamp': datetime.now().isoformat(),
                        'elapsed': elapsed,
                        'gpus': [{
                            'id': gpu.id,
                            'utilization': gpu.load * 100,
                            'memory_percent': gpu.memoryUtil * 100,
                            'memory_used': gpu.memoryUsed,
                            'temperature': gpu.temperature
                        } for gpu in gpus]
                    })
                    
                except Exception as e:
                    print(f"Error getting GPU stats: {e}")
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoring stopped by user")
        
        # Save data
        if self.data:
            filename = f"simple_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'monitoring_info': {
                        'interval': self.interval,
                        'duration': elapsed,
                        'data_points': len(self.data)
                    },
                    'data': self.data
                }, f, indent=2)
            
            print(f"💾 Data saved to: {filename}")
            
            # Summary
            if self.data:
                print(f"\n📊 Summary:")
                print(f"  Duration: {elapsed:.1f}s")
                print(f"  Data points: {len(self.data)}")
                
                for gpu_id in range(len(self.data[0]['gpus'])):
                    utils = [d['gpus'][gpu_id]['utilization'] for d in self.data]
                    avg_util = sum(utils) / len(utils)
                    max_util = max(utils)
                    print(f"  GPU {gpu_id}: Avg {avg_util:.1f}%, Peak {max_util:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Real-time GPU monitoring")
    parser.add_argument("--simple", action="store_true", help="Use simple text-based monitoring")
    parser.add_argument("--interval", type=int, default=2, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, help="Monitoring duration in seconds")
    parser.add_argument("--output", type=str, help="Output file for monitoring data")
    
    args = parser.parse_args()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Stopping monitor...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if args.simple:
        monitor = SimpleMonitor(interval=args.interval, duration=args.duration)
        monitor.start_monitoring()
    else:
        try:
            monitor = RealTimeGPUMonitor()
            monitor.start_monitoring()
        except Exception as e:
            print(f"Dashboard mode failed ({e}), falling back to simple mode...")
            monitor = SimpleMonitor(interval=args.interval, duration=args.duration)
            monitor.start_monitoring()


if __name__ == "__main__":
    main()