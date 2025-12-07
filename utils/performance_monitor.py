import time
import torch
import psutil
import threading
import numpy as np
from collections import defaultdict
import json
import os
from contextlib import contextmanager

class PerformanceMonitor:
    def __init__(self, device='cuda', log_file='performance_log.json'):
        self.device = device
        self.log_file = log_file
        self.metrics = {
            'training': {
                'times': [],
                'gpu_memory': [],
                'gpu_utilization': []
            },
            'inference': {
                'times': [],
                'gpu_memory': [],
                'gpu_utilization': []
            }
        }
        self.current_metrics = {}
        self._monitoring = False
        self._monitor_thread = None
        
    def reset_metrics(self):
        """Reset all collected metrics"""
        self.metrics = {
            'training': {
                'times': [],
                'gpu_memory': [],
                'gpu_utilization': []
            },
            'inference': {
                'times': [],
                'gpu_memory': [],
                'gpu_utilization': []
            }
        }
    
    def _monitor_gpu_usage(self, mode):
        """Monitor GPU usage in a separate thread"""
        max_memory = 0
        max_utilization = 0
        
        while self._monitoring:
            if torch.cuda.is_available():
                # GPU Memory
                current_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                max_memory = max(max_memory, current_memory)
                
                # GPU Utilization (approximation using memory usage)
                total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                current_utilization = (current_memory / total_memory) * 100
                max_utilization = max(max_utilization, current_utilization)
            
            time.sleep(0.1)  # Monitor every 100ms
        
        self.current_metrics[f'{mode}_max_memory'] = max_memory
        self.current_metrics[f'{mode}_max_utilization'] = max_utilization
    
    @contextmanager
    def monitor_training(self):
        """Context manager for monitoring training phase"""
        start_time = time.time()
        
        # Start GPU monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_gpu_usage, args=('training',))
        self._monitor_thread.start()
        
        try:
            yield
        finally:
            # Stop monitoring
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join()
            
            # Record training time
            training_time = time.time() - start_time
            self.metrics['training']['times'].append(training_time)
            self.metrics['training']['gpu_memory'].append(
                self.current_metrics.get('training_max_memory', 0)
            )
            self.metrics['training']['gpu_utilization'].append(
                self.current_metrics.get('training_max_utilization', 0)
            )
    
    @contextmanager
    def monitor_inference(self):
        """Context manager for monitoring inference phase"""
        start_time = time.time()
        
        # Start GPU monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_gpu_usage, args=('inference',))
        self._monitor_thread.start()
        
        try:
            yield
        finally:
            # Stop monitoring
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join()
            
            # Record inference time
            inference_time = time.time() - start_time
            self.metrics['inference']['times'].append(inference_time)
            self.metrics['inference']['gpu_memory'].append(
                self.current_metrics.get('inference_max_memory', 0)
            )
            self.metrics['inference']['gpu_utilization'].append(
                self.current_metrics.get('inference_max_utilization', 0)
            )
    
    def get_training_stats(self):
        """Get training performance statistics"""
        if not self.metrics['training']['times']:
            return None
        
        return {
            'avg_training_time': np.mean(self.metrics['training']['times']),
            'total_training_time': np.sum(self.metrics['training']['times']),
            'max_gpu_memory_gb': np.max(self.metrics['training']['gpu_memory']) if self.metrics['training']['gpu_memory'] else 0,
            'avg_gpu_memory_gb': np.mean(self.metrics['training']['gpu_memory']) if self.metrics['training']['gpu_memory'] else 0,
            'max_gpu_utilization': np.max(self.metrics['training']['gpu_utilization']) if self.metrics['training']['gpu_utilization'] else 0,
            'num_training_sessions': len(self.metrics['training']['times'])
        }
    
    def get_inference_stats(self):
        """Get inference performance statistics"""
        if not self.metrics['inference']['times']:
            return None
        
        return {
            'avg_inference_time': np.mean(self.metrics['inference']['times']),
            'total_inference_time': np.sum(self.metrics['inference']['times']),
            'max_gpu_memory_gb': np.max(self.metrics['inference']['gpu_memory']) if self.metrics['inference']['gpu_memory'] else 0,
            'avg_gpu_memory_gb': np.mean(self.metrics['inference']['gpu_memory']) if self.metrics['inference']['gpu_memory'] else 0,
            'max_gpu_utilization': np.max(self.metrics['inference']['gpu_utilization']) if self.metrics['inference']['gpu_utilization'] else 0,
            'num_inference_sessions': len(self.metrics['inference']['times'])
        }
    
    def get_all_stats(self):
        """Get all performance statistics"""
        return {
            'training': self.get_training_stats(),
            'inference': self.get_inference_stats()
        }
    
    def save_metrics(self, filepath=None):
        """Save metrics to JSON file"""
        if filepath is None:
            filepath = self.log_file
        
        stats = self.get_all_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def load_metrics(self, filepath=None):
        """Load metrics from JSON file"""
        if filepath is None:
            filepath = self.log_file
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    def print_stats(self):
        """Print formatted performance statistics"""
        stats = self.get_all_stats()
        
        print("="*50)
        print("PERFORMANCE MONITORING RESULTS")
        print("="*50)
        
        if stats['training']:
            print("\n📊 TRAINING METRICS:")
            print(f"  • Average Training Time: {stats['training']['avg_training_time']:.2f} seconds")
            print(f"  • Total Training Time: {stats['training']['total_training_time']:.2f} seconds")
            print(f"  • Max GPU Memory Usage: {stats['training']['max_gpu_memory_gb']:.2f} GB")
            print(f"  • Average GPU Memory: {stats['training']['avg_gpu_memory_gb']:.2f} GB")
            print(f"  • Max GPU Utilization: {stats['training']['max_gpu_utilization']:.1f}%")
            print(f"  • Training Sessions: {stats['training']['num_training_sessions']}")
        
        if stats['inference']:
            print("\n🔍 INFERENCE METRICS:")
            print(f"  • Average Inference Time: {stats['inference']['avg_inference_time']:.4f} seconds")
            print(f"  • Total Inference Time: {stats['inference']['total_inference_time']:.2f} seconds")
            print(f"  • Max GPU Memory Usage: {stats['inference']['max_gpu_memory_gb']:.2f} GB")
            print(f"  • Average GPU Memory: {stats['inference']['avg_gpu_memory_gb']:.2f} GB")
            print(f"  • Max GPU Utilization: {stats['inference']['max_gpu_utilization']:.1f}%")
            print(f"  • Inference Sessions: {stats['inference']['num_inference_sessions']}")
        
        print("="*50)


