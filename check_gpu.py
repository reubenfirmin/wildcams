#!/usr/bin/env python3
"""
GPU compatibility checker for AMD/NVIDIA systems.
"""

import sys
import subprocess

def check_system_info():
    """Check system GPU information."""
    print("🖥️  System GPU Information:")
    try:
        result = subprocess.run(['lspci', '|', 'grep', '-i', 'vga'], 
                              shell=True, capture_output=True, text=True)
        print(f"   {result.stdout.strip()}")
    except:
        print("   Could not detect GPU via lspci")
    
    try:
        result = subprocess.run(['lscpu', '|', 'grep', 'Model name'], 
                              shell=True, capture_output=True, text=True)
        print(f"   {result.stdout.strip()}")
    except:
        print("   Could not detect CPU")

def check_pytorch():
    """Check PyTorch GPU availability."""
    print("\n🤖 PyTorch GPU Status:")
    
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                print(f"   Device {i}: {device_name}")
                
                # Test basic operations
                try:
                    x = torch.randn(1000, 1000).cuda(i)
                    y = torch.mm(x, x.t())
                    print(f"   ✅ Device {i} working (basic tensor ops)")
                except Exception as e:
                    print(f"   ❌ Device {i} failed: {e}")
        else:
            print("   No GPU acceleration available")
            print(f"   CPU threads: {torch.get_num_threads()}")
            
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False
    
    return torch.cuda.is_available()

def check_rocm():
    """Check ROCm installation."""
    print("\n🚀 ROCm Status:")
    
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ ROCm installed and working")
            print("   GPU Status:")
            for line in result.stdout.split('\n')[:10]:  # First 10 lines
                if line.strip():
                    print(f"   {line}")
        else:
            print("   ❌ ROCm not working")
    except FileNotFoundError:
        print("   ❌ ROCm not installed")
        print("   Install with: sudo apt install rocm-smi")

def check_ultralytics():
    """Check if YOLO can use GPU."""
    print("\n🎯 YOLO GPU Status:")
    
    try:
        from ultralytics import YOLO
        # This will show device info when loading
        model = YOLO('yolov8n.pt')
        print("   ✅ YOLO model loaded successfully")
        
        # Check what device YOLO is using
        device = model.device
        print(f"   YOLO using device: {device}")
        
    except ImportError:
        print("   ❌ Ultralytics not installed")
    except Exception as e:
        print(f"   ⚠️  YOLO error: {e}")

def main():
    print("🔍 Wildlife Cam GPU Compatibility Check\n")
    
    check_system_info()
    has_gpu = check_pytorch()
    check_rocm()
    check_ultralytics()
    
    print("\n💡 Recommendations:")
    
    if has_gpu:
        print("   ✅ GPU acceleration should work!")
        print("   Your videos will process much faster.")
    else:
        print("   🔄 CPU-only mode will be used")
        print("   Still works, but will be slower for large videos.")
        print("\n   To enable AMD GPU acceleration:")
        print("   1. Install ROCm: https://rocm.docs.amd.com/")
        print("   2. Install ROCm PyTorch:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6")
        print("   3. Restart and run this check again")

if __name__ == "__main__":
    main()