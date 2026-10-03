"""
XORZEN Neural Engine - Desktop Application
Copyright (c) 2026 FRAZIYM AI. All Rights Reserved.
Author: Akik Faraji, Founder & CEO
"""

import sys
import os
import threading
from xorzen.gui.bootloader import XorzenBootloader
from xorzen.gui.dashboard import XorzenDashboard
from xorzen.config import ConfigFactory, ModelSize
import xorzen

def main():
    # 1. Initialize Configuration
    # We default to the 277M flagship for the desktop app
    config = ConfigFactory.get_config(ModelSize.MINI_277M)
    
    # 2. Run BIOS-style Bootloader
    # This checks hardware, AVX2 support, and compiled kernels
    boot = XorzenBootloader(config)
    print("XORZEN: Starting boot sequence...")
    boot.run() 
    
    # 3. Launch Dashboard & Start Training Thread
    # In a real app, this would be triggered by a "Start" button
    # For now, we launch the monitor for the engine
    dash = XorzenDashboard(config, total_steps=1000)
    
    def training_placeholder():
        # This is where the actual model training logic goes
        # For the standalone app, it could be a "demo" or a specific task
        import time
        for step in range(1001):
            time.sleep(0.1)
            metrics = {
                "loss": 4.5 - (step * 0.004),
                "accuracy": 0.1 + (step * 0.0008),
                "throughput": 1250.5,
                "experts": {i: (0.1 if i % 10 == 0 else 0.01) for i in range(192)}
            }
            dash.update(step, metrics)
    
    print("XORZEN: Launching Dashboard...")
    dash.start_mainloop(training_placeholder)

if __name__ == "__main__":
    # Ensure relative imports work when compiled
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
