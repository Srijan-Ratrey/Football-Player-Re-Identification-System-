#!/usr/bin/env python3
"""
Demo Script for Football Player Re-Identification

This script demonstrates how to use the player re-identification system
with the provided video files.
"""

import os
import sys
import argparse

def run_single_feed_demo():
    """Run demo for single feed re-identification."""
    print("="*60)
    print("RUNNING SINGLE FEED RE-IDENTIFICATION DEMO")
    print("="*60)
    
    cmd = [
        "python", "single_feed_reidentification.py",
        "--input", "data/15sec_input_720p.mp4",
        "--model", "best.pt",
        "--output", "results/demo_single_feed_output.mp4",
        "--output-dir", "results/demo_single_feed/",
        "--conf-threshold", "0.5"
    ]
    
    print("Command:", " ".join(cmd))
    print("\nProcessing...")
    
    # Execute the command
    result = os.system(" ".join(cmd))
    
    if result == 0:
        print("\n✓ Single feed demo completed successfully!")
        print("Check results in: results/demo_single_feed/")
    else:
        print("\n✗ Single feed demo failed!")

def run_quick_test():
    """Run a quick test without visualization for faster execution."""
    print("="*60)
    print("RUNNING QUICK TEST (NO VISUALIZATION)")
    print("="*60)
    
    cmd = [
        "python", "single_feed_reidentification.py",
        "--input", "data/15sec_input_720p.mp4",
        "--model", "best.pt",
        "--output-dir", "results/quick_test/",
        "--no-visualize"
    ]
    
    print("Command:", " ".join(cmd))
    print("\nProcessing...")
    
    # Execute the command
    result = os.system(" ".join(cmd))
    
    if result == 0:
        print("\n✓ Quick test completed successfully!")
        print("Check results in: results/quick_test/")
    else:
        print("\n✗ Quick test failed!")

def show_help():
    """Show help information."""
    print("="*60)
    print("FOOTBALL PLAYER RE-IDENTIFICATION DEMO")
    print("="*60)
    print()
    print("Available demo options:")
    print()
    print("1. Full Demo (with visualization):")
    print("   python demo.py --full")
    print("   - Processes 15sec_input_720p.mp4")
    print("   - Creates output video with tracking visualization")
    print("   - Generates comprehensive statistics and plots")
    print()
    print("2. Quick Test (no visualization):")
    print("   python demo.py --quick")
    print("   - Faster processing without video output")
    print("   - Still generates tracking results and statistics")
    print()
    print("3. Custom Usage:")
    print("   python single_feed_reidentification.py --help")
    print("   - See all available options and parameters")
    print()
    print("Prerequisites:")
    print("- Run 'python setup_environment.py' first")
    print("- Ensure all dependencies are installed")
    print("- Verify model file (best.pt) exists")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Demo script for Football Player Re-Identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full demo with visualization'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test without visualization'
    )
    
    args = parser.parse_args()
    
    # Check if setup has been run
    if not os.path.exists("best.pt"):
        print("Error: Model file 'best.pt' not found!")
        print("Please run 'python setup_environment.py' first.")
        return 1
    
    if not os.path.exists("data/15sec_input_720p.mp4"):
        print("Error: Input video 'data/15sec_input_720p.mp4' not found!")
        print("Please ensure video files are in the data/ directory.")
        return 1
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    if args.full:
        run_single_feed_demo()
    elif args.quick:
        run_quick_test()
    else:
        show_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 