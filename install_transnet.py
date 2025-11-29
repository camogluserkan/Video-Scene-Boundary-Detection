"""
TransNet V2 Helper Installation Script

This script automates the installation of TransNet V2 and its dependencies (PyTorch).
TransNet V2 is not hosted on PyPI, so it must be installed directly from its GitHub repository.
This script handles that process and ensures prerequisites are met.
"""

import subprocess
import sys
import os

def install_package(package):
    """
    Helper function to install a Python package via pip.
    
    Args:
        package: The name of the package (e.g., "torch", "git+https://...")
    """
    print(f"\n{'='*60}")
    print(f"Installing: {package}")
    print('='*60)
    # Use the current Python executable to run pip, ensuring it installs to the active environment
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("="*60)
    print("TransNet V2 Installation Assistant")
    print("="*60)
    print("\nThis script will install PyTorch and TransNet V2.")
    print("Note: This process will download ~100MB of model data.")
    print("Please ensure you have an active internet connection.")
    
    response = input("\nDo you want to continue? (Y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Installation cancelled.")
        return
    
    # Core dependencies for TransNet V2
    packages = [
        'torch>=2.0.0',       # Deep learning framework
        'torchvision>=0.15.0', # Computer vision utilities
    ]
    
    # The Git URL for the TransNet V2 repository
    transnet_package = 'git+https://github.com/soCzech/TransNetV2.git'
    
    try:
        # 1. Install PyTorch dependencies first
        for package in packages:
            install_package(package)
        
        # 2. Install TransNet V2 from GitHub
        print(f"\n{'='*60}")
        print("Installing TransNet V2 from GitHub...")
        print('='*60)
        subprocess.check_call([sys.executable, "-m", "pip", "install", transnet_package])
        
        print("\n" + "="*60)
        print("✓ INSTALLATION COMPLETE!")
        print("="*60)
        print("\nTransNet V2 is ready to use.")
        print("\nTo test the installation, run:")
        print("  python src/shot_detection.py --input <video_path> --method transnet")
        
        # 3. Verify the installation by attempting to import and load the model
        print("\nVerifying TransNet V2...")
        from transnetv2 import TransNetV2
        # Initialize the model (this might trigger the weight download)
        model = TransNetV2()
        print("✓ TransNet V2 loaded successfully!")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR!")
        print("="*60)
        print(f"An error occurred during installation: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection.")
        print("2. Ensure Git is installed and added to your PATH.")
        print("3. Try installing manually using:")
        print("   pip install torch torchvision")
        print("   pip install git+https://github.com/soCzech/TransNetV2.git")

if __name__ == '__main__':
    main()
