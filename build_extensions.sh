#!/bin/bash

# Build script for pre-building CUDA extensions
echo "Building SegQuant CUDA Extensions..."

# Set CUTLASS path if not set
export CUTLASS_PATH=${CUTLASS_PATH:-/usr/local/cutlass}
echo "Using CUTLASS_PATH: $CUTLASS_PATH"

# Clean previous build
echo "Cleaning previous build..."
rm -rf build/
rm -rf segquant.egg-info/
rm -rf dist/

# Install in development mode with pre-built extensions
echo "Installing with pre-built extensions..."
pip install -e . --verbose

# After successful build, switch to pre-built loader
if [ $? -eq 0 ]; then
    echo "Build successful! Switching to pre-built extension loader..."
    
    # Backup original __init__.py
    if [ ! -f segquant/__init___original.py ]; then
        cp segquant/__init__.py segquant/__init___original.py
        echo "Backed up original __init__.py"
    fi
    
    # Replace with pre-built version
    cp segquant/__init___prebuilt.py segquant/__init__.py
    echo "Switched to pre-built extension loader"
    
    echo "✅ All extensions pre-built successfully!"
    echo "Extensions will now load instantly on import."
else
    echo "❌ Build failed!"
    exit 1
fi 