# KAIST Project.

This repository contains various implementations and optimizations of the FAST (Features from Accelerated Segment Test) algorithm, along with hyperparameter optimization tools and analysis scripts.

## Project Overview

The KAIST project focuses on computer vision algorithms, specifically the FAST corner detection algorithm, with multiple implementations and optimization strategies!!!!

## Files Description


### Core FAST Implementations
- `FAST.py` - Basic FAST algorithm implementation
- `FAST_improved.py` - Enhanced version with optimizations.
- `FAST_final.py` - Final optimized version
- `FAST_no_numba.py` - FAST implementation without Numba acceleration
- `FAST_mega_search.py` - Comprehensive search implementation

### Optimization Scripts
- `fast_optimizer.py` - Main optimization script for FAST parameters
- `numpy_fast_optimizer.py` - NumPy-based optimization
- `ultra_fast_optimizer.py` - Ultra-fast optimization implementation
- `super_quick_optimizer.py` - Quick optimization script
- `manual_optimizer.py` - Manual parameter optimization
- `hyperparameter_optimizer.py` - Hyperparameter optimization framework

### Search and Analysis
- `smart_search.py` - Intelligent search algorithm
- `ultimate_search.py` - Ultimate search implementation
- `grid.py` - Grid search functionality
- `aa.py` - Additional analysis script

### C++ Implementation
- `ultra_fast_cpp.cpp` - High-performance C++ implementation

### Data and Results
- `search_progress.json` - Progress tracking data
- `grid_search_results_*.json` - Grid search results from different runs
- `hyperparameter_analysis_*.png` - Analysis visualization plots

## Features

- Multiple FAST algorithm implementations with different optimization strategies
- Comprehensive hyperparameter optimization framework
- Grid search and intelligent search algorithms
- Performance analysis and visualization tools
- C++ implementation for high-performance computing
- Progress tracking and result storage

## Requirements

The project uses various Python libraries including:
- NumPy
- Numba (for some implementations)
- Matplotlib (for visualization)
- JSON (for data storage)

## Usage

Each script can be run independently depending on your specific needs:

```bash
# Run basic FAST implementation
python FAST.py

# Run optimization
python fast_optimizer.py

# Run grid search
python grid.py
```

## Results

The project includes saved results from hyperparameter optimization runs and analysis visualizations that demonstrate the performance characteristics of different implementations.

## Contributing

Feel free to contribute by:
- Improving existing implementations
- Adding new optimization strategies
- Enhancing documentation
- Reporting issues

## License

This project is part of the KAIST research initiative. 
