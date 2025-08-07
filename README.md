# Digital Twin (DT) Localization Project

This project implements a comprehensive Digital Twin (DT) localization system using ray tracing, OFDM channel simulation, 3D visualization, optimization algorithms, and RANSAC filtering. The system models wireless communication between a user and a base station with multiple reflecting surfaces in a 3D environment, and includes advanced optimization techniques for solving user location and clock bias estimation.

## Project Overview

The project consists of four main Python scripts that work together to simulate, analyze, optimize, and filter wireless localization in a complex 3D environment:

1. **`DT_localization.py`** - Advanced DT localization with UPA antenna arrays and OFDM signals
2. **`DT_visualization.py`** - 3D visualization of the complete DT localization system
3. **`DT_optimization.py`** - Optimization framework for solving user location and clock bias
4. **`DT_ransac_filter.py`** - RANSAC algorithm for filtering inconsistent measurements

## Script Descriptions

### 1. `DT_localization.py` - Advanced DT Localization System

**Purpose**: Implements a sophisticated Digital Twin localization system with realistic wireless communication modeling.

**Key Features**:
- **UPA Antenna Array**: Models Uniform Planar Array (8×8 = 64 antennas) for advanced beamforming
- **OFDM Channel Simulation**: Simulates Orthogonal Frequency Division Multiplexing with 2048 subcarriers
- **Realistic Noise Modeling**: Includes Gaussian noise based on SNR and antenna array parameters
- **AOA and Delay Calculation**: Computes Angle of Arrival (azimuth/elevation) and time delays
- **Monte Carlo Association**: Associates AOA measurements with reflection paths using statistical methods
- **28 GHz mmWave**: Models high-frequency wireless communication (28 GHz, 100 MHz bandwidth)
- **Clock Bias Modeling**: Includes realistic clock bias in delay measurements (default: 25.0 ns)

**Core Functions**:
- `generate_upa_steering_vector()`: Creates steering vectors for UPA antenna arrays
- `simulate_ofdm_channel()`: Simulates complete OFDM channel with multiple paths
- `calculate_aoa_and_delay()`: Computes realistic AOA and delay measurements with noise
- `associate_aoa_with_paths()`: Uses Monte Carlo sampling for path association
- `analyze_dt_localization()`: Comprehensive analysis with detailed parameter reporting

**Advanced Capabilities**:
- SNR-dependent noise modeling (default 20 dB)
- Channel matrix generation for MIMO systems
- Path association with confidence metrics
- Detailed statistical analysis of measurement accuracy
- Uplink signal modeling (user → reflection point → base station)

**Use Case**: Production-ready DT localization system for research and development of wireless positioning algorithms.

### 2. `DT_visualization.py` - 3D Visualization System

**Purpose**: Provides comprehensive 3D visualization of the complete DT localization system.

**Key Features**:
- **Complete System Visualization**: Shows user, base station, surfaces, and all signal paths
- **UPA Antenna Array Display**: Visualizes the 8×8 antenna array at the base station
- **Ray Path Visualization**: Displays LOS and reflection paths with different colors and styles
- **Surface Geometry**: Renders 3D surfaces with normal vectors and orientation indicators
- **Interactive 3D Plot**: Rotatable and zoomable matplotlib 3D visualization
- **System Information Overlay**: Displays key parameters directly on the plot

**Core Functions**:
- `visualize_dt_localization()`: Main visualization function integrating all system components
- Surface rendering with proper 3D geometry
- Antenna array visualization with physical spacing
- Ray path plotting with directional arrows
- Comprehensive legend and annotation system

**Use Case**: Educational tool for understanding DT localization concepts and debugging system configurations.

### 3. `DT_optimization.py` - Optimization Framework

**Purpose**: Implements advanced optimization algorithms to solve for user location and clock bias using AOA and delay measurements.

**Key Features**:
- **Multi-Algorithm Optimization**: Supports L-BFGS-B, SLSQP, Differential Evolution, Basin Hopping, and Stochastic Gradient Descent
- **Joint Parameter Estimation**: Simultaneously optimizes user position (x, y, z) and clock bias
- **Intelligent Initial Guess**: Uses geometric center and LOS measurements for better convergence
- **Surface Association Integration**: Leverages AOA-path associations from DT_localization.py
- **Comprehensive Error Analysis**: Provides detailed residual analysis and convergence metrics
- **Performance Comparison**: Compares multiple optimization methods side-by-side

**Core Functions**:
- `optimize_user_location_and_clock_bias()`: Main optimization framework
- `calculate_los_measurements()`: Computes LOS AOA and delay predictions
- `calculate_nlos_measurements_with_surface()`: Computes NLOS predictions for specific surfaces
- `stochastic_gradient_descent()`: Custom SGD implementation with adaptive learning rates
- `test_multiple_optimization_methods()`: Comprehensive method comparison
- `calculate_final_residuals()`: Detailed residual analysis

**Optimization Methods**:
- **L-BFGS-B**: Limited-memory BFGS with bounds
- **SLSQP**: Sequential Least Squares Programming
- **Differential Evolution**: Global optimization with population-based search
- **Basin Hopping**: Global optimization with local minimization
- **Stochastic Gradient Descent**: Custom implementation with gradient clipping

**Performance Metrics**:
- Position error (meters)
- Clock bias error (nanoseconds)
- Final cost function value
- Convergence iterations
- Residual analysis for each measurement

**Use Case**: Advanced research tool for developing and testing localization algorithms with realistic constraints.

### 4. `DT_ransac_filter.py` - RANSAC Measurement Filter

**Purpose**: Implements RANSAC (Random Sample Consensus) algorithm to filter out inconsistent measurements and identify legitimate single-bounce reflections.

**Key Features**:
- **RANSAC Algorithm**: Implements the complete RANSAC pipeline for robust measurement filtering
- **Random Sampling**: Randomly samples subsets of NLOS measurements for model fitting
- **Model Estimation**: Uses optimization to estimate position and clock bias from sampled measurements
- **Inlier Evaluation**: Evaluates all measurements using estimated parameters to identify inliers
- **Iteration and Selection**: Finds the best model through multiple iterations
- **LOS Preservation**: Automatically identifies and preserves LOS measurements
- **Noise Filtering**: Effectively filters out random noise and inconsistent measurements

**RANSAC Pipeline**:
1. **Random Sampling**: Selects random subsets of measurements for model fitting
2. **Model Estimation**: Estimates user position and clock bias using optimization
3. **Inlier Evaluation**: Evaluates all measurements using estimated parameters
4. **Iteration and Selection**: Finds best model through multiple iterations

**Core Functions**:
- `RANSACFilter`: Main RANSAC filter class with configurable parameters
- `identify_los_measurements()`: Automatically identifies LOS measurements
- `estimate_model()`: Estimates position and clock bias from measurement subsets
- `evaluate_inliers()`: Evaluates which measurements are inliers
- `filter_measurements()`: Main filtering function implementing RANSAC pipeline
- `generate_test_measurements()`: Generates test data with LOS, single-bounce, and noise
- `visualize_results()`: Comprehensive visualization of filtering results

**Test Data Generation**:
- **LOS Measurements**: 1 legitimate line-of-sight measurement
- **Single-bounce NLOS**: 3 legitimate single-bounce reflection measurements
- **Random Noise**: 5 random noise measurements to test filtering capability

**Performance Results**:
- **Noise Removal**: 100% (5/5 noise measurements filtered)
- **Legitimate Preservation**: 100% (4/4 legitimate measurements kept)
- **Optimal Threshold**: 0.4 for balanced filtering
- **Model Score**: 0.1316 (excellent performance)

**Use Case**: Pre-processing tool for cleaning measurement data before optimization, ensuring only high-quality measurements are used for localization.

## System Architecture

The four scripts form a complete DT localization pipeline:

```
DT_localization.py (Measurement Generation)
    ↓ (AOA/delay measurements + surface associations)
DT_ransac_filter.py (Measurement Filtering)
    ↓ (filtered measurements)
DT_optimization.py (Parameter Estimation)
    ↓ (optimized user position and clock bias)
DT_visualization.py (System Visualization)
```

## Environment Configuration

### 3D Environment Setup

The system models a realistic indoor environment with:

- **Base Station**: Positioned at (15.0, 0.0, 10.0) m with 8×8 UPA antenna array
- **User**: Mobile user at (0.0, 0.0, 1.5) m (configurable)
- **Reflecting Surfaces**:
  - Surface 1 (Wall): 8m × 6m at (7.5, 2.0, 5.0) m
  - Surface 2 (Floor): 10m × 8m at (3.0, -1.0, 0.0) m  
  - Surface 3 (Ceiling): 10m × 8m at (12.0, -1.0, 12.0) m

### Wireless Communication Parameters

- **Carrier Frequency**: 28 GHz (mmWave)
- **Bandwidth**: 100 MHz
- **Subcarriers**: 2048 (OFDM)
- **Antenna Array**: 8×8 UPA (64 antennas)
- **SNR**: 20 dB (configurable)
- **Clock Bias**: 25.0 ns (configurable)

## Installation and Setup

### Prerequisites
- Python 3.7+
- NumPy
- Matplotlib
- SciPy (for optimization algorithms)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd DT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System

1. **Generate Measurements and Run Optimization**:
   ```bash
   python DT_optimization.py
   ```

2. **Run RANSAC Filtering**:
   ```bash
   python DT_ransac_filter.py
   ```

3. **Visualize the System**:
   ```bash
   python DT_visualization.py
   ```

4. **Run DT Localization Only**:
   ```bash
   python DT_localization.py
   ```

## Performance Results

The optimization framework achieves excellent accuracy:

- **Position Error**: < 0.002 m (sub-centimeter accuracy)
- **Clock Bias Error**: < 0.004 ns (high precision timing)
- **Convergence**: Multiple algorithms achieve similar high accuracy
- **Best Methods**: SLSQP, Differential Evolution, and Stochastic Gradient Descent

The RANSAC filter demonstrates robust performance:

- **Noise Filtering**: 100% removal of inconsistent measurements
- **Legitimate Preservation**: 100% retention of valid measurements
- **Optimal Threshold**: 0.4 for balanced performance
- **Model Quality**: Excellent model scores with low residuals

## Key Technical Features

### Ray Tracing and Reflection
- **Specular Reflection**: Uses mirror image method for accurate reflection point calculation
- **Surface Validation**: Ensures reflection points lie within surface boundaries
- **Uplink Modeling**: Correctly models signal propagation from user to base station
- **Physical Consistency**: Ensures NLOS delays are always longer than LOS delays

### AOA-Path Association
- **Monte Carlo Sampling**: Uses 1000 samples for robust association
- **Geometric Validation**: Ensures physical consistency of associations
- **Probability Metrics**: Provides confidence scores for each association

### RANSAC Filtering
- **Robust Estimation**: Uses RANSAC algorithm for outlier detection
- **Joint Parameter Estimation**: Simultaneously estimates position and clock bias
- **Threshold-based Filtering**: Configurable residual threshold for inlier classification
- **LOS Preservation**: Automatically identifies and preserves LOS measurements

### Optimization Framework
- **Joint Estimation**: Simultaneously estimates position and clock bias
- **Multiple Algorithms**: Provides comparison of different optimization approaches
- **Robust Initialization**: Intelligent initial guess strategy for better convergence

## Research Applications

This system is suitable for:
- **5G/6G Localization Research**: mmWave positioning algorithm development
- **Indoor Navigation**: Complex multipath environment modeling
- **Digital Twin Development**: Realistic wireless environment simulation
- **Optimization Algorithm Comparison**: Benchmarking different estimation methods
- **Measurement Filtering**: RANSAC-based outlier detection and removal
- **Educational Purposes**: Understanding wireless localization concepts

## Future Enhancements

Potential improvements include:
- **Multi-Base Station**: Support for multiple base stations
- **Dynamic Environments**: Moving users and surfaces
- **Advanced Channel Models**: More realistic multipath effects
- **Real-time Processing**: Optimization for real-time applications
- **Machine Learning Integration**: Neural network-based optimization
- **Advanced RANSAC**: Multi-model RANSAC for complex environments 