# Digital Twin (DT) Localization Project

This project implements a comprehensive Digital Twin (DT) localization system using ray tracing, OFDM channel simulation, and 3D visualization. The system models wireless communication between a user and a base station with multiple reflecting surfaces in a 3D environment.

## Project Overview

The project consists of three main Python scripts that work together to simulate and analyze wireless localization in a complex 3D environment:

1. **`ray_tracing_sim.py`** - Basic ray tracing simulation with specular reflections
2. **`DT_localization.py`** - Advanced DT localization with UPA antenna arrays and OFDM signals
3. **`DT_visualization.py`** - 3D visualization of the complete DT localization system

## Script Descriptions

### 1. `ray_tracing_sim.py` - Basic Ray Tracing Simulation

**Purpose**: Implements fundamental ray tracing algorithms for wireless signal propagation in 3D environments.

**Key Features**:
- **3D Ray Tracing**: Simulates Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) signal propagation
- **Specular Reflection Modeling**: Uses mirror image method to calculate reflection points on surfaces
- **Surface Geometry**: Supports multiple reflecting surfaces with different orientations and dimensions
- **Geometric Validation**: Ensures reflection points lie within surface boundaries
- **3D Visualization**: Interactive matplotlib 3D plots showing ray paths and surfaces

**Core Functions**:
- `generate_surface_vertices()`: Creates 3D surface geometry from parameters
- `reflect_point_about_plane()`: Implements mirror image reflection method
- `find_reflection_point()`: Calculates reflection points using geometric intersection
- `calculate_specular_reflections()`: Identifies valid specular reflection paths
- `simulate_ray_tracing()`: Main simulation function with visualization

**Use Case**: Foundation for understanding basic ray tracing concepts and surface reflection physics.

### 2. `DT_localization.py` - Advanced DT Localization System

**Purpose**: Implements a sophisticated Digital Twin localization system with realistic wireless communication modeling.

**Key Features**:
- **UPA Antenna Array**: Models Uniform Planar Array (8×8 = 64 antennas) for advanced beamforming
- **OFDM Channel Simulation**: Simulates Orthogonal Frequency Division Multiplexing with 2048 subcarriers
- **Realistic Noise Modeling**: Includes Gaussian noise based on SNR and antenna array parameters
- **AOA and Delay Calculation**: Computes Angle of Arrival (azimuth/elevation) and time delays
- **Monte Carlo Association**: Associates AOA measurements with reflection paths using statistical methods
- **28 GHz mmWave**: Models high-frequency wireless communication (28 GHz, 100 MHz bandwidth)

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

**Use Case**: Production-ready DT localization system for research and development of wireless positioning algorithms.

### 3. `DT_visualization.py` - 3D Visualization System

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

## System Architecture

The three scripts form a hierarchical system:

```
ray_tracing_sim.py (Foundation)
    ↓ (basic ray tracing concepts)
DT_localization.py (Advanced System)
    ↓ (realistic wireless modeling)
DT_visualization.py (Visualization)
```

## Environment Setup

### Prerequisites
- Python 3.7+
- NumPy
- Matplotlib
- Virtual environment (recommended)

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Scripts

1. **Basic Ray Tracing**:
   ```bash
   python ray_tracing_sim.py
   ```

2. **DT Localization Analysis**:
   ```bash
   python DT_localization.py
   ```

3. **3D Visualization**:
   ```bash
   python DT_visualization.py
   ```

## System Parameters

### Default Configuration
- **User Position**: (0, 0, 1.5) m (ground level with 1.5m height)
- **Base Station Position**: (15, 0, 10) m (elevated position)
- **UPA Array**: 8×8 = 64 antennas
- **Carrier Frequency**: 28 GHz (mmWave)
- **Bandwidth**: 100 MHz
- **OFDM Subcarriers**: 2048
- **SNR**: 20 dB
- **Antenna Spacing**: 0.5 wavelengths

### Reflecting Surfaces
1. **Surface 1 (Wall)**: Vertical surface at (5, 2, 5) m, 8×6 m
2. **Surface 2 (Floor)**: Horizontal surface at (8, -1, 3) m, 5×4 m  
3. **Surface 3 (Inclined)**: Angled surface at (12, 1, 7) m, 3×3 m

## Key Features

### Ray Tracing Capabilities
- **LOS Path**: Direct line-of-sight between user and base station
- **Specular Reflections**: Realistic reflection modeling using mirror image method
- **Surface Validation**: Ensures reflection points lie within surface boundaries
- **Angle Validation**: Verifies specular reflection law (incident = reflected angle)

### Wireless Communication Modeling
- **MIMO Systems**: Full channel matrix generation for multiple antennas
- **OFDM Simulation**: Multi-carrier signal processing with phase shifts
- **Noise Modeling**: Realistic Gaussian noise based on SNR and system parameters
- **AOA Estimation**: Azimuth and elevation angle calculations with noise

### Visualization Features
- **3D Interactive Plots**: Rotatable and zoomable visualizations
- **Color-coded Paths**: Different colors for LOS and reflection paths
- **Surface Rendering**: Transparent surfaces with normal vector indicators
- **Antenna Array Display**: Visual representation of UPA configuration
- **Information Overlay**: Key parameters displayed on plots

## Applications

This project is suitable for:
- **Research**: Wireless localization algorithm development
- **Education**: Understanding ray tracing and wireless communication concepts
- **Simulation**: Testing DT localization systems before real-world deployment
- **Analysis**: Performance evaluation of different antenna configurations
- **Visualization**: Communicating complex wireless concepts through 3D graphics

## Technical Details

### Mathematical Models
- **Mirror Image Method**: For specular reflection calculation
- **Geometric Intersection**: Ray-surface intersection algorithms
- **Steering Vector Generation**: UPA antenna array modeling
- **OFDM Channel Modeling**: Multi-carrier signal processing
- **Monte Carlo Sampling**: Statistical path association

### Performance Considerations
- **Computational Efficiency**: Optimized algorithms for real-time simulation
- **Memory Management**: Efficient handling of large channel matrices
- **Visualization Performance**: Smooth 3D rendering with matplotlib
- **Scalability**: Configurable parameters for different system sizes

## Future Enhancements

Potential improvements and extensions:
- **Diffuse Reflections**: Modeling of non-specular reflections
- **Multiple Users**: Support for multiple user terminals
- **Dynamic Environments**: Time-varying surface configurations
- **Machine Learning**: Integration with ML-based localization algorithms
- **Real-time Processing**: Optimization for real-time applications
- **Hardware Integration**: Interface with actual wireless hardware

## Contributing

This project is designed for educational and research purposes. Contributions are welcome for:
- Algorithm improvements
- Additional visualization features
- Performance optimizations
- Documentation enhancements
- New use case implementations

## License

This project is provided for educational and research purposes. Please ensure proper attribution when using or modifying the code. 