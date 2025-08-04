import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DT_localization import (generate_surface_vertices, calculate_specular_reflections, 
                           calculate_aoa_and_delay, simulate_ofdm_channel)

def visualize_dt_localization():
    """
    Visualize DT localization with UPA antenna array and ray paths.
    """
    
    # Define positions in 3D space (x, y, z)
    user_position = np.array([0, 0, 1.5])      # User at ground level with height 1.5m
    bs_position = np.array([15, 0, 10])        # Base station at height 10m
    
    # Define 3 surfaces in the space with length vectors
    surfaces = [
        {
            'reference_point': np.array([5, 2, 5]),      # Surface 1: vertical wall
            'length': 8,  # 2x larger
            'width': 6,   # 2x larger
            'normal_vector': np.array([0, -1, 0]),       # Facing negative Y direction
            'length_vector': np.array([1, 0, 0]),        # Length direction along X-axis
            'color': 'red',
            'label': 'Surface 1 (Wall)'
        },
        {
            'reference_point': np.array([8, -1, 3]),     # Surface 2: horizontal surface
            'length': 5,  # Original size
            'width': 4,   # Original size
            'normal_vector': np.array([0, 0, 1]),        # Facing upward
            'length_vector': np.array([1, 0, 0]),        # Length direction along X-axis
            'color': 'green',
            'label': 'Surface 2 (Floor)'
        },
        {
            'reference_point': np.array([12, 1, 7]),     # Surface 3: inclined surface
            'length': 3,  # Original size
            'width': 3,   # Original size
            'normal_vector': np.array([-1, 0, 1]),       # Inclined surface
            'length_vector': np.array([0, 1, 0]),        # Length direction along Y-axis
            'color': 'orange',
            'label': 'Surface 3 (Inclined)'
        }
    ]
    
    # UPA antenna array parameters
    num_antennas_x = 8
    num_antennas_y = 8
    total_antennas = num_antennas_x * num_antennas_y
    
    # OFDM parameters
    frequency = 28e9  # 28 GHz mmWave
    bandwidth = 100e6  # 100 MHz
    num_subcarriers = 2048
    
    # Calculate specular reflections
    reflections = calculate_specular_reflections(user_position, bs_position, surfaces)
    
    # Simulate OFDM channel
    channel_info = simulate_ofdm_channel(bs_position, user_position, reflections, 
                                        num_antennas_x, num_antennas_y, frequency, bandwidth, num_subcarriers)
    
    # Create the 3D visualization
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot LOS path
    los_path = np.vstack([user_position, bs_position])
    ax.plot(los_path[:, 0], los_path[:, 1], los_path[:, 2], 
            'b-', linewidth=4, label='LOS Path')
    
    # Plot reflection paths
    for i, reflection in enumerate(reflections):
        color = reflection['surface_color']
        # Create reflection path: User -> Reflection Point -> BS
        reflection_path = np.vstack([user_position, reflection['reflection_point'], bs_position])
        
        ax.plot(reflection_path[:, 0], reflection_path[:, 1], reflection_path[:, 2], 
                color=color, linestyle='--', linewidth=3, 
                label=f'Reflection via Surface {reflection["surface_index"]+1}')
        
        # Mark reflection point
        ax.scatter(*reflection['reflection_point'], color=color, s=150, marker='o', alpha=0.8)
        
        # Add arrows for incident and reflected rays
        arrow_length = 1.5
        # Incident ray arrow (from user to reflection point)
        incident_direction = (reflection['reflection_point'] - user_position) / np.linalg.norm(reflection['reflection_point'] - user_position)
        ax.quiver(user_position[0], user_position[1], user_position[2],
                  incident_direction[0] * arrow_length,
                  incident_direction[1] * arrow_length,
                  incident_direction[2] * arrow_length,
                  color=color, arrow_length_ratio=0.2, linewidth=2, alpha=0.6)
        
        # Reflected ray arrow (from reflection point to BS)
        reflected_direction = (bs_position - reflection['reflection_point']) / np.linalg.norm(bs_position - reflection['reflection_point'])
        ax.quiver(reflection['reflection_point'][0], reflection['reflection_point'][1], reflection['reflection_point'][2],
                  reflected_direction[0] * arrow_length,
                  reflected_direction[1] * arrow_length,
                  reflected_direction[2] * arrow_length,
                  color=color, arrow_length_ratio=0.2, linewidth=2, alpha=0.6)
    
    # Plot user and base station positions
    ax.scatter(*user_position, color='red', s=200, label='User (Single Antenna)', zorder=5)
    ax.scatter(*bs_position, color='green', s=200, label='Base Station (UPA)', zorder=5)
    
    # Add arrows to show LOS ray direction
    los_direction = (bs_position - user_position) / np.linalg.norm(bs_position - user_position)
    arrow_length = 2
    ax.quiver(user_position[0], user_position[1], user_position[2],
              los_direction[0] * arrow_length, los_direction[1] * arrow_length, los_direction[2] * arrow_length,
              color='blue', arrow_length_ratio=0.2, linewidth=2)
    
    # Generate and plot surfaces
    for i, surface in enumerate(surfaces):
        vertices = generate_surface_vertices(
            surface['reference_point'],
            surface['length'],
            surface['width'],
            surface['normal_vector'],
            surface['length_vector']
        )
        
        # Plot surface
        x = vertices[:, 0].reshape(2, 2)
        y = vertices[:, 1].reshape(2, 2)
        z = vertices[:, 2].reshape(2, 2)
        ax.plot_surface(x, y, z, color=surface['color'], alpha=0.4, label=surface['label'])
        
        # Add normal vector arrow to show surface orientation
        normal = surface['normal_vector'] / np.linalg.norm(surface['normal_vector'])
        arrow_length = 1.5
        ax.quiver(surface['reference_point'][0], surface['reference_point'][1], surface['reference_point'][2],
                  normal[0] * arrow_length, normal[1] * arrow_length, normal[2] * arrow_length,
                  color=surface['color'], arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
        
        # Add length vector arrow
        length_dir = surface['length_vector'] / np.linalg.norm(surface['length_vector'])
        # Project onto surface plane
        length_dir = length_dir - np.dot(length_dir, normal) * normal
        length_dir = length_dir / np.linalg.norm(length_dir)
        ax.quiver(surface['reference_point'][0], surface['reference_point'][1], surface['reference_point'][2],
                  length_dir[0] * arrow_length, length_dir[1] * arrow_length, length_dir[2] * arrow_length,
                  color='black', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
        
        # Add surface reference point
        ax.scatter(*surface['reference_point'], color=surface['color'], s=100, marker='s', alpha=0.8)
    
    # Visualize UPA antenna array at BS
    antenna_spacing = 0.5  # 0.5 wavelengths
    wavelength = 3e8 / frequency
    physical_spacing = antenna_spacing * wavelength
    
    # Create UPA grid
    for i in range(num_antennas_x):
        for j in range(num_antennas_y):
            antenna_pos = bs_position + np.array([
                (i - num_antennas_x/2 + 0.5) * physical_spacing,
                (j - num_antennas_y/2 + 0.5) * physical_spacing,
                0
            ])
            ax.scatter(*antenna_pos, color='green', s=20, alpha=0.6)
    
    # Add ground plane for reference
    x_ground = np.linspace(-2, 17, 20)
    y_ground = np.linspace(-3, 3, 10)
    X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
    Z_ground = np.zeros_like(X_ground)
    ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color='gray', label='Ground')
    
    # Set plot properties
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('DT Localization: Ray Tracing with UPA Antenna Array and OFDM Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([-2, 17])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 12])
    
    # Add text annotation with system information
    los_info = channel_info['los_path']
    ax.text2D(0.02, 0.98, 
              f'UPA: {num_antennas_x}×{num_antennas_y} = {total_antennas} antennas\n'
              f'Frequency: {frequency/1e9:.1f} GHz\n'
              f'LOS AOA: ({los_info["aoa_azimuth"]:.1f}°, {los_info["aoa_elevation"]:.1f}°)\n'
              f'LOS Delay: {los_info["time_delay"]:.1f} ns\n'
              f'Valid Reflections: {len(channel_info["reflection_paths"])}', 
              transform=ax.transAxes, fontsize=10, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
              verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("=== DT Localization Visualization Results ===")
    print(f"User Position: ({user_position[0]:.1f}, {user_position[1]:.1f}, {user_position[2]:.1f}) m")
    print(f"Base Station Position: ({bs_position[0]:.1f}, {bs_position[1]:.1f}, {bs_position[2]:.1f}) m")
    print(f"UPA Configuration: {num_antennas_x}×{num_antennas_y} = {total_antennas} antennas")
    print(f"Carrier Frequency: {frequency/1e9:.1f} GHz")
    print(f"Bandwidth: {bandwidth/1e6:.1f} MHz")
    
    print("\n=== Path Analysis ===")
    print("LOS Path:")
    print(f"  AOA Azimuth: {los_info['aoa_azimuth']:.1f}°")
    print(f"  AOA Elevation: {los_info['aoa_elevation']:.1f}°")
    print(f"  Time Delay: {los_info['time_delay']:.2f} ns")
    print(f"  Path Length: {los_info['path_length']:.2f} m")
    print()
    
    print(f"Reflection Paths ({len(channel_info['reflection_paths'])} valid):")
    for i, path in enumerate(channel_info['reflection_paths']):
        print(f"  Reflection {i+1} via Surface {path['surface_index']+1}:")
        print(f"    AOA Azimuth: {path['aoa_azimuth']:.1f}°")
        print(f"    AOA Elevation: {path['aoa_elevation']:.1f}°")
        print(f"    Time Delay: {path['time_delay']:.2f} ns")
        print(f"    Path Length: {path['path_length']:.2f} m")
        print(f"    Delay Difference (vs LOS): {path['time_delay'] - los_info['time_delay']:.2f} ns")
        print()
    
    return {
        'user_position': user_position,
        'bs_position': bs_position,
        'channel_info': channel_info,
        'surfaces': surfaces,
        'reflections': reflections,
        'upa_config': {
            'num_antennas_x': num_antennas_x,
            'num_antennas_y': num_antennas_y,
            'total_antennas': total_antennas
        }
    }

if __name__ == "__main__":
    # Run the visualization
    results = visualize_dt_localization() 