import numpy as np

def generate_surface_vertices(reference_point, length, width, normal_vector, length_vector):
    """
    Generate vertices for a surface given its parameters.
    
    Args:
        reference_point: 3D point (x, y, z) at the center of the surface
        length: length of the surface (scalar)
        width: width of the surface (scalar)
        normal_vector: normal vector of the surface
        length_vector: vector perpendicular to normal, lying on surface, in length direction
    
    Returns:
        vertices: 4 vertices defining the surface corners
    """
    # Normalize the vectors
    normal = normal_vector / np.linalg.norm(normal_vector)
    length_dir = length_vector / np.linalg.norm(length_vector)
    
    # Ensure length_vector is perpendicular to normal
    # Project length_vector onto the surface plane
    length_dir = length_dir - np.dot(length_dir, normal) * normal
    length_dir = length_dir / np.linalg.norm(length_dir)
    
    # Calculate width direction (perpendicular to both normal and length)
    width_dir = np.cross(normal, length_dir)
    width_dir = width_dir / np.linalg.norm(width_dir)
    
    # Generate the four corners of the surface
    corners = []
    for l_sign in [-length/2, length/2]:
        for w_sign in [-width/2, width/2]:
            corner = reference_point + l_sign * length_dir + w_sign * width_dir
            corners.append(corner)
    
    return np.array(corners)

def reflect_point_about_plane(point, plane_point, plane_normal):
    """
    Reflect a point about a plane using the mirror image method.
    
    Args:
        point: Point to reflect
        plane_point: Point on the plane
        plane_normal: Normal vector of the plane (normalized)
    
    Returns:
        reflected_point: Reflected point
    """
    # Ensure normal is normalized
    normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Vector from plane point to the point to reflect
    v = point - plane_point
    
    # Calculate reflected point: P' = P - 2(N·V)N
    reflected = point - 2 * np.dot(v, normal) * normal
    
    return reflected

def find_reflection_point(user_position, bs_position, surface_center, surface_normal):
    """
    Find the reflection point on a surface using the mirror image method.
    
    Args:
        user_position: Position of the user
        bs_position: Position of the base station
        surface_center: Center point of the surface
        surface_normal: Normal vector of the surface
    
    Returns:
        reflection_point: Point on the surface where reflection occurs, or None if not possible
    """
    # Reflect the base station about the surface plane
    bs_mirror = reflect_point_about_plane(bs_position, surface_center, surface_normal)
    
    # The reflection point is where the line from user to mirror BS intersects the surface
    # Line equation: P = user + t * (bs_mirror - user)
    line_direction = bs_mirror - user_position
    line_direction = line_direction / np.linalg.norm(line_direction)
    
    # Find intersection with the plane containing the surface
    normal = surface_normal / np.linalg.norm(surface_normal)
    
    # Plane equation: N·(P - P₀) = 0
    # Line equation: P = user + t * direction
    # Substitute: N·(user + t*direction - P₀) = 0
    # Solve for t: t = N·(P₀ - user) / (N·direction)
    
    numerator = np.dot(normal, surface_center - user_position)
    denominator = np.dot(normal, line_direction)
    
    if abs(denominator) < 1e-6:  # Line is parallel to plane
        return None
    
    t = numerator / denominator
    
    if t < 0:  # Intersection is behind the user
        return None
    
    reflection_point = user_position + t * line_direction
    
    return reflection_point

def check_point_on_surface(point, surface_vertices, surface_normal, length, width, reference_point, length_vector):
    """
    Check if a point lies within the bounds of a surface.
    
    Args:
        point: Point to check
        surface_vertices: 4 vertices defining the surface
        surface_normal: Normal vector of the surface
        length: Length of the surface (scalar)
        width: Width of the surface (scalar)
        reference_point: Reference point of the surface (center)
        length_vector: Vector in length direction on the surface
    
    Returns:
        bool: True if point is on the surface, False otherwise
    """
    # Normalize the vectors
    normal = surface_normal / np.linalg.norm(surface_normal)
    length_dir = length_vector / np.linalg.norm(length_vector)
    
    # Ensure length_vector is perpendicular to normal
    # Project length_vector onto the surface plane
    length_dir = length_dir - np.dot(length_dir, normal) * normal
    length_dir = length_dir / np.linalg.norm(length_dir)
    
    # Calculate width direction (perpendicular to both normal and length)
    width_dir = np.cross(normal, length_dir)
    width_dir = width_dir / np.linalg.norm(width_dir)
    
    # Vector from reference point to the point to check
    v = point - reference_point
    
    # Project the vector onto the surface plane
    # Remove the component along the normal
    v_projected = v - np.dot(v, normal) * normal
    
    # Calculate coordinates along the surface axes
    coord_length = np.dot(v_projected, length_dir)
    coord_width = np.dot(v_projected, width_dir)
    
    # Check if point is within the surface bounds
    # Point should be no further than length/2 and width/2 from reference point
    return abs(coord_length) <= length/2 and abs(coord_width) <= width/2

def calculate_aoa_and_delay(bs_position, user_position, reflection_point=None, c=3e8, 
                           add_noise=True, snr_db=20, num_antennas=64, clock_bias=0.0):
    """
    Calculate Angle of Arrival (AOA) and time delay for a path with realistic noise.
    
    Args:
        bs_position: Position of the base station
        user_position: Position of the user
        reflection_point: Reflection point (None for LOS)
        c: Speed of light
        add_noise: Whether to add Gaussian noise to measurements
        snr_db: Signal-to-Noise Ratio in dB
        num_antennas: Number of antennas for noise modeling
        clock_bias: Clock bias in nanoseconds (default: 0.0)
    
    Returns:
        aoa_azimuth: Azimuth angle in degrees (with noise if enabled)
        aoa_elevation: Elevation angle in degrees (with noise if enabled)
        time_delay: Time delay in nanoseconds (with noise if enabled)
        path_length: Total path length in meters
        noise_info: Dictionary containing noise parameters
    """
    if reflection_point is None:
        # LOS path
        path_vector = user_position - bs_position
        path_length = np.linalg.norm(path_vector)
    else:
        # Reflection path: User -> Reflection Point -> BS
        incident_path = reflection_point - user_position
        reflected_path = bs_position - reflection_point
        path_length = np.linalg.norm(incident_path) + np.linalg.norm(reflected_path)
        path_vector = reflected_path  # Direction from reflection point to BS
    
    # Calculate AOA (from BS perspective)
    path_vector_normalized = path_vector / np.linalg.norm(path_vector)
    
    # Azimuth angle (horizontal angle)
    aoa_azimuth = np.arctan2(path_vector_normalized[1], path_vector_normalized[0]) * 180 / np.pi
    
    # Elevation angle (vertical angle)
    horizontal_distance = np.sqrt(path_vector_normalized[0]**2 + path_vector_normalized[1]**2)
    aoa_elevation = np.arctan2(path_vector_normalized[2], horizontal_distance) * 180 / np.pi
    
    # Time delay (including clock bias)
    time_delay = path_length / c * 1e9 + clock_bias  # Convert to nanoseconds and add clock bias
    
    noise_info = {'snr_db': snr_db, 'num_antennas': num_antennas, 'clock_bias': clock_bias}
    
    if add_noise:
        # Convert SNR from dB to linear scale
        snr_linear = 10**(snr_db / 10)
        
        # Realistic noise variances based on antenna array and SNR
        # AOA noise decreases with more antennas and higher SNR
        aoa_noise_std = 0.5 / (np.sqrt(snr_linear) * np.sqrt(num_antennas))  # degrees
        
        # Time delay noise depends on bandwidth and SNR
        # Assuming 100 MHz bandwidth for delay estimation
        bandwidth = 100e6  # 100 MHz
        delay_noise_std = 1e-9 / (np.sqrt(snr_linear) * np.sqrt(bandwidth * 1e-6))  # nanoseconds
        
        # Add Gaussian noise
        aoa_azimuth_noisy = aoa_azimuth + np.random.normal(0, aoa_noise_std)
        aoa_elevation_noisy = aoa_elevation + np.random.normal(0, aoa_noise_std)
        time_delay_noisy = time_delay + np.random.normal(0, delay_noise_std)
        
        noise_info.update({
            'aoa_noise_std': aoa_noise_std,
            'delay_noise_std': delay_noise_std,
            'aoa_azimuth_true': aoa_azimuth,
            'aoa_elevation_true': aoa_elevation,
            'time_delay_true': time_delay
        })
        
        return aoa_azimuth_noisy, aoa_elevation_noisy, time_delay_noisy, path_length, noise_info
    
    return aoa_azimuth, aoa_elevation, time_delay, path_length, noise_info

def generate_upa_steering_vector(azimuth, elevation, num_antennas_x, num_antennas_y, frequency, antenna_spacing=0.5):
    """
    Generate steering vector for UPA (Uniform Planar Array).
    
    Args:
        azimuth: Azimuth angle in degrees
        elevation: Elevation angle in degrees
        num_antennas_x: Number of antennas in x-direction
        num_antennas_y: Number of antennas in y-direction
        frequency: Carrier frequency in Hz
        antenna_spacing: Antenna spacing in wavelengths
    
    Returns:
        steering_vector: Complex steering vector
    """
    # Convert angles to radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    
    # Calculate direction cosines
    u = np.sin(elevation_rad) * np.cos(azimuth_rad)
    v = np.sin(elevation_rad) * np.sin(azimuth_rad)
    
    # Generate steering vector
    steering_vector = np.zeros((num_antennas_x * num_antennas_y,), dtype=complex)
    
    for i in range(num_antennas_x):
        for j in range(num_antennas_y):
            antenna_idx = i * num_antennas_y + j
            phase = 2 * np.pi * antenna_spacing * (i * u + j * v)
            steering_vector[antenna_idx] = np.exp(1j * phase)
    
    return steering_vector

def simulate_ofdm_channel(bs_position, user_position, reflections, num_antennas_x=8, num_antennas_y=8, 
                         frequency=28e9, bandwidth=100e6, num_subcarriers=2048, c=3e8,
                         add_noise=True, snr_db=20, clock_bias=0.0):
    """
    Simulate OFDM channel for UPA antenna array with noise.
    
    Args:
        bs_position: Position of the base station
        user_position: Position of the user
        reflections: List of reflection information
        num_antennas_x: Number of antennas in x-direction
        num_antennas_y: Number of antennas in y-direction
        frequency: Carrier frequency in Hz
        bandwidth: System bandwidth in Hz
        num_subcarriers: Number of OFDM subcarriers
        c: Speed of light
        add_noise: Whether to add noise to measurements
        snr_db: Signal-to-Noise Ratio in dB
        clock_bias: Clock bias in nanoseconds (default: 0.0)
    
    Returns:
        channel_info: Dictionary containing channel information
    """
    num_antennas = num_antennas_x * num_antennas_y
    subcarrier_spacing = bandwidth / num_subcarriers
    
    # Initialize channel matrix
    H = np.zeros((num_antennas, num_subcarriers), dtype=complex)
    
    # LOS path
    aoa_azimuth_los, aoa_elevation_los, time_delay_los, path_length_los, noise_info_los = calculate_aoa_and_delay(
        bs_position, user_position, add_noise=add_noise, snr_db=snr_db, num_antennas=num_antennas, clock_bias=clock_bias
    )
    
    # Generate LOS steering vector
    a_los = generate_upa_steering_vector(aoa_azimuth_los, aoa_elevation_los, 
                                       num_antennas_x, num_antennas_y, frequency)
    
    # Add LOS path to channel matrix
    for k in range(num_subcarriers):
        phase_shift = np.exp(-1j * 2 * np.pi * k * subcarrier_spacing * time_delay_los * 1e-9)
        H[:, k] += a_los * phase_shift
    
    # Reflection paths
    reflection_paths = []
    for i, reflection in enumerate(reflections):
        aoa_azimuth, aoa_elevation, time_delay, path_length, noise_info = calculate_aoa_and_delay(
            bs_position, user_position, reflection['reflection_point'], 
            add_noise=add_noise, snr_db=snr_db, num_antennas=num_antennas, clock_bias=clock_bias
        )
        
        # Generate steering vector for reflection
        a_reflection = generate_upa_steering_vector(aoa_azimuth, aoa_elevation, 
                                                  num_antennas_x, num_antennas_y, frequency)
        
        # Add reflection path to channel matrix
        for k in range(num_subcarriers):
            phase_shift = np.exp(-1j * 2 * np.pi * k * subcarrier_spacing * time_delay * 1e-9)
            H[:, k] += a_reflection * phase_shift
        
        reflection_paths.append({
            'surface_index': reflection['surface_index'],
            'aoa_azimuth': aoa_azimuth,
            'aoa_elevation': aoa_elevation,
            'time_delay': time_delay,
            'path_length': path_length,
            'noise_info': noise_info
        })
    
    return {
        'channel_matrix': H,
        'los_path': {
            'aoa_azimuth': aoa_azimuth_los,
            'aoa_elevation': aoa_elevation_los,
            'time_delay': time_delay_los,
            'path_length': path_length_los,
            'noise_info': noise_info_los
        },
        'reflection_paths': reflection_paths,
        'num_antennas': num_antennas,
        'num_subcarriers': num_subcarriers,
        'frequency': frequency,
        'bandwidth': bandwidth
    }

def calculate_specular_reflections(user_position, bs_position, surfaces):
    """
    Calculate specular reflections using the mirror image method.
    
    Args:
        user_position: Position of the user
        bs_position: Position of the base station
        surfaces: List of surface dictionaries
    
    Returns:
        reflections: List of valid reflection information
    """
    reflections = []
    
    for i, surface in enumerate(surfaces):
        # Generate surface vertices
        vertices = generate_surface_vertices(
            surface['reference_point'],
            surface['length'],
            surface['width'],
            surface['normal_vector'],
            surface['length_vector']
        )
        
        # Find reflection point using mirror image method
        reflection_point = find_reflection_point(
            user_position,
            bs_position,
            surface['reference_point'],
            surface['normal_vector']
        )
        
        if reflection_point is not None:
            # Check if reflection point lies on the surface
            if check_point_on_surface(reflection_point, vertices, surface['normal_vector'], 
                                    surface['length'], surface['width'], surface['reference_point'],
                                    surface['length_vector']):
                # Calculate incident and reflected directions
                incident_direction = (reflection_point - user_position) / np.linalg.norm(reflection_point - user_position)
                reflected_direction = (bs_position - reflection_point) / np.linalg.norm(bs_position - reflection_point)
                
                # Verify specular reflection law: angle of incidence = angle of reflection
                normal = surface['normal_vector'] / np.linalg.norm(surface['normal_vector'])
                incident_angle = np.arccos(abs(np.dot(incident_direction, normal)))
                reflected_angle = np.arccos(abs(np.dot(reflected_direction, normal)))
                
                # Check if angles are approximately equal (within 1 degree)
                if abs(incident_angle - reflected_angle) < np.pi/180:
                    # Calculate path lengths
                    incident_path_length = np.linalg.norm(reflection_point - user_position)
                    reflected_path_length = np.linalg.norm(bs_position - reflection_point)
                    total_path_length = incident_path_length + reflected_path_length
                    
                    reflections.append({
                        'surface_index': i,
                        'surface_color': surface['color'],
                        'reflection_point': reflection_point,
                        'incident_direction': incident_direction,
                        'reflected_direction': reflected_direction,
                        'surface_normal': surface['normal_vector'],
                        'incident_angle': incident_angle * 180 / np.pi,
                        'reflected_angle': reflected_angle * 180 / np.pi,
                        'incident_path_length': incident_path_length,
                        'reflected_path_length': reflected_path_length,
                        'total_path_length': total_path_length
                    })
    
    return reflections

def analyze_dt_localization(add_noise=True, snr_db=20, clock_bias=25.0):
    """
    Analyze DT localization with UPA antenna array and OFDM signals.
    
    Args:
        add_noise: Whether to add Gaussian noise to measurements
        snr_db: Signal-to-Noise Ratio in dB
        clock_bias: Clock bias in nanoseconds (default: 15.0 ns)
    """
    
    # Define positions in 3D space (x, y, z)
    user_position = np.array([0, 0, 1.5])      # User at ground level with height 1.5m
    #user_position = np.array([4.536, 0.591, 3.654])  # New user position
    bs_position = np.array([15, 0, 10])        # Base station at height 10m
    
    # Define 3 surfaces in the space with length vectors
    surfaces = [
        {
            'reference_point': np.array([7.5, 2, 5]),      # Surface 1: vertical wall (moved +2.5m along x-axis)
            'length': 8,  # 2x larger
            'width': 6,   # 2x larger
            'normal_vector': np.array([0, -1, 0]),       # Facing negative Y direction
            'length_vector': np.array([1, 0, 0]),        # Length direction along X-axis
            'color': 'red',
            'label': 'Surface 1 (Wall)'
        },
        {
            'reference_point': np.array([3, -1, 0]),     # Surface 2: horizontal surface (moved -5m along x-axis)
            'length': 10, # 2x larger (5*2)
            'width': 8,   # 2x larger (4*2)
            'normal_vector': np.array([0, 0, 1]),        # Facing upward
            'length_vector': np.array([1, 0, 0]),        # Length direction along X-axis
            'color': 'green',
            'label': 'Surface 2 (Floor)'
        },
        {
            'reference_point': np.array([12, -1, 12]),    # Surface 3: ceiling (moved +4m along x-axis)
            'length': 10, # Similar to Surface 2 (5*2)
            'width': 8,   # Similar to Surface 2 (4*2)
            'normal_vector': np.array([0, 0, -1]),       # Facing downward (ceiling)
            'length_vector': np.array([1, 0, 0]),        # Length direction along X-axis
            'color': 'blue',
            'label': 'Surface 3 (Ceiling)'
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
    
    # Simulate OFDM channel with noise
    channel_info = simulate_ofdm_channel(bs_position, user_position, reflections, 
                                        num_antennas_x, num_antennas_y, frequency, bandwidth, num_subcarriers,
                                        add_noise=add_noise, snr_db=snr_db, clock_bias=clock_bias)
    
    # Print analysis results
    print("=== DT Localization Analysis Results ===")
    print(f"User Position: ({user_position[0]:.1f}, {user_position[1]:.1f}, {user_position[2]:.1f}) m")
    print(f"Base Station Position: ({bs_position[0]:.1f}, {bs_position[1]:.1f}, {bs_position[2]:.1f}) m")
    print(f"True Clock Bias: {clock_bias:.1f} ns")
    print(f"UPA Configuration: {num_antennas_x}×{num_antennas_y} = {total_antennas} antennas")
    print(f"Carrier Frequency: {frequency/1e9:.1f} GHz")
    print(f"Bandwidth: {bandwidth/1e6:.1f} MHz")
    print(f"Number of Subcarriers: {num_subcarriers}")
    
    print("\n=== Surface Information ===")
    for i, surface in enumerate(surfaces):
        print(f"Surface {i+1}:")
        print(f"  Reference Point: ({surface['reference_point'][0]:.1f}, {surface['reference_point'][1]:.1f}, {surface['reference_point'][2]:.1f}) m")
        print(f"  Dimensions: {surface['length']}m × {surface['width']}m")
        print(f"  Normal Vector: [{surface['normal_vector'][0]:.1f}, {surface['normal_vector'][1]:.1f}, {surface['normal_vector'][2]:.1f}]")
        print(f"  Length Vector: [{surface['length_vector'][0]:.1f}, {surface['length_vector'][1]:.1f}, {surface['length_vector'][2]:.1f}]")
        print(f"  Color: {surface['color']}")
        print()
    
    print("=== Path Analysis ===")
    print("LOS Path:")
    los_info = channel_info['los_path']
    print(f"  AOA Azimuth: {los_info['aoa_azimuth']:.1f}°")
    print(f"  AOA Elevation: {los_info['aoa_elevation']:.1f}°")
    print(f"  Time Delay: {los_info['time_delay']:.2f} ns")
    print(f"  Path Length: {los_info['path_length']:.2f} m")
    
    if add_noise and 'noise_info' in los_info:
        noise_info = los_info['noise_info']
        if 'aoa_azimuth_true' in noise_info:
            print(f"  True AOA Azimuth: {noise_info['aoa_azimuth_true']:.1f}°")
            print(f"  True AOA Elevation: {noise_info['aoa_elevation_true']:.1f}°")
            print(f"  True Time Delay: {noise_info['time_delay_true']:.2f} ns")
            print(f"  AOA Noise Std: {noise_info['aoa_noise_std']:.3f}°")
            print(f"  Delay Noise Std: {noise_info['delay_noise_std']:.3f} ns")
    print()
    
    print(f"Reflection Paths ({len(channel_info['reflection_paths'])} valid):")
    for i, path in enumerate(channel_info['reflection_paths']):
        print(f"  Reflection {i+1} via Surface {path['surface_index']+1}:")
        print(f"    AOA Azimuth: {path['aoa_azimuth']:.1f}°")
        print(f"    AOA Elevation: {path['aoa_elevation']:.1f}°")
        print(f"    Time Delay: {path['time_delay']:.2f} ns")
        print(f"    Path Length: {path['path_length']:.2f} m")
        print(f"    Delay Difference (vs LOS): {path['time_delay'] - los_info['time_delay']:.2f} ns")
        
        if add_noise and 'noise_info' in path:
            noise_info = path['noise_info']
            if 'aoa_azimuth_true' in noise_info:
                print(f"    True AOA Azimuth: {noise_info['aoa_azimuth_true']:.1f}°")
                print(f"    True AOA Elevation: {noise_info['aoa_elevation_true']:.1f}°")
                print(f"    True Time Delay: {noise_info['time_delay_true']:.2f} ns")
                print(f"    AOA Noise Std: {noise_info['aoa_noise_std']:.3f}°")
                print(f"    Delay Noise Std: {noise_info['delay_noise_std']:.3f} ns")
        print()
    
    # Channel matrix statistics
    H = channel_info['channel_matrix']
    print("=== Channel Matrix Statistics ===")
    print(f"Channel Matrix Shape: {H.shape}")
    print(f"Average Channel Power: {np.mean(np.abs(H)**2):.4f}")
    print(f"Channel Condition Number: {np.linalg.cond(H):.2e}")
    
    # Noise summary
    if add_noise:
        print("\n=== Noise Analysis ===")
        print(f"SNR: {snr_db} dB")
        print(f"Number of Antennas: {total_antennas}")
        print(f"Bandwidth: {bandwidth/1e6:.1f} MHz")
        
        # Calculate average noise statistics
        aoa_noise_stds = []
        delay_noise_stds = []
        
        if 'noise_info' in los_info and 'aoa_noise_std' in los_info['noise_info']:
            aoa_noise_stds.append(los_info['noise_info']['aoa_noise_std'])
            delay_noise_stds.append(los_info['noise_info']['delay_noise_std'])
        
        for path in channel_info['reflection_paths']:
            if 'noise_info' in path and 'aoa_noise_std' in path['noise_info']:
                aoa_noise_stds.append(path['noise_info']['aoa_noise_std'])
                delay_noise_stds.append(path['noise_info']['delay_noise_std'])
        
        if aoa_noise_stds:
            print(f"Average AOA Noise Std: {np.mean(aoa_noise_stds):.3f}°")
            print(f"Average Delay Noise Std: {np.mean(delay_noise_stds):.3f} ns")
            print(f"AOA Noise Range: {min(aoa_noise_stds):.3f}° - {max(aoa_noise_stds):.3f}°")
            print(f"Delay Noise Range: {min(delay_noise_stds):.3f} ns - {max(delay_noise_stds):.3f} ns")
    
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
        },
        'ofdm_config': {
            'frequency': frequency,
            'bandwidth': bandwidth,
            'num_subcarriers': num_subcarriers
        },
        'clock_bias': clock_bias
    }

def associate_aoa_with_paths(aoa_measurements, bs_position, surfaces, K=1000, threshold=0.8, 
                            snr_db=20, num_antennas=64):
    """
    Associate AOA measurements with paths using Monte Carlo sampling and geometric ray tracing.
    
    Args:
        aoa_measurements: List of dictionaries with 'azimuth', 'elevation', 'time_delay'
        bs_position: Position of the base station
        surfaces: List of surface dictionaries
        K: Number of Monte Carlo samples
        threshold: Threshold percentage for valid association (0.8 = 80%)
        snr_db: Signal-to-Noise Ratio in dB for noise modeling
        num_antennas: Number of antennas for noise modeling
    
    Returns:
        associations: List of association results for each measurement
    """
    # Calculate noise parameters
    snr_linear = 10**(snr_db / 10)
    aoa_noise_std = 0.5 / (np.sqrt(snr_linear) * np.sqrt(num_antennas))  # degrees
    
    associations = []
    
    for i, measurement in enumerate(aoa_measurements):
        azimuth_mean = measurement['azimuth']
        elevation_mean = measurement['elevation']
        time_delay = measurement['time_delay']
        
        # Check if this is likely a LOS measurement (first measurement is typically LOS)
        is_los = (i == 0)  # Assume first measurement is LOS
        is_reflection = not is_los  # All non-LOS measurements are reflections
        
        # Count hits for each surface
        surface_hits = {j: 0 for j in range(len(surfaces))}
        total_valid_samples = 0
        
        # Generate K Monte Carlo samples
        for k in range(K):
            # Sample AOA with Gaussian noise
            azimuth_sample = np.random.normal(azimuth_mean, aoa_noise_std)
            elevation_sample = np.random.normal(elevation_mean, aoa_noise_std)
            
            # Convert AOA to unit direction vector
            azimuth_rad = np.radians(azimuth_sample)
            elevation_rad = np.radians(elevation_sample)
            
            # Calculate direction vector from BS (negate to get direction FROM BS)
            direction_vector = np.array([
                -np.cos(elevation_rad) * np.cos(azimuth_rad),
                -np.cos(elevation_rad) * np.sin(azimuth_rad),
                -np.sin(elevation_rad)
            ])
            
            # Geometric ray tracing: extend the ray from BS in the AOA direction
            ray_origin = bs_position
            ray_direction = direction_vector  # This is the direction FROM BS
            
            # Check intersection with each surface using pure geometry
            for j, surface in enumerate(surfaces):
                # Generate surface vertices
                vertices = generate_surface_vertices(
                    surface['reference_point'],
                    surface['length'],
                    surface['width'],
                    surface['normal_vector'],
                    surface['length_vector']
                )
                
                # Check if ray intersects with this surface plane and is within bounds
                intersection_point = check_ray_surface_intersection(
                    ray_origin, ray_direction, vertices, surface['normal_vector'],
                    surface['length'], surface['width'], surface['reference_point'], surface['length_vector']
                )
                
                if intersection_point is not None:
                    # Intersection point is within surface bounds (already checked in check_ray_surface_intersection)
                    surface_hits[j] += 1
                    total_valid_samples += 1
                    break  # Only count first valid intersection
        
        # Calculate association probabilities
        association_results = {
            'measurement_index': i,
            'azimuth': azimuth_mean,
            'elevation': elevation_mean,
            'time_delay': time_delay,
            'surface_associations': {},
            'valid_path': False,
            'associated_surface': None,
            'is_los': is_los
        }
        
        # For LOS measurements, don't associate with any surface
        if is_los:
            association_results['valid_path'] = False
            association_results['associated_surface'] = None
            # Still show zero probabilities for all surfaces
            for j in range(len(surfaces)):
                association_results['surface_associations'][j] = {
                    'hits': 0,
                    'probability': 0.0,
                    'surface_label': surfaces[j]['label']
                }
        elif total_valid_samples > 0:
            for j, hits in surface_hits.items():
                probability = hits / total_valid_samples
                association_results['surface_associations'][j] = {
                    'hits': hits,
                    'probability': probability,
                    'surface_label': surfaces[j]['label']
                }
                
                # Check if this surface meets the threshold
                if probability >= threshold:
                    association_results['valid_path'] = True
                    association_results['associated_surface'] = j
                    break
        else:
            # If no valid samples, still show the surface associations with zero hits
            for j in range(len(surfaces)):
                association_results['surface_associations'][j] = {
                    'hits': 0,
                    'probability': 0.0,
                    'surface_label': surfaces[j]['label']
                }
        
        associations.append(association_results)
    
    return associations

def check_ray_surface_intersection(ray_origin, ray_direction, surface_vertices, surface_normal, 
                                 surface_length=None, surface_width=None, reference_point=None, length_vector=None):
    """
    Check if a ray intersects with a surface and return the intersection point.
    
    Args:
        ray_origin: Origin point of the ray
        ray_direction: Direction vector of the ray (normalized)
        surface_vertices: 4 vertices defining the surface
        surface_normal: Normal vector of the surface
        surface_length: Length of the surface (optional, for boundary check)
        surface_width: Width of the surface (optional, for boundary check)
        reference_point: Reference point of the surface (optional, for boundary check)
        length_vector: Length vector of the surface (optional, for boundary check)
    
    Returns:
        intersection_point: Point of intersection (if exists and within bounds), None otherwise
    """
    # Get a point on the surface (use the reference point)
    surface_point = reference_point
    normal = surface_normal / np.linalg.norm(surface_normal)
    
    # Calculate intersection with the plane containing the surface
    # Using the formula: t = (N·(P0 - O)) / (N·D)
    # where P0 is a point on the plane, O is ray origin, D is ray direction
    numerator = np.dot(normal, surface_point - ray_origin)
    denominator = np.dot(normal, ray_direction)
    
    if abs(denominator) < 1e-6:  # Ray is parallel to surface
        return None
    
    t = numerator / denominator
    
    if t < 0:  # Intersection is behind the ray origin
        return None
    
    intersection_point = ray_origin + t * ray_direction
    
    # Check if intersection point lies within surface boundaries (if parameters provided)
    if (surface_length is not None and surface_width is not None and 
        reference_point is not None and length_vector is not None):
        if not check_point_on_surface(intersection_point, surface_vertices, surface_normal,
                                    surface_length, surface_width, reference_point, length_vector):
            return None  # Intersection point is outside surface boundaries
    
    return intersection_point

if __name__ == "__main__":
    # Run the DT localization analysis with noise
    print("Running DT Localization with Gaussian Noise...")
    results = analyze_dt_localization(add_noise=True, snr_db=20)
    
    print("\n" + "="*50)
    print("Running DT Localization without noise for comparison...")
    results_no_noise = analyze_dt_localization(add_noise=False)
    
    print("\n" + "="*50)
    print("Demonstrating AOA-Path Association...")
    
    # Extract AOA measurements from the results
    aoa_measurements = []
    
    # Add LOS measurement
    los_info = results['channel_info']['los_path']
    aoa_measurements.append({
        'azimuth': los_info['aoa_azimuth'],
        'elevation': los_info['aoa_elevation'],
        'time_delay': los_info['time_delay']
    })
    
    # Add reflection measurements
    for path in results['channel_info']['reflection_paths']:
        aoa_measurements.append({
            'azimuth': path['aoa_azimuth'],
            'elevation': path['aoa_elevation'],
            'time_delay': path['time_delay']
        })
    

    

    
    # Run AOA-path association
    associations = associate_aoa_with_paths(
        aoa_measurements, 
        results['bs_position'], 
        results['surfaces'],
        K=1000, 
        threshold=0.8,
        snr_db=20,
        num_antennas=64
    )
    
    # Print association results
    print("\n=== AOA-Path Association Results ===")
    for i, association in enumerate(associations):
        print(f"Measurement {i+1}:")
        if association['is_los']:
            print(f"  Type: LOS (Line-of-Sight)")
        else:
            print(f"  Type: NLOS (Non-Line-of-Sight)")
        print(f"  AOA: ({association['azimuth']:.1f}°, {association['elevation']:.1f}°)")
        print(f"  Time Delay: {association['time_delay']:.2f} ns")
        print(f"  Valid Path: {association['valid_path']}")
        
        if association['valid_path']:
            surface_idx = association['associated_surface']
            surface_label = results['surfaces'][surface_idx]['label']
            probability = association['surface_associations'][surface_idx]['probability']
            print(f"  Associated Surface: {surface_label} (Probability: {probability:.3f})")
        else:
            print("  Associated Surface: Invalid Path")
        
        print("  Surface Association Probabilities:")
        for surface_idx, surface_info in association['surface_associations'].items():
            print(f"    {surface_info['surface_label']}: {surface_info['probability']:.3f} ({surface_info['hits']} hits)")
        print() 