import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_surface_vertices(reference_point, length, width, normal_vector):
    """
    Generate vertices for a surface given its reference point, dimensions, and normal vector.
    
    Args:
        reference_point: 3D point (x, y, z) where the surface is located
        length: length of the surface
        width: width of the surface  
        normal_vector: normal vector of the surface
    
    Returns:
        vertices: 4 vertices defining the surface corners
    """
    # Normalize the normal vector
    normal = normal_vector / np.linalg.norm(normal_vector)
    
    # Find two perpendicular vectors on the surface
    # First, find a vector perpendicular to the normal
    if abs(normal[0]) < 0.9:
        v1 = np.cross(normal, [1, 0, 0])
    else:
        v1 = np.cross(normal, [0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)
    
    # Second vector is perpendicular to both normal and v1
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Generate the four corners of the surface
    corners = []
    for l_sign in [-length/2, length/2]:
        for w_sign in [-width/2, width/2]:
            corner = reference_point + l_sign * v1 + w_sign * v2
            corners.append(corner)
    
    return np.array(corners)

def plot_surface(ax, vertices, color='blue', alpha=0.3, label=None):
    """
    Plot a surface defined by its vertices.
    """
    # Reshape vertices for plotting
    x = vertices[:, 0].reshape(2, 2)
    y = vertices[:, 1].reshape(2, 2)
    z = vertices[:, 2].reshape(2, 2)
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha, label=label)

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

def check_point_on_surface(point, surface_vertices, surface_normal, length, width, reference_point):
    """
    Check if a point lies within the bounds of a surface.
    
    Args:
        point: Point to check
        surface_vertices: 4 vertices defining the surface
        surface_normal: Normal vector of the surface
        length: Length of the surface
        width: Width of the surface
        reference_point: Reference point of the surface
    
    Returns:
        bool: True if point is on the surface, False otherwise
    """
    # Normalize the normal vector
    normal = surface_normal / np.linalg.norm(surface_normal)
    
    # Find two perpendicular vectors on the surface
    if abs(normal[0]) < 0.9:
        v1 = np.cross(normal, [1, 0, 0])
    else:
        v1 = np.cross(normal, [0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)
    
    # Second vector is perpendicular to both normal and v1
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Vector from reference point to the point to check
    v = point - reference_point
    
    # Project the vector onto the surface plane
    # Remove the component along the normal
    v_projected = v - np.dot(v, normal) * normal
    
    # Calculate coordinates along the surface axes
    coord1 = np.dot(v_projected, v1)
    coord2 = np.dot(v_projected, v2)
    
    # Check if point is within the surface bounds
    # Point should be no further than length/2 and width/2 from reference point
    return abs(coord1) <= length/2 and abs(coord2) <= width/2

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
            surface['normal_vector']
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
            if check_point_on_surface(reflection_point, vertices, surface['normal_vector'], surface['length'], surface['width'], surface['reference_point']):
                # Calculate incident and reflected directions
                incident_direction = (reflection_point - user_position) / np.linalg.norm(reflection_point - user_position)
                reflected_direction = (bs_position - reflection_point) / np.linalg.norm(bs_position - reflection_point)
                
                # Verify specular reflection law: angle of incidence = angle of reflection
                normal = surface['normal_vector'] / np.linalg.norm(surface['normal_vector'])
                incident_angle = np.arccos(abs(np.dot(incident_direction, normal)))
                reflected_angle = np.arccos(abs(np.dot(reflected_direction, normal)))
                
                # Check if angles are approximately equal (within 1 degree)
                if abs(incident_angle - reflected_angle) < np.pi/180:
                    # Create reflection path: User -> Reflection Point -> BS
                    reflection_path = np.vstack([user_position, reflection_point, bs_position])
                    
                    reflections.append({
                        'surface_index': i,
                        'surface_color': surface['color'],
                        'reflection_point': reflection_point,
                        'reflection_path': reflection_path,
                        'incident_direction': incident_direction,
                        'reflected_direction': reflected_direction,
                        'surface_normal': surface['normal_vector'],
                        'incident_angle': incident_angle * 180 / np.pi,
                        'reflected_angle': reflected_angle * 180 / np.pi
                    })
    
    return reflections

def simulate_ray_tracing():
    """
    Simulate Line-of-Sight (LOS) ray tracing from a user to a base station in 3D.
    """
    
    # Define positions in 3D space (x, y, z)
    user_position = np.array([0, 0, 1.5])      # User at ground level with height 1.5m
    bs_position = np.array([15, 0, 10])        # Base station at height 10m
    
    # Define 3 surfaces in the space (Surface 1 is 2x larger, others original size)
    surfaces = [
        {
            'reference_point': np.array([5, 2, 5]),      # Surface 1: vertical wall
            'length': 8,  # 2x larger
            'width': 6,   # 2x larger
            'normal_vector': np.array([0, -1, 0]),       # Facing negative Y direction
            'color': 'red',
            'label': 'Surface 1 (Wall)'
        },
        {
            'reference_point': np.array([8, -1, 3]),     # Surface 2: horizontal surface
            'length': 5,  # Original size
            'width': 4,   # Original size
            'normal_vector': np.array([0, 0, 1]),        # Facing upward
            'color': 'green',
            'label': 'Surface 2 (Floor)'
        },
        {
            'reference_point': np.array([12, 1, 7]),     # Surface 3: inclined surface
            'length': 3,  # Original size
            'width': 3,   # Original size
            'normal_vector': np.array([-1, 0, 1]),       # Inclined surface
            'color': 'orange',
            'label': 'Surface 3 (Inclined)'
        }
    ]
    
    # Calculate LOS path
    los_path = np.vstack([user_position, bs_position])
    
    # Calculate specular reflections
    reflections = calculate_specular_reflections(user_position, bs_position, surfaces)
    
    # Calculate distance and angle information
    distance_3d = np.linalg.norm(bs_position - user_position)
    distance_2d = np.sqrt((bs_position[0] - user_position[0])**2 + 
                         (bs_position[1] - user_position[1])**2)
    elevation_angle = np.arctan2(bs_position[2] - user_position[2], distance_2d) * 180 / np.pi
    
    # Create the 3D visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot LOS ray path
    ax.plot(los_path[:, 0], los_path[:, 1], los_path[:, 2], 
            'b-', linewidth=4, label='LOS Ray Path')
    
    # Plot reflection paths
    for i, reflection in enumerate(reflections):
        color = reflection['surface_color']
        ax.plot(reflection['reflection_path'][:, 0], 
                reflection['reflection_path'][:, 1], 
                reflection['reflection_path'][:, 2], 
                color=color, linestyle='--', linewidth=3, 
                label=f'Specular Reflection via Surface {reflection["surface_index"]+1}')
        
        # Mark reflection point
        ax.scatter(*reflection['reflection_point'], color=color, s=150, marker='o', alpha=0.8)
        
        # Add arrows for incident and reflected rays
        arrow_length = 1.5
        # Incident ray arrow (from user to reflection point)
        ax.quiver(user_position[0], user_position[1], user_position[2],
                  reflection['incident_direction'][0] * arrow_length,
                  reflection['incident_direction'][1] * arrow_length,
                  reflection['incident_direction'][2] * arrow_length,
                  color=color, arrow_length_ratio=0.2, linewidth=2, alpha=0.6)
        
        # Reflected ray arrow (from reflection point to BS)
        ax.quiver(reflection['reflection_point'][0], reflection['reflection_point'][1], reflection['reflection_point'][2],
                  reflection['reflected_direction'][0] * arrow_length,
                  reflection['reflected_direction'][1] * arrow_length,
                  reflection['reflected_direction'][2] * arrow_length,
                  color=color, arrow_length_ratio=0.2, linewidth=2, alpha=0.6)
    
    # Plot user and base station positions
    ax.scatter(*user_position, color='red', s=200, label='User', zorder=5)
    ax.scatter(*bs_position, color='green', s=200, label='Base Station', zorder=5)
    
    # Add arrows to show ray direction
    ray_direction = (bs_position - user_position) / np.linalg.norm(bs_position - user_position)
    arrow_length = 2
    ax.quiver(user_position[0], user_position[1], user_position[2],
              ray_direction[0] * arrow_length, ray_direction[1] * arrow_length, ray_direction[2] * arrow_length,
              color='blue', arrow_length_ratio=0.2, linewidth=2)
    
    # Generate and plot surfaces
    for i, surface in enumerate(surfaces):
        vertices = generate_surface_vertices(
            surface['reference_point'],
            surface['length'],
            surface['width'],
            surface['normal_vector']
        )
        plot_surface(ax, vertices, surface['color'], alpha=0.4, label=surface['label'])
        
        # Add normal vector arrow to show surface orientation
        normal = surface['normal_vector'] / np.linalg.norm(surface['normal_vector'])
        arrow_length = 1.5
        ax.quiver(surface['reference_point'][0], surface['reference_point'][1], surface['reference_point'][2],
                  normal[0] * arrow_length, normal[1] * arrow_length, normal[2] * arrow_length,
                  color=surface['color'], arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
        
        # Add surface reference point
        ax.scatter(*surface['reference_point'], color=surface['color'], s=100, marker='s', alpha=0.8)
    
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
    ax.set_title('3D Ray Tracing Simulation with LOS and Specular Reflections')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([-2, 17])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 12])
    
    # Add text annotation with simulation results
    ax.text2D(0.02, 0.98, f'Distance: {distance_3d:.1f}m\nElevation: {elevation_angle:.1f}°\nValid Reflections: {len(reflections)}', 
              transform=ax.transAxes, fontsize=10, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
              verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # Print simulation results
    print("=== 3D Ray Tracing Simulation Results ===")
    print(f"User Position: ({user_position[0]:.1f}, {user_position[1]:.1f}, {user_position[2]:.1f}) m")
    print(f"Base Station Position: ({bs_position[0]:.1f}, {bs_position[1]:.1f}, {bs_position[2]:.1f}) m")
    print(f"3D Distance: {distance_3d:.2f} m")
    print(f"2D Distance: {distance_2d:.2f} m")
    print(f"Elevation Angle: {elevation_angle:.1f}°")
    print(f"Ray Direction Vector: [{ray_direction[0]:.3f}, {ray_direction[1]:.3f}, {ray_direction[2]:.3f}]")
    
    print("\n=== Surface Information ===")
    for i, surface in enumerate(surfaces):
        print(f"Surface {i+1}:")
        print(f"  Reference Point: ({surface['reference_point'][0]:.1f}, {surface['reference_point'][1]:.1f}, {surface['reference_point'][2]:.1f}) m")
        print(f"  Dimensions: {surface['length']}m × {surface['width']}m")
        print(f"  Normal Vector: [{surface['normal_vector'][0]:.1f}, {surface['normal_vector'][1]:.1f}, {surface['normal_vector'][2]:.1f}]")
        print(f"  Color: {surface['color']}")
        print()
    
    print(f"\n=== Specular Reflection Analysis ===")
    print(f"Number of valid specular reflections: {len(reflections)}")
    for i, reflection in enumerate(reflections):
        print(f"Reflection {i+1} via Surface {reflection['surface_index']+1}:")
        print(f"  Reflection Point: ({reflection['reflection_point'][0]:.2f}, {reflection['reflection_point'][1]:.2f}, {reflection['reflection_point'][2]:.2f}) m")
        print(f"  Incident Direction: [{reflection['incident_direction'][0]:.3f}, {reflection['incident_direction'][1]:.3f}, {reflection['incident_direction'][2]:.3f}]")
        print(f"  Reflected Direction: [{reflection['reflected_direction'][0]:.3f}, {reflection['reflected_direction'][1]:.3f}, {reflection['reflected_direction'][2]:.3f}]")
        print(f"  Incident Angle: {reflection['incident_angle']:.1f}°")
        print(f"  Reflected Angle: {reflection['reflected_angle']:.1f}°")
        print()
    
    return {
        'user_position': user_position,
        'bs_position': bs_position,
        'distance_3d': distance_3d,
        'distance_2d': distance_2d,
        'elevation_angle': elevation_angle,
        'ray_direction': ray_direction,
        'surfaces': surfaces,
        'reflections': reflections
    }

if __name__ == "__main__":
    # Run the simulation
    results = simulate_ray_tracing() 