import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.optimize import minimize_scalar
from DT_localization import (analyze_dt_localization, associate_aoa_with_paths,
                           generate_surface_vertices, find_reflection_point, 
                           check_point_on_surface)

def stochastic_gradient_descent(objective_function, initial_guess, bounds, 
                               learning_rate=0.01, max_iterations=1000, 
                               tolerance=1e-6, batch_size=1):
    """
    Custom stochastic gradient descent implementation.
    
    Args:
        objective_function: Function to minimize
        initial_guess: Initial parameter values
        bounds: Parameter bounds [(min, max), ...]
        learning_rate: Learning rate for gradient descent
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        batch_size: Batch size for stochastic updates (1 for SGD)
    
    Returns:
        result: Dictionary with optimization results
    """
    params = np.array(initial_guess, dtype=float)
    n_params = len(params)
    
    # Store history
    cost_history = []
    param_history = []
    best_cost = float('inf')
    best_params = params.copy()
    
    # Adaptive learning rate
    initial_lr = learning_rate
    
    for iteration in range(max_iterations):
        # Calculate current cost
        current_cost = objective_function(params)
        cost_history.append(current_cost)
        param_history.append(params.copy())
        
        # Keep track of best solution
        if current_cost < best_cost:
            best_cost = current_cost
            best_params = params.copy()
        
        # Calculate gradient using finite differences
        gradient = np.zeros(n_params)
        h = 1e-8  # Smaller step for more accurate gradients
        
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += h
            params_minus = params.copy()
            params_minus[i] -= h
            
            gradient[i] = (objective_function(params_plus) - objective_function(params_minus)) / (2 * h)
        
        # Gradient clipping to prevent large updates
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 1.0:
            gradient = gradient / gradient_norm
        
        # Adaptive learning rate
        if iteration > 0:
            if cost_history[-1] > cost_history[-2]:  # Cost increased
                learning_rate *= 0.95  # Reduce learning rate
            else:
                learning_rate = min(initial_lr, learning_rate * 1.01)  # Slightly increase
        
        # Update parameters
        params_new = params - learning_rate * gradient
        
        # Apply bounds
        for i in range(n_params):
            params_new[i] = np.clip(params_new[i], bounds[i][0], bounds[i][1])
        
        # Check convergence
        param_change = np.linalg.norm(params_new - params)
        if param_change < tolerance:
            break
            
        params = params_new
        
        # Early stopping if cost is very small
        if current_cost < 1e-10:
            break
    
    # Create a result object that mimics scipy's optimization result
    class SGDResult:
        def __init__(self, x, fun, success, nit, cost_history, param_history):
            self.x = x
            self.fun = fun
            self.success = success
            self.nit = nit
            self.cost_history = cost_history
            self.param_history = param_history
    
    return SGDResult(
        x=best_params,  # Return best parameters found
        fun=best_cost,  # Return best cost
        success=True,
        nit=iteration + 1,
        cost_history=cost_history,
        param_history=param_history
    )

def optimize_user_location_and_clock_bias(aoa_measurements, bs_position, surfaces, 
                                         surface_associations, initial_guess=None, 
                                         method='L-BFGS-B', add_noise=True, 
                                         snr_db=20, num_antennas=64, true_clock_bias=25.0):
    """
    Optimization framework to solve for user location and clock bias using AOA measurements.
    
    Args:
        aoa_measurements: List of dictionaries with 'azimuth', 'elevation', 'time_delay'
        bs_position: Position of the base station
        surfaces: List of surface dictionaries
        surface_associations: List of association results from DT_localization
        initial_guess: Initial guess for [x, y, z] (optional, clock_bias will be fixed)
        method: Optimization method ('L-BFGS-B', 'SLSQP', 'differential_evolution', 
                'basinhopping', 'stochastic_gradient_descent')
        add_noise: Whether to add noise to measurements
        snr_db: Signal-to-Noise Ratio in dB
        num_antennas: Number of antennas for noise modeling
        true_clock_bias: True clock bias value (if provided, clock bias is fixed)
    
    Returns:
        optimization_result: Dictionary containing optimization results
    """
    
    # Extract measurements
    num_measurements = len(aoa_measurements)
    
    # Set initial guess if not provided
    if initial_guess is None:
        # Intelligent initial guess strategy
        np.random.seed(43)  # For reproducibility
        
        # Strategy 1: Use geometric center of the environment
        # Based on BS position and surface locations, estimate likely user area
        bs_pos = bs_position
        surface_centers = [surface['reference_point'] for surface in surfaces]
        
        # Calculate geometric center of all surfaces and BS
        all_points = [bs_pos] + surface_centers
        center_x = np.mean([p[0] for p in all_points])
        center_y = np.mean([p[1] for p in all_points])
        center_z = np.mean([p[2] for p in all_points])
        
        # Strategy 2: Use LOS measurement to estimate rough position
        if len(aoa_measurements) > 0:
            los_measurement = None
            for i, measurement in enumerate(aoa_measurements):
                if i < len(surface_associations) and surface_associations[i]['is_los']:
                    los_measurement = measurement
                    break
            
            if los_measurement is not None:
                # Rough estimate from LOS AOA (assuming user is at reasonable distance)
                azimuth_rad = np.radians(los_measurement['azimuth'])
                elevation_rad = np.radians(los_measurement['elevation'])
                
                # Assume user is at a reasonable distance (e.g., 10-20m from BS)
                estimated_distance = np.random.uniform(10, 20)
                
                # Convert AOA to rough position estimate
                rough_x = bs_pos[0] + estimated_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
                rough_y = bs_pos[1] + estimated_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
                rough_z = bs_pos[2] + estimated_distance * np.sin(elevation_rad)
                
                # Use weighted combination of geometric center and LOS estimate
                weight = 0.7  # Weight for LOS estimate
                center_x = weight * rough_x + (1 - weight) * center_x
                center_y = weight * rough_y + (1 - weight) * center_y
                center_z = weight * rough_z + (1 - weight) * center_z
        
        # Strategy 3: Add small random perturbation around the estimated center
        # Much smaller range than before
        perturbation_scale = 2.0  # meters
        
        initial_guess = np.array([
            center_x + np.random.normal(0, perturbation_scale),  # x position
            center_y + np.random.normal(0, perturbation_scale),  # y position
            max(0.5, center_z + np.random.normal(0, perturbation_scale))  # z position (ensure above ground)
        ])
        
        # Ensure within bounds
        initial_guess[0] = np.clip(initial_guess[0], -10, 20)
        initial_guess[1] = np.clip(initial_guess[1], -10, 20)
        initial_guess[2] = np.clip(initial_guess[2], 0, 15)
        
        print(f"Intelligent Initial Guess: ({initial_guess[0]:.3f}, {initial_guess[1]:.3f}, {initial_guess[2]:.3f})")
    
    # Add clock bias to initial guess
    if initial_guess is None:
        # Add random clock bias initial guess
        clock_bias_guess = np.random.uniform(10.0, 40.0)  # Reasonable range for clock bias
        initial_guess = np.append(initial_guess, clock_bias_guess)
    else:
        # If initial_guess is provided, it should include clock bias as 4th parameter
        if len(initial_guess) == 3:
            # Add random clock bias if not provided
            clock_bias_guess = np.random.uniform(10.0, 40.0)
            initial_guess = np.append(initial_guess, clock_bias_guess)
    
    print(f"Initial Guess with Clock Bias: ({initial_guess[0]:.3f}, {initial_guess[1]:.3f}, {initial_guess[2]:.3f}, {initial_guess[3]:.3f} ns)")
    
    # Define objective function
    def objective_function(params):
        """
        Objective function to minimize.
        
        Args:
            params: [x, y, z, clock_bias] - user position and clock bias
            
        Returns:
            cost: Total cost (negative log-likelihood or sum of squared errors)
        """
        user_position = params[:3]
        clock_bias = params[3]  # Extract clock bias from parameters
        
        total_cost = 0.0
        
        # Process each measurement
        for i, measurement in enumerate(aoa_measurements):
            measured_azimuth = measurement['azimuth']
            measured_elevation = measurement['elevation']
            measured_delay = measurement['time_delay']
            
            # Get surface association for this measurement
            association = surface_associations[i]
            
            # Calculate predicted AOA and delay for current user position
            if association['is_los']:  # LOS measurement
                predicted_azimuth, predicted_elevation, predicted_delay = calculate_los_measurements(
                    bs_position, user_position
                )
            else:  # NLOS measurement - use the associated surface
                if association['valid_path'] and association['associated_surface'] is not None:
                    surface_idx = association['associated_surface']
                    associated_surface = surfaces[surface_idx]
                    predicted_azimuth, predicted_elevation, predicted_delay = calculate_nlos_measurements_with_surface(
                        bs_position, user_position, associated_surface
                    )
                else:
                    # If no valid association, skip this measurement or use fallback
                    continue
            
            # Calculate residuals
            azimuth_residual = measured_azimuth - predicted_azimuth
            elevation_residual = measured_elevation - predicted_elevation
            delay_residual = measured_delay - predicted_delay - clock_bias
            
            # Add to total cost (weighted sum of squared errors)
            azimuth_weight = 1.0
            elevation_weight = 1.0
            delay_weight = 1.0
            
            cost = (azimuth_weight * azimuth_residual**2 + 
                   elevation_weight * elevation_residual**2 + 
                   delay_weight * delay_residual**2)
            
            total_cost += cost
        
        return total_cost
    
    # Define constraints (if needed)
    constraints = []
    
    # Define bounds for parameters (position and clock bias)
    bounds = [
        (-10, 20),    # x position bounds
        (-10, 20),    # y position bounds
        (0, 15),      # z position bounds (user should be above ground)
        (0, 50)       # clock bias bounds (0 to 50 ns)
    ]
    
    # Run optimization based on method
    try:
        if method == 'stochastic_gradient_descent':
            result = stochastic_gradient_descent(
                objective_function,
                initial_guess,
                bounds,
                learning_rate=0.01,
                max_iterations=5000,  # Increased from 2000
                tolerance=1e-8        # Tighter tolerance
            )
        elif method == 'differential_evolution':
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=1000,         # Back to original
                popsize=15,           # Back to original
                seed=42,
                disp=True
            )
        elif method == 'basinhopping':
            result = basinhopping(
                objective_function,
                initial_guess,
                niter=200,            # Increased from 100
                T=0.5,                # Lower temperature for more local search
                stepsize=0.3,         # Smaller step size
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxiter': 2000}},
                seed=42,
                interval=50           # Check for improvement every 50 steps
            )
        else:
            # Standard scipy minimize methods
            result = minimize(
                objective_function,
                initial_guess,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'disp': True, 'gtol': 1e-8, 'ftol': 1e-8}
            )
        
        # Extract results
        optimized_user_position = result.x[:3]
        optimized_clock_bias = result.x[3]  # Extract optimized clock bias
        final_cost = result.fun
        success = result.success
        
        # Calculate final residuals
        final_residuals = calculate_final_residuals(
            aoa_measurements, bs_position, surfaces, 
            optimized_user_position, optimized_clock_bias, surface_associations
        )
        
        optimization_result = {
            'user_position': optimized_user_position,
            'clock_bias': optimized_clock_bias,
            'final_cost': final_cost,
            'success': success,
            'iterations': result.nit,
            'final_residuals': final_residuals,
            'optimization_result': result
        }
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        optimization_result = {
            'user_position': initial_guess[:3],
            'clock_bias': initial_guess[3] if len(initial_guess) > 3 else 0.0,
            'final_cost': float('inf'),
            'success': False,
            'error': str(e)
        }
    
    return optimization_result

def calculate_los_measurements(bs_position, user_position):
    """
    Calculate LOS AOA and delay measurements.
    
    Args:
        bs_position: Position of the base station
        user_position: Position of the user
        
    Returns:
        azimuth, elevation, delay: Predicted measurements
    """
    # Vector from BS to user
    direction_vector = user_position - bs_position
    distance = np.linalg.norm(direction_vector)
    
    # Normalize direction vector
    direction_normalized = direction_vector / distance
    
    # Calculate azimuth (horizontal angle)
    azimuth = np.arctan2(direction_normalized[1], direction_normalized[0]) * 180 / np.pi
    
    # Calculate elevation (vertical angle)
    horizontal_distance = np.sqrt(direction_normalized[0]**2 + direction_normalized[1]**2)
    elevation = np.arctan2(direction_normalized[2], horizontal_distance) * 180 / np.pi
    
    # Calculate delay (speed of light = 3e8 m/s)
    c = 3e8
    delay = distance / c * 1e9  # Convert to nanoseconds
    
    return azimuth, elevation, delay

def calculate_nlos_measurements_with_surface(bs_position, user_position, surface):
    """
    Calculate NLOS AOA and delay measurements for a specific surface.
    
    Args:
        bs_position: Position of the base station
        user_position: Position of the user
        surface: Surface dictionary for the associated surface
        
    Returns:
        azimuth, elevation, delay: Predicted measurements for this surface
    """
    # Calculate reflection point for this specific surface
    reflection_point = find_reflection_point(
        user_position, bs_position, 
        surface['reference_point'], surface['normal_vector']
    )
    
    if reflection_point is not None:
        # Check if reflection point is on surface
        vertices = generate_surface_vertices(
            surface['reference_point'],
            surface['length'],
            surface['width'],
            surface['normal_vector'],
            surface['length_vector']
        )
        
        is_on_surface = check_point_on_surface(reflection_point, vertices, surface['normal_vector'],
                                surface['length'], surface['width'], surface['reference_point'],
                                surface['length_vector'])
        
        if is_on_surface:
            # Calculate AOA and delay for this reflection path
            azimuth, elevation, delay = calculate_reflection_measurements(
                bs_position, user_position, reflection_point
            )
            
            return azimuth, elevation, delay
    
    # If no valid reflection, return default values
    return 0.0, 0.0, float('inf')

def calculate_nlos_measurements(bs_position, user_position, surfaces):
    """
    Calculate NLOS AOA and delay measurements for reflection paths.
    
    Args:
        bs_position: Position of the base station
        user_position: Position of the user
        surfaces: List of surface dictionaries
        
    Returns:
        azimuth, elevation, delay: Predicted measurements (best reflection path)
    """
    # Find the best reflection path
    best_azimuth = 0.0
    best_elevation = 0.0
    best_delay = float('inf')
    
    for surface in surfaces:
        # Calculate reflection point
        reflection_point = find_reflection_point(
            user_position, bs_position, 
            surface['reference_point'], surface['normal_vector']
        )
        
        if reflection_point is not None:
            # Check if reflection point is on surface
            vertices = generate_surface_vertices(
                surface['reference_point'],
                surface['length'],
                surface['width'],
                surface['normal_vector'],
                surface['length_vector']
            )
            
            if check_point_on_surface(reflection_point, vertices, surface['normal_vector'],
                                    surface['length'], surface['width'], surface['reference_point'],
                                    surface['length_vector']):
                
                # Calculate AOA and delay for this reflection path
                azimuth, elevation, delay = calculate_reflection_measurements(
                    bs_position, user_position, reflection_point
                )
                
                # Use the shortest delay path (strongest signal)
                if delay < best_delay:
                    best_azimuth = azimuth
                    best_elevation = elevation
                    best_delay = delay
    
    return best_azimuth, best_elevation, best_delay

def calculate_reflection_measurements(bs_position, user_position, reflection_point):
    """
    Calculate AOA and delay for a reflection path.
    
    Args:
        bs_position: Position of the base station
        user_position: Position of the user
        reflection_point: Point of reflection on surface
        
    Returns:
        azimuth, elevation, delay: Predicted measurements
    """
    # Vector from reflection point to BS (for uplink signal)
    direction_vector = bs_position - reflection_point
    distance = np.linalg.norm(direction_vector)
    
    # Normalize direction vector
    direction_normalized = direction_vector / distance
    
    # Calculate azimuth (horizontal angle)
    azimuth = np.arctan2(direction_normalized[1], direction_normalized[0]) * 180 / np.pi
    
    # Calculate elevation (vertical angle)
    horizontal_distance = np.sqrt(direction_normalized[0]**2 + direction_normalized[1]**2)
    elevation = np.arctan2(direction_normalized[2], horizontal_distance) * 180 / np.pi
    
    # Calculate total path length: user -> reflection -> BS
    incident_path = reflection_point - user_position
    reflected_path = bs_position - reflection_point
    total_distance = np.linalg.norm(incident_path) + np.linalg.norm(reflected_path)
    
    # Calculate delay
    c = 3e8
    delay = total_distance / c * 1e9  # Convert to nanoseconds
    
    return azimuth, elevation, delay

def calculate_final_residuals(aoa_measurements, bs_position, surfaces, 
                            user_position, clock_bias, surface_associations):
    """
    Calculate final residuals for all measurements.
    
    Args:
        aoa_measurements: List of measurements
        bs_position: Position of the base station
        surfaces: List of surface dictionaries
        user_position: Optimized user position
        clock_bias: Optimized clock bias
        surface_associations: List of association results from DT_localization
        
    Returns:
        residuals: Dictionary containing residuals for each measurement
    """
    residuals = []
    
    for i, measurement in enumerate(aoa_measurements):
        measured_azimuth = measurement['azimuth']
        measured_elevation = measurement['elevation']
        measured_delay = measurement['time_delay']
        
        # Get surface association for this measurement
        association = surface_associations[i]
        
        # Calculate predicted measurements
        if association['is_los']:  # LOS
            predicted_azimuth, predicted_elevation, predicted_delay = calculate_los_measurements(
                bs_position, user_position
            )
        else:  # NLOS - use the associated surface
            if association['valid_path'] and association['associated_surface'] is not None:
                surface_idx = association['associated_surface']
                associated_surface = surfaces[surface_idx]
                predicted_azimuth, predicted_elevation, predicted_delay = calculate_nlos_measurements_with_surface(
                    bs_position, user_position, associated_surface
                )
            else:
                # If no valid association, skip this measurement
                continue
        
        # Calculate residuals
        azimuth_residual = measured_azimuth - predicted_azimuth
        elevation_residual = measured_elevation - predicted_elevation
        delay_residual = measured_delay - predicted_delay - clock_bias
        
        residuals.append({
            'measurement_index': i,
            'azimuth_residual': azimuth_residual,
            'elevation_residual': elevation_residual,
            'delay_residual': delay_residual,
            'predicted_azimuth': predicted_azimuth,
            'predicted_elevation': predicted_elevation,
            'predicted_delay': predicted_delay
        })
    
    return residuals

def test_multiple_optimization_methods(aoa_measurements, bs_position, surfaces, associations, true_user_position, true_clock_bias):
    """
    Test multiple optimization methods and compare their performance.
    
    Args:
        aoa_measurements: List of AOA measurements
        bs_position: Base station position
        surfaces: List of surfaces
        associations: Surface associations
        true_user_position: True user position
        true_clock_bias: True clock bias (for comparison with optimized result)
    
    Returns:
        results: Dictionary with results from all methods
    """
    methods = [
        'L-BFGS-B',
        'SLSQP', 
        'differential_evolution',
        'basinhopping',
        'stochastic_gradient_descent'
    ]
    
    results = {}
    
    print("=== Testing Multiple Optimization Methods ===")
    
    for method in methods:
        print(f"\n--- Testing {method} ---")
        try:
            result = optimize_user_location_and_clock_bias(
                aoa_measurements,
                bs_position,
                surfaces,
                associations,
                initial_guess=None,  # Use random initial guess
                method=method
            )
            
            # Calculate errors
            position_error = np.linalg.norm(result['user_position'] - true_user_position)
            clock_bias_error = abs(result['clock_bias'] - true_clock_bias)
            
            results[method] = {
                'result': result,
                'position_error': position_error,
                'clock_bias_error': clock_bias_error,
                'success': result['success'],
                'final_cost': result['final_cost'],
                'iterations': result['iterations']
            }
            
            print(f"  Success: {result['success']}")
            print(f"  Optimized Position: ({result['user_position'][0]:.3f}, {result['user_position'][1]:.3f}, {result['user_position'][2]:.3f})")
            print(f"  Optimized Clock Bias: {result['clock_bias']:.3f} ns")
            print(f"  Position Error: {position_error:.3f} m")
            print(f"  Clock Bias Error: {clock_bias_error:.3f} ns")
            print(f"  Final Cost: {result['final_cost']:.6f}")
            print(f"  Iterations: {result['iterations']}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[method] = {'error': str(e)}
    
    # Find best method
    successful_methods = {k: v for k, v in results.items() if 'error' not in v and v['success']}
    if successful_methods:
        best_method = min(successful_methods.keys(), 
                         key=lambda x: successful_methods[x]['position_error'])
        print(f"\n=== Best Method: {best_method} ===")
        print(f"True Position: ({true_user_position[0]:.3f}, {true_user_position[1]:.3f}, {true_user_position[2]:.3f})")
        print(f"Optimized Position: ({successful_methods[best_method]['result']['user_position'][0]:.3f}, {successful_methods[best_method]['result']['user_position'][1]:.3f}, {successful_methods[best_method]['result']['user_position'][2]:.3f})")
        print(f"Position Error: {successful_methods[best_method]['position_error']:.3f} m")
        print(f"Clock Bias Error: {successful_methods[best_method]['clock_bias_error']:.3f} ns")
    
    return results

def run_optimization_example():
    """
    Example function to demonstrate the optimization framework.
    """
    print("=== DT Optimization Example ===")
    
    # Run DT localization to get measurements
    print("Running DT localization to generate measurements...")
    results = analyze_dt_localization(add_noise=True, snr_db=20)
    
    # Extract AOA measurements
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
    
    print(f"Generated {len(aoa_measurements)} measurements:")
    for i, measurement in enumerate(aoa_measurements):
        print(f"  Measurement {i+1}: AOA=({measurement['azimuth']:.1f}°, {measurement['elevation']:.1f}°), Delay={measurement['time_delay']:.2f} ns")
    
    # Get surface associations from DT localization
    print("\nGetting surface associations...")
    associations = associate_aoa_with_paths(
        aoa_measurements, 
        results['bs_position'], 
        results['surfaces'],
        K=1000, 
        threshold=0.8,
        snr_db=20,
        num_antennas=64
    )
    
    # Display associations
    print("\n=== Surface Associations ===")
    for i, association in enumerate(associations):
        if association['is_los']:
            print(f"Measurement {i+1}: LOS (no surface association)")
        else:
            if association['valid_path']:
                surface_idx = association['associated_surface']
                surface_label = results['surfaces'][surface_idx]['label']
                probability = association['surface_associations'][surface_idx]['probability']
                print(f"Measurement {i+1}: Associated with {surface_label} (Probability: {probability:.3f})")
            else:
                print(f"Measurement {i+1}: No valid surface association")
    
    # Run optimization with random initial guess (including clock bias)
    print("\nRunning optimization with random initial guess (including clock bias)...")
    optimization_result = optimize_user_location_and_clock_bias(
        aoa_measurements,
        results['bs_position'],
        results['surfaces'],
        associations,  # Pass the surface associations
        initial_guess=None  # Will use random initial guess (including clock bias)
    )
    
    # Display results
    print("\n=== Optimization Results ===")
    print(f"Success: {optimization_result['success']}")
    print(f"Final Cost: {optimization_result['final_cost']:.6f}")
    print(f"Iterations: {optimization_result['iterations']}")
    
    # Show the random initial guess that was used
    if 'optimization_result' in optimization_result and hasattr(optimization_result['optimization_result'], 'x0'):
        initial_guess = optimization_result['optimization_result'].x0
        print(f"\nRandom Initial Guess:")
        print(f"  Initial Position: ({initial_guess[0]:.3f}, {initial_guess[1]:.3f}, {initial_guess[2]:.3f}) m")
        print(f"  Initial Clock Bias: {initial_guess[3]:.3f} ns")
    
    print(f"\nOptimized User Position: ({optimization_result['user_position'][0]:.3f}, "
          f"{optimization_result['user_position'][1]:.3f}, {optimization_result['user_position'][2]:.3f}) m")
    print(f"Optimized Clock Bias: {optimization_result['clock_bias']:.3f} ns")
    
    print(f"\nTrue User Position: ({results['user_position'][0]:.3f}, "
          f"{results['user_position'][1]:.3f}, {results['user_position'][2]:.3f}) m")
    print(f"True Clock Bias: {results['clock_bias']:.3f} ns")
    
    # Calculate position error
    position_error = np.linalg.norm(optimization_result['user_position'] - results['user_position'])
    print(f"Position Error: {position_error:.3f} m")
    
    # Calculate clock bias error
    clock_bias_error = abs(optimization_result['clock_bias'] - results['clock_bias'])
    print(f"Clock Bias Error: {clock_bias_error:.3f} ns")
    
    # Show reflection point information for initial guess
    print(f"\n{'='*60}")
    print("=== REFLECTION POINT ANALYSIS FOR INITIAL GUESS ===")
    print("="*60)
    
    # Get the initial guess that will be used
    np.random.seed(43)  # For reproducibility
    
    # Strategy 1: Use geometric center of the environment
    bs_pos = results['bs_position']
    surface_centers = [surface['reference_point'] for surface in results['surfaces']]
    
    # Calculate geometric center of all surfaces and BS
    all_points = [bs_pos] + surface_centers
    center_x = np.mean([p[0] for p in all_points])
    center_y = np.mean([p[1] for p in all_points])
    center_z = np.mean([p[2] for p in all_points])
    
    # Strategy 2: Use LOS measurement to estimate rough position
    if len(aoa_measurements) > 0:
        los_measurement = None
        for i, measurement in enumerate(aoa_measurements):
            if i < len(associations) and associations[i]['is_los']:
                los_measurement = measurement
                break
        
        if los_measurement is not None:
            # Rough estimate from LOS AOA (assuming user is at reasonable distance)
            azimuth_rad = np.radians(los_measurement['azimuth'])
            elevation_rad = np.radians(los_measurement['elevation'])
            
            # Assume user is at a reasonable distance (e.g., 10-20m from BS)
            estimated_distance = np.random.uniform(10, 20)
            
            # Convert AOA to rough position estimate
            rough_x = bs_pos[0] + estimated_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            rough_y = bs_pos[1] + estimated_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            rough_z = bs_pos[2] + estimated_distance * np.sin(elevation_rad)
            
            # Use weighted combination of geometric center and LOS estimate
            weight = 0.7  # Weight for LOS estimate
            center_x = weight * rough_x + (1 - weight) * center_x
            center_y = weight * rough_y + (1 - weight) * center_y
            center_z = weight * rough_z + (1 - weight) * center_z
    
    # Strategy 3: Add small random perturbation around the estimated center
    perturbation_scale = 2.0  # meters
    
    initial_guess = np.array([
        center_x + np.random.normal(0, perturbation_scale),  # x position
        center_y + np.random.normal(0, perturbation_scale),  # y position
        max(0.5, center_z + np.random.normal(0, perturbation_scale))  # z position (ensure above ground)
    ])
    
    # Ensure within bounds
    initial_guess[0] = np.clip(initial_guess[0], -10, 20)
    initial_guess[1] = np.clip(initial_guess[1], -10, 20)
    initial_guess[2] = np.clip(initial_guess[2], 0, 15)
    
    print(f"Initial Guess Position: ({initial_guess[0]:.3f}, {initial_guess[1]:.3f}, {initial_guess[2]:.3f})")
    print()
    
    # Check reflection points for each surface
    for i, surface in enumerate(results['surfaces']):
        print(f"Surface {i+1} ({surface['label']}):")
        
        # Calculate reflection point
        reflection_point = find_reflection_point(
            initial_guess, results['bs_position'], 
            surface['reference_point'], surface['normal_vector']
        )
        
        if reflection_point is not None:
            # Check if reflection point is on surface
            vertices = generate_surface_vertices(
                surface['reference_point'],
                surface['length'],
                surface['width'],
                surface['normal_vector'],
                surface['length_vector']
            )
            
            is_on_surface = check_point_on_surface(reflection_point, vertices, surface['normal_vector'],
                                    surface['length'], surface['width'], surface['reference_point'],
                                    surface['length_vector'])
            
            print(f"  Reflection Point: ({reflection_point[0]:.3f}, {reflection_point[1]:.3f}, {reflection_point[2]:.3f})")
            print(f"  Surface Reference: ({surface['reference_point'][0]:.3f}, {surface['reference_point'][1]:.3f}, {surface['reference_point'][2]:.3f})")
            print(f"  Surface Dimensions: {surface['length']}m × {surface['width']}m")
            print(f"  Point on Surface: {is_on_surface}")
        else:
            print(f"  No valid reflection point found")
        print()
    
    # Calculate predicted measurements for initial guess
    print("=== PREDICTED MEASUREMENTS FOR INITIAL GUESS ===")
    print()
    
    total_initial_cost = 0.0
    for i, measurement in enumerate(aoa_measurements):
        measured_azimuth = measurement['azimuth']
        measured_elevation = measurement['elevation']
        measured_delay = measurement['time_delay']
        
        # Get surface association for this measurement
        association = associations[i]
        
        # Calculate predicted AOA and delay using initial guess
        if association['is_los']:  # LOS measurement
            predicted_azimuth, predicted_elevation, predicted_delay = calculate_los_measurements(
                results['bs_position'], initial_guess
            )
            path_type = "LOS"
        else:  # NLOS measurement - use the associated surface
            if association['valid_path'] and association['associated_surface'] is not None:
                surface_idx = association['associated_surface']
                associated_surface = results['surfaces'][surface_idx]
                predicted_azimuth, predicted_elevation, predicted_delay = calculate_nlos_measurements_with_surface(
                    results['bs_position'], initial_guess, associated_surface
                )
                path_type = f"NLOS (Surface {surface_idx+1})"
            else:
                continue
        
        # Calculate residuals
        azimuth_residual = measured_azimuth - predicted_azimuth
        elevation_residual = measured_elevation - predicted_elevation
        delay_residual = measured_delay - predicted_delay - results['clock_bias']
        
        # Calculate cost for this measurement
        cost = azimuth_residual**2 + elevation_residual**2 + delay_residual**2
        total_initial_cost += cost
        
        print(f"Measurement {i+1} ({path_type}):")
        print(f"  Measured:    Azimuth={measured_azimuth:8.3f}°, Elevation={measured_elevation:8.3f}°, Delay={measured_delay:8.3f}ns")
        print(f"  Predicted:   Azimuth={predicted_azimuth:8.3f}°, Elevation={predicted_elevation:8.3f}°, Delay={predicted_delay:8.3f}ns")
        print(f"  Residuals:   Azimuth={azimuth_residual:8.3f}°, Elevation={elevation_residual:8.3f}°, Delay={delay_residual:8.3f}ns")
        print()
    
    print(f"Initial Cost: {total_initial_cost:.6f}")
    print("="*60)
    
    # Test multiple optimization methods
    print(f"\n{'='*60}")
    method_results = test_multiple_optimization_methods(
        aoa_measurements, results['bs_position'], results['surfaces'], associations,
        results['user_position'], results['clock_bias']
    )
    
    # Display residuals for the best method
    best_method = None
    if method_results:
        successful_methods = {k: v for k, v in method_results.items() if 'error' not in v and v['success']}
        if successful_methods:
            best_method = min(successful_methods.keys(), 
                             key=lambda x: successful_methods[x]['position_error'])
            best_result = method_results[best_method]['result']
            
            print(f"\n=== Final Residuals (Best Method: {best_method}) ===")
            for residual in best_result['final_residuals']:
                print(f"Measurement {residual['measurement_index']+1}:")
                print(f"  Azimuth Residual: {residual['azimuth_residual']:.3f}°")
                print(f"  Elevation Residual: {residual['elevation_residual']:.3f}°")
                print(f"  Delay Residual: {residual['delay_residual']:.3f} ns")
    
    return optimization_result, results, method_results

if __name__ == "__main__":
    # Run the optimization example
    optimization_result, dt_results, method_results = run_optimization_example() 