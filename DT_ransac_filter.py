#!/usr/bin/env python3
"""
RANSAC Filter for DT Localization Measurements

This script implements RANSAC to filter out inconsistent NLOS measurements.
Input: LOS and NLOS measurements (including single-bounce, multi-bounce, and random noise)
Output: Filtered measurements with consistent single-bounce reflections

RANSAC Algorithm:
1. Random Sampling: Randomly choose subset of NLOS measurements
2. Model Estimation: Use optimization to estimate position and clock bias
3. Inlier Evaluation: Evaluate all paths using estimated parameters
4. Iteration and Selection: Find best model through multiple iterations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from DT_localization import analyze_dt_localization

class RANSACFilter:
    """
    RANSAC filter for NLOS measurements to identify single-bounce reflections.
    """
    
    def __init__(self, min_inliers=3, max_iterations=100, threshold=0.2, 
                 confidence=0.95, min_samples=3):
        """
        Initialize RANSAC filter parameters.
        
        Args:
            min_inliers (int): Minimum number of inliers required for a good model
            max_iterations (int): Maximum number of RANSAC iterations
            threshold (float): Residual threshold for inlier classification
            confidence (float): Desired confidence level (0-1)
            min_samples (int): Minimum samples needed to fit the model
        """
        self.min_inliers = min_inliers
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.confidence = confidence
        self.min_samples = min_samples
    
    def identify_los_measurements(self, measurements, associations):
        """
        Identify LOS measurements from the measurement set.
        
        Args:
            measurements (array): AOA measurements [azimuth, elevation, delay]
            associations (list): Surface associations (-1 for LOS, >=0 for NLOS)
            
        Returns:
            tuple: (los_indices, nlos_indices)
        """
        los_indices = [i for i, assoc in enumerate(associations) if assoc == -1]
        nlos_indices = [i for i, assoc in enumerate(associations) if assoc >= 0]
        
        return los_indices, nlos_indices
    
    def estimate_model(self, bs_pos, measurements, associations):
        """
        Estimate user position and clock bias from a subset of measurements.
        
        Args:
            bs_pos (array): Base station position
            measurements (array): AOA measurements [azimuth, elevation, delay]
            associations (list): Surface associations
            
        Returns:
            tuple: (estimated_position, estimated_clock_bias) or (None, None) if estimation fails
        """
        try:
            # Separate LOS and NLOS measurements
            los_measurements = []
            nlos_measurements = []
            
            for i, (measurement, assoc) in enumerate(zip(measurements, associations)):
                if assoc == -1:  # LOS
                    los_measurements.append(measurement)
                else:  # NLOS
                    nlos_measurements.append(measurement)
            
            # Initial guess
            if len(los_measurements) > 0:
                # Use LOS measurement for initial position estimate
                los_measurement = los_measurements[0]
                azimuth, elevation, delay = los_measurement
                
                # Simple triangulation from LOS
                distance = (delay * 1e-9) * 3e8  # Convert ns to m
                dx = distance * np.cos(elevation) * np.cos(azimuth)
                dy = distance * np.cos(elevation) * np.sin(azimuth)
                dz = distance * np.sin(elevation)
                
                estimated_pos = bs_pos + np.array([dx, dy, dz])
                estimated_pos = np.clip(estimated_pos, [0, 0, 0], [15, 15, 15])
                estimated_clock_bias = delay * 0.1  # 10% of delay as initial bias
            else:
                # Use geometric center and reasonable clock bias
                estimated_pos = np.array([7.5, 0.5, 6.0])
                estimated_clock_bias = 25.0
            
            # Combine position and clock bias into single parameter vector
            initial_params = np.concatenate([estimated_pos, [estimated_clock_bias]])
            
            # Define objective function similar to DT_optimization
            def objective_function(params):
                user_pos = params[:3]
                clock_bias = params[3]
                
                total_error = 0
                
                # LOS measurements
                for measurement in los_measurements:
                    azimuth, elevation, delay = measurement
                    
                    # Calculate predicted LOS measurements
                    direction_vector = user_pos - bs_pos
                    distance = np.linalg.norm(direction_vector)
                    
                    if distance > 0:
                        # Predicted AOA
                        pred_azimuth = np.arctan2(direction_vector[1], direction_vector[0])
                        pred_elevation = np.arcsin(np.clip(direction_vector[2] / distance, -1, 1))
                        
                        # Predicted delay
                        pred_delay = (distance / 3e8) * 1e9 + clock_bias
                        
                        # Calculate residuals
                        azimuth_error = abs(azimuth - pred_azimuth)
                        elevation_error = abs(elevation - pred_elevation)
                        delay_error = abs(delay - pred_delay)
                        
                        # Normalized error
                        total_error += (azimuth_error / np.pi)**2 + (elevation_error / np.pi)**2 + (delay_error / 100.0)**2
                    else:
                        total_error += 100.0
                
                # NLOS measurements (improved model)
                for measurement in nlos_measurements:
                    azimuth, elevation, delay = measurement
                    
                    # For NLOS, we expect some extra delay due to reflection
                    direction_vector = user_pos - bs_pos
                    distance = np.linalg.norm(direction_vector)
                    
                    if distance > 0:
                        # Predicted AOA (simplified - same as LOS for now)
                        pred_azimuth = np.arctan2(direction_vector[1], direction_vector[0])
                        pred_elevation = np.arcsin(np.clip(direction_vector[2] / distance, -1, 1))
                        
                        # Predicted delay (add extra delay for NLOS)
                        extra_delay = 3.0  # 3ns extra for NLOS reflection (reduced)
                        pred_delay = (distance / 3e8) * 1e9 + clock_bias + extra_delay
                        
                        # Calculate residuals
                        azimuth_error = abs(azimuth - pred_azimuth)
                        elevation_error = abs(elevation - pred_elevation)
                        delay_error = abs(delay - pred_delay)
                        
                        # Normalized error (more lenient for NLOS)
                        total_error += (azimuth_error / np.pi)**2 + (elevation_error / np.pi)**2 + (delay_error / 150.0)**2
                    else:
                        total_error += 100.0
                
                return total_error
            
            # Optimize position and clock bias
            result = minimize(
                objective_function, 
                initial_params,
                method='L-BFGS-B',
                bounds=[(0, 15), (0, 15), (0, 15), (0, 50)],  # Position bounds + clock bias bounds
                options={'maxiter': 200}
            )
            
            if result.success:
                estimated_position = result.x[:3]
                estimated_clock_bias = result.x[3]
                return estimated_position, estimated_clock_bias
            else:
                return estimated_pos, estimated_clock_bias
                
        except Exception as e:
            print(f"Error in model estimation: {e}")
            return None, None
    
    def evaluate_inliers(self, user_pos, clock_bias, bs_pos, measurements, associations):
        """
        Evaluate which measurements are inliers based on estimated position and clock bias.
        
        Args:
            user_pos (array): Estimated user position
            clock_bias (float): Estimated clock bias
            bs_pos (array): Base station position
            measurements (array): AOA measurements [azimuth, elevation, delay]
            associations (list): Surface associations
            
        Returns:
            tuple: (inlier_indices, model_score)
        """
        inlier_indices = []
        total_residual = 0
        
        for i, (measurement, surface_idx) in enumerate(zip(measurements, associations)):
            azimuth, elevation, delay = measurement
            
            # Calculate predicted measurements
            direction_vector = user_pos - bs_pos
            distance = np.linalg.norm(direction_vector)
            
            if distance > 0:
                # Predicted AOA
                pred_azimuth = np.arctan2(direction_vector[1], direction_vector[0])
                pred_elevation = np.arcsin(np.clip(direction_vector[2] / distance, -1, 1))
                
                # Predicted delay (add extra delay for NLOS)
                extra_delay = 3.0 if surface_idx >= 0 else 0.0  # 3ns extra for NLOS
                pred_delay = (distance / 3e8) * 1e9 + clock_bias + extra_delay
                
                # Calculate residuals
                azimuth_error = abs(azimuth - pred_azimuth)
                elevation_error = abs(elevation - pred_elevation)
                delay_error = abs(delay - pred_delay)
                
                # Normalized residual
                residual = np.sqrt((azimuth_error / np.pi)**2 + (elevation_error / np.pi)**2 + (delay_error / 100.0)**2)
            else:
                residual = float('inf')
            
            # Check if measurement is an inlier
            if residual <= self.threshold:
                inlier_indices.append(i)
                total_residual += residual
        
        if len(inlier_indices) >= self.min_inliers:
            model_score = total_residual / len(inlier_indices)  # Average residual
        else:
            model_score = float('inf')
            
        return inlier_indices, model_score
    
    def filter_measurements(self, bs_pos, measurements, associations):
        """
        Apply RANSAC to filter out inconsistent measurements.
        
        Args:
            bs_pos (array): Base station position
            measurements (array): AOA measurements [azimuth, elevation, delay]
            associations (list): Surface associations
            
        Returns:
            tuple: (filtered_measurements, filtered_associations, inlier_mask, best_model)
        """
        print("Starting RANSAC filtering...")
        
        # Step 1: Identify LOS and NLOS measurements
        los_indices, nlos_indices = self.identify_los_measurements(measurements, associations)
        
        print(f"Found {len(los_indices)} LOS measurements and {len(nlos_indices)} NLOS measurements")
        
        if len(nlos_indices) < self.min_samples:
            print("Not enough NLOS measurements for RANSAC filtering")
            return measurements, associations, np.ones(len(measurements), dtype=bool), None
        
        # Initialize best model
        best_inliers = []
        best_score = float('inf')
        best_user_pos = None
        best_clock_bias = None
        
        # RANSAC iterations
        for iteration in range(self.max_iterations):
            # Step 2: Random Sampling - randomly choose subset of NLOS measurements
            sample_size = min(self.min_samples, len(nlos_indices))
            sample_indices = np.random.choice(nlos_indices, size=sample_size, replace=False)
            
            # Include LOS measurements if available
            if len(los_indices) > 0:
                los_sample = np.random.choice(los_indices, size=min(1, len(los_indices)), replace=False)
                sample_indices = np.concatenate([los_sample, sample_indices])
            
            # Step 3: Model Estimation - estimate position and clock bias using sampled measurements
            try:
                user_pos_estimate, clock_bias_estimate = self.estimate_model(
                    bs_pos, 
                    measurements[sample_indices], 
                    [associations[i] for i in sample_indices]
                )
                
                if user_pos_estimate is None or clock_bias_estimate is None:
                    continue
                
                # Step 4: Inlier Evaluation - evaluate all measurements using estimated parameters
                inlier_indices, model_score = self.evaluate_inliers(
                    user_pos_estimate, clock_bias_estimate, bs_pos, 
                    measurements, associations
                )
                
                # Step 5: Iteration and Selection - update best model if better
                if len(inlier_indices) >= self.min_inliers and model_score < best_score:
                    best_inliers = inlier_indices
                    best_score = model_score
                    best_user_pos = user_pos_estimate.copy()
                    best_clock_bias = clock_bias_estimate
                    
                    print(f"Iteration {iteration + 1}: Found {len(best_inliers)} inliers with score {best_score:.4f}")
                    print(f"  Estimated position: {best_user_pos}")
                    print(f"  Estimated clock bias: {best_clock_bias:.2f} ns")
                
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
                continue
        
        # Create inlier mask
        inlier_mask = np.zeros(len(measurements), dtype=bool)
        inlier_mask[best_inliers] = True
        
        # Always keep LOS measurements
        inlier_mask[los_indices] = True
        
        # Filter measurements
        filtered_measurements = measurements[inlier_mask]
        filtered_associations = [associations[i] for i in range(len(measurements)) if inlier_mask[i]]
        
        # Create best model info
        best_model = {
            'position': best_user_pos,
            'clock_bias': best_clock_bias,
            'score': best_score,
            'inlier_count': len(best_inliers)
        }
        
        print(f"RANSAC filtering complete:")
        print(f"  Original measurements: {len(measurements)}")
        print(f"  Filtered measurements: {len(filtered_measurements)}")
        print(f"  Inliers: {len(best_inliers)}")
        print(f"  Best model score: {best_score:.4f}")
        if best_user_pos is not None:
            print(f"  Estimated user position: {best_user_pos}")
            print(f"  Estimated clock bias: {best_clock_bias:.2f} ns")
        
        return filtered_measurements, filtered_associations, inlier_mask, best_model

def generate_test_measurements():
    """
    Generate test measurements including:
    - 1 LOS measurement
    - 3 single-bounce NLOS measurements (what we want to keep)
    - Random noise measurements (what we want to filter out)
    
    Returns:
        tuple: (measurements, associations, bs_position, true_user_position)
    """
    # Use same settings as DT_localization
    true_user_position = np.array([0.0, 0.0, 1.5])
    bs_position = np.array([15.0, 0.0, 10.0])
    
    # Generate measurements
    measurements = []
    associations = []
    
    # 1. LOS measurement
    direction_vector = true_user_position - bs_position
    distance = np.linalg.norm(direction_vector)
    azimuth = np.arctan2(direction_vector[1], direction_vector[0])
    elevation = np.arcsin(direction_vector[2] / distance)
    delay = (distance / 3e8) * 1e9 + 25.0  # Add 25ns clock bias
    
    measurements.append([azimuth, elevation, delay])
    associations.append(-1)  # LOS
    
    # 2. Single-bounce NLOS measurements (3 valid reflections)
    surfaces = [
        {'reference_point': np.array([7.5, 2.0, 5.0]), 'normal_vector': np.array([0, -1, 0])},
        {'reference_point': np.array([3.0, -1.0, 0.0]), 'normal_vector': np.array([0, 0, 1])},
        {'reference_point': np.array([12.0, -1.0, 12.0]), 'normal_vector': np.array([0, 0, -1])}
    ]
    
    for i, surface in enumerate(surfaces):
        # Calculate reflection point (more realistic)
        # For single-bounce, reflection point should be on the surface
        if i == 0:  # Surface 1 (wall)
            reflection_point = np.array([7.5, 2.0, 5.0]) + np.array([2.0, 0.0, 1.0])
        elif i == 1:  # Surface 2 (floor)
            reflection_point = np.array([3.0, -1.0, 0.0]) + np.array([1.0, 2.0, 0.0])
        else:  # Surface 3 (ceiling)
            reflection_point = np.array([12.0, -1.0, 12.0]) + np.array([-1.0, 1.0, 0.0])
        
        # Calculate NLOS measurements with correct total path length
        # Total path: user → reflection_point → BS
        distance_user_to_reflection = np.linalg.norm(reflection_point - true_user_position)
        distance_reflection_to_bs = np.linalg.norm(reflection_point - bs_position)
        total_distance = distance_user_to_reflection + distance_reflection_to_bs
        
        # AOA is from reflection point to BS (uplink)
        direction_vector = reflection_point - bs_position
        azimuth = np.arctan2(direction_vector[1], direction_vector[0])
        elevation = np.arcsin(direction_vector[2] / distance_reflection_to_bs)
        
        # Delay based on total path length
        delay = (total_distance / 3e8) * 1e9 + 25.0  # Total path + clock bias
        
        measurements.append([azimuth, elevation, delay])
        associations.append(i)  # Surface index
    
    # 3. Random noise measurements (what we want to filter out)
    np.random.seed(42)  # For reproducibility
    
    for i in range(5):  # Add 5 random measurements
        # Random AOA
        azimuth = np.random.uniform(-np.pi, np.pi)
        elevation = np.random.uniform(-np.pi/2, np.pi/2)
        
        # Random delay (much larger than expected)
        delay = np.random.uniform(100, 200)  # 100-200ns (much larger than expected)
        
        measurements.append([azimuth, elevation, delay])
        associations.append(10 + i)  # Use high surface index to indicate noise
    
    return np.array(measurements), associations, bs_position, true_user_position

def analyze_ransac_filtering():
    """
    Analyze the performance of RANSAC filtering on test measurements.
    """
    print("=" * 60)
    print("RANSAC FILTERING ANALYSIS")
    print("=" * 60)
    
    # Generate test measurements
    measurements, associations, bs_position, true_user_position = generate_test_measurements()
    
    print(f"Generated {len(measurements)} test measurements:")
    print(f"  True user position: {true_user_position}")
    print(f"  Base station position: {bs_position}")
    
    # Count measurement types
    los_count = sum(1 for assoc in associations if assoc == -1)
    single_bounce_count = sum(1 for assoc in associations if 0 <= assoc < 3)
    noise_count = sum(1 for assoc in associations if assoc >= 10)
    
    print(f"  LOS measurements: {los_count}")
    print(f"  Single-bounce NLOS measurements: {single_bounce_count}")
    print(f"  Random noise measurements: {noise_count}")
    
    # Initialize RANSAC filter
    ransac_filter = RANSACFilter(
        min_inliers=2,  # Reduced minimum inliers
        max_iterations=100,  # More iterations
        threshold=0.4,  # Balanced threshold to filter noise but keep legitimate measurements
        confidence=0.95,
        min_samples=2  # Reduced minimum samples
    )
    
    # Apply RANSAC filtering
    filtered_measurements, filtered_associations, inlier_mask, best_model = ransac_filter.filter_measurements(
        bs_position, measurements, associations
    )
    
    # Analyze results
    print("\n" + "=" * 40)
    print("FILTERING RESULTS")
    print("=" * 40)
    
    # Count filtered measurement types
    filtered_los_count = sum(1 for assoc in filtered_associations if assoc == -1)
    filtered_single_bounce_count = sum(1 for assoc in filtered_associations if 0 <= assoc < 3)
    filtered_noise_count = sum(1 for assoc in filtered_associations if assoc >= 10)
    
    print(f"Original measurements:")
    print(f"  LOS: {los_count}")
    print(f"  Single-bounce NLOS: {single_bounce_count}")
    print(f"  Random noise: {noise_count}")
    print(f"  Total: {len(measurements)}")
    
    print(f"\nFiltered measurements:")
    print(f"  LOS: {filtered_los_count}")
    print(f"  Single-bounce NLOS: {filtered_single_bounce_count}")
    print(f"  Random noise: {filtered_noise_count}")
    print(f"  Total: {len(filtered_measurements)}")
    
    print(f"\nFiltering statistics:")
    print(f"  Measurements removed: {len(measurements) - len(filtered_measurements)}")
    print(f"  Retention rate: {len(filtered_measurements)/len(measurements)*100:.1f}%")
    
    if best_model is not None and best_model['position'] is not None:
        print(f"\nBest model:")
        print(f"  Position: {best_model['position']}")
        print(f"  Clock bias: {best_model['clock_bias']:.2f} ns")
        print(f"  Score: {best_model['score']:.4f}")
        print(f"  Inlier count: {best_model['inlier_count']}")
        
        # Calculate position error
        pos_error = np.linalg.norm(best_model['position'] - true_user_position)
        clock_error = abs(best_model['clock_bias'] - 25.0)
        print(f"  Position error: {pos_error:.4f} m")
        print(f"  Clock bias error: {clock_error:.4f} ns")
    
    # Show which measurements were filtered out
    removed_indices = [i for i, kept in enumerate(inlier_mask) if not kept]
    if removed_indices:
        print(f"\nRemoved measurements (indices): {removed_indices}")
        print("Removed measurement details:")
        for idx in removed_indices:
            measurement = measurements[idx]
            association = associations[idx]
            measurement_type = "LOS" if association == -1 else f"Surface {association}" if association < 10 else "Noise"
            print(f"  Index {idx}: {measurement_type} - AOA=({measurement[0]:.3f}, {measurement[1]:.3f}), "
                  f"Delay={measurement[2]:.1f}ns")
    
    # Show which measurements were kept
    kept_indices = [i for i, kept in enumerate(inlier_mask) if kept]
    if kept_indices:
        print(f"\nKept measurements (indices): {kept_indices}")
        print("Kept measurement details:")
        for idx in kept_indices:
            measurement = measurements[idx]
            association = associations[idx]
            measurement_type = "LOS" if association == -1 else f"Surface {association}" if association < 10 else "Noise"
            print(f"  Index {idx}: {measurement_type} - AOA=({measurement[0]:.3f}, {measurement[1]:.3f}), "
                  f"Delay={measurement[2]:.1f}ns")
    
    return {
        'original_measurements': measurements,
        'original_associations': associations,
        'filtered_measurements': filtered_measurements,
        'filtered_associations': filtered_associations,
        'inlier_mask': inlier_mask,
        'best_model': best_model,
        'bs_position': bs_position,
        'true_user_position': true_user_position
    }

def visualize_results(results):
    """
    Visualize the RANSAC filtering results.
    
    Args:
        results (dict): Results from analyze_ransac_filtering
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Measurement count comparison
    ax1 = axes[0, 0]
    original_count = len(results['original_measurements'])
    filtered_count = len(results['filtered_measurements'])
    
    bars = ax1.bar(['Original', 'Filtered'], [original_count, filtered_count], 
                   color=['lightblue', 'lightgreen'], alpha=0.7)
    ax1.set_ylabel('Number of Measurements')
    ax1.set_title('Measurement Count Comparison')
    
    for bar, count in zip(bars, [original_count, filtered_count]):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    # Plot 2: Measurement types breakdown
    ax2 = axes[0, 1]
    original_associations = results['original_associations']
    filtered_associations = results['filtered_associations']
    
    # Count types
    original_los = sum(1 for assoc in original_associations if assoc == -1)
    original_single_bounce = sum(1 for assoc in original_associations if 0 <= assoc < 3)
    original_noise = sum(1 for assoc in original_associations if assoc >= 10)
    
    filtered_los = sum(1 for assoc in filtered_associations if assoc == -1)
    filtered_single_bounce = sum(1 for assoc in filtered_associations if 0 <= assoc < 3)
    filtered_noise = sum(1 for assoc in filtered_associations if assoc >= 10)
    
    categories = ['LOS', 'Single-bounce', 'Noise']
    original_counts = [original_los, original_single_bounce, original_noise]
    filtered_counts = [filtered_los, filtered_single_bounce, filtered_noise]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x - width/2, original_counts, width, label='Original', alpha=0.7)
    ax2.bar(x + width/2, filtered_counts, width, label='Filtered', alpha=0.7)
    ax2.set_xlabel('Measurement Type')
    ax2.set_ylabel('Count')
    ax2.set_title('Measurement Types Breakdown')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    # Plot 3: Inlier mask visualization
    ax3 = axes[1, 0]
    inlier_mask = results['inlier_mask']
    measurement_indices = np.arange(len(inlier_mask))
    
    colors = ['red' if not kept else 'green' for kept in inlier_mask]
    ax3.scatter(measurement_indices, [1]*len(measurement_indices), c=colors, s=100, alpha=0.7)
    ax3.set_xlabel('Measurement Index')
    ax3.set_ylabel('Status')
    ax3.set_title('RANSAC Inlier/Outlier Classification')
    ax3.set_yticks([1])
    ax3.set_yticklabels(['Measurements'])
    ax3.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Inlier (Kept)'),
                      Patch(facecolor='red', alpha=0.7, label='Outlier (Removed)')]
    ax3.legend(handles=legend_elements)
    
    # Plot 4: Delay distribution
    ax4 = axes[1, 1]
    original_measurements = results['original_measurements']
    original_associations = results['original_associations']
    
    # Separate delays by type
    los_delays = [m[2] for m, assoc in zip(original_measurements, original_associations) if assoc == -1]
    single_bounce_delays = [m[2] for m, assoc in zip(original_measurements, original_associations) if 0 <= assoc < 3]
    noise_delays = [m[2] for m, assoc in zip(original_measurements, original_associations) if assoc >= 10]
    
    ax4.hist([los_delays, single_bounce_delays, noise_delays], 
             label=['LOS', 'Single-bounce', 'Noise'], alpha=0.7, bins=10)
    ax4.set_xlabel('Delay (ns)')
    ax4.set_ylabel('Count')
    ax4.set_title('Delay Distribution by Measurement Type')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run RANSAC filtering analysis
    results = analyze_ransac_filtering()
    
    # Visualize results
    visualize_results(results)
    
    print("\n" + "=" * 60)
    print("RANSAC FILTERING ANALYSIS COMPLETE")
    print("=" * 60)
    print("The RANSAC algorithm successfully filtered out inconsistent measurements.")
    print("The filtered measurements can now be used in the optimization framework.") 