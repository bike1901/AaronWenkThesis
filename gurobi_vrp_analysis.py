import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from haversine import haversine, Unit
import json
import os
import random
from collections import defaultdict
import time
import networkx as nx

# Check if Gurobi is available
gurobi_available = False
try:
    import gurobipy as gp
    from gurobipy import GRB
    gurobi_available = True
except ImportError:
    print("Gurobi not available. Please install it first with:")
    print("pip install gurobipy")

# Constants
BRAZIL_DATASET = 'brazil-coffee-v2.5.1-2024-04-26.csv'
GEOCACHE_FILE = 'municipality_geocache.json'
OUTPUT_DIR = 'gurobi_vrp_analysis'
COLLECTIVE_DIR = 'collective_analysis'
COLLECTIVE_RECOMMENDATIONS_FILE = os.path.join(COLLECTIVE_DIR, 'collective_recommendations.txt')

# Brazilian Coffee Export Ports (major ports that handle coffee exports)
MAJOR_PORTS = {
    'Santos': {'latitude': -23.9619, 'longitude': -46.3042, 'capacity': 100},  # Santos is the largest coffee export port
    'Vitoria': {'latitude': -20.2976, 'longitude': -40.2958, 'capacity': 40},  # Important for Espírito Santo coffee
}

# Parameters for simulation
NUM_VEHICLES = 35
VEHICLE_CAPACITY = 1500  # in tonnes
TIME_LIMIT_SECONDS = 180  # Time limit for the solver (seconds)
RANDOM_SEED = 42  # For reproducibility
DEBUG_MODE = True  # Enable debug output

# Target collective to analyze - we'll select a promising one from the recommendations
TARGET_COLLECTIVE_ID = 1  # Use collective #1 from the expanded recommendations by default

def create_output_directory():
    """Create output directory for analysis results."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def load_municipality_locations():
    """Load municipality geocache data."""
    if not os.path.exists(GEOCACHE_FILE):
        print(f"Error: Geocache file {GEOCACHE_FILE} not found.")
        print("Run munGeo.py first to generate municipality coordinates.")
        return None
    
    try:
        with open(GEOCACHE_FILE, 'r') as f:
            geocache = json.load(f)
        
        # Convert from municipality_full to municipality
        municipality_locations = {}
        for muni_full, coords in geocache.items():
            if coords:  # Skip municipalities with no coordinates
                # Extract just the municipality name (before the first comma)
                municipality = muni_full.split(',')[0].strip()
                municipality_locations[municipality] = coords
        
        print(f"Loaded coordinates for {len(municipality_locations)} municipalities.")
        return municipality_locations
    
    except Exception as e:
        print(f"Error loading geocache: {e}")
        return None

def load_coffee_data():
    """Load coffee production data from Brazil dataset."""
    try:
        df = pd.read_csv(BRAZIL_DATASET)
        print(f"Loaded dataset with {len(df)} records")
        
        # Calculate total volume by municipality
        municipality_volumes = df.groupby('municipality')['volume'].sum().to_dict()
        print(f"Calculated volumes for {len(municipality_volumes)} municipalities")
        
        return df, municipality_volumes
    
    except FileNotFoundError:
        print(f"Error: Dataset file {BRAZIL_DATASET} not found.")
        return None, None

def parse_target_collective():
    """Parse the collective recommendations file to extract the target collective."""
    try:
        with open(COLLECTIVE_RECOMMENDATIONS_FILE, 'r') as f:
            content = f.read()
        
        # Extract expanded collectives section
        expanded_section = content.split("2. EXPANDED COLLECTIVE RECOMMENDATIONS")[1].split("3. SUMMARY STATISTICS")[0] if "3. SUMMARY STATISTICS" in content else content.split("2. EXPANDED COLLECTIVE RECOMMENDATIONS")[1]
        expanded_blocks = expanded_section.strip().split("Expanded Collective #")[1:]
        
        # If the TARGET_COLLECTIVE_ID is valid, use it, otherwise use the first one
        if 0 <= TARGET_COLLECTIVE_ID - 1 < len(expanded_blocks):
            block = expanded_blocks[TARGET_COLLECTIVE_ID - 1]
        else:
            print(f"Warning: Target collective #{TARGET_COLLECTIVE_ID} not found, using the first collective instead.")
            block = expanded_blocks[0] if expanded_blocks else None
        
        if not block:
            print("Error: No expanded collectives found in recommendations file.")
            return None
        
        # Parse the collective information
        municipalities = []
        max_distance = None
        total_volume = None
        suggested_hub = None
        
        lines = block.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            if 'Municipalities: ' in line:
                municipalities = line.split('Municipalities: ')[1].split(', ')
            elif 'Maximum Distance: ' in line:
                try:
                    max_distance = float(line.split('Maximum Distance: ')[1].split(' km')[0])
                except:
                    pass
            elif 'Total Volume: ' in line:
                try:
                    total_volume = float(line.split('Total Volume: ')[1])
                except:
                    pass
            elif 'Suggested Central Hub: ' in line:
                suggested_hub = line.split('Suggested Central Hub: ')[1]
        
        if not municipalities:
            print("Error: Could not parse municipalities from collective.")
            return None
        
        collective_info = {
            'municipalities': municipalities,
            'max_distance_km': max_distance,
            'total_volume': total_volume,
            'suggested_hub': suggested_hub
        }
        
        print(f"Selected collective with {len(municipalities)} municipalities")
        print(f"Municipalities: {', '.join(municipalities)}")
        if max_distance:
            print(f"Maximum Distance: {max_distance} km")
        if total_volume:
            print(f"Total Volume: {total_volume}")
        if suggested_hub:
            print(f"Suggested Central Hub: {suggested_hub}")
        
        return collective_info
    
    except Exception as e:
        print(f"Error parsing collective recommendations: {e}")
        return None

def create_distance_matrix(locations, port_locations):
    """
    Create a distance matrix between all locations (municipalities/collectives and ports).
    
    Parameters:
    - locations (dict): Dictionary of municipality locations
    - port_locations (dict): Dictionary of port locations
    
    Returns:
    - distance_matrix: 2D array of distances
    - location_indices: Dict mapping location names to indices in the matrix
    """
    # Combine all locations
    all_locations = {}
    all_locations.update(locations)
    for port, data in port_locations.items():
        all_locations[port] = (data['latitude'], data['longitude'])
    
    # Create an ordered list of location names
    location_names = list(all_locations.keys())
    location_indices = {name: i for i, name in enumerate(location_names)}
    
    # Initialize the distance matrix
    n = len(location_names)
    distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    # Debug: Print sample of input coordinates
    if DEBUG_MODE:
        print("\nSample of location coordinates:")
        sample_locations = list(all_locations.keys())[:min(3, len(all_locations))]
        for loc in sample_locations:
            print(f"  {loc}: {all_locations[loc]} (type: {type(all_locations[loc]).__name__})")
    
    # Calculate distances
    for i, origin in enumerate(location_names):
        origin_coords = all_locations[origin]
        
        # Handle both tuple and list formats for coordinates
        if isinstance(origin_coords, (tuple, list)) and len(origin_coords) == 2:
            origin_lat, origin_lon = origin_coords
        else:
            print(f"Warning: Invalid coordinates for {origin}: {origin_coords}")
            continue
        
        for j, destination in enumerate(location_names):
            if i == j:
                distance_matrix[i][j] = 0
                continue
            
            destination_coords = all_locations[destination]
            
            # Handle both tuple and list formats for coordinates
            if isinstance(destination_coords, (tuple, list)) and len(destination_coords) == 2:
                dest_lat, dest_lon = destination_coords
            else:
                print(f"Warning: Invalid coordinates for {destination}: {destination_coords}")
                continue
            
            try:
                # Calculate haversine distance
                distance_km = haversine((origin_lat, origin_lon), (dest_lat, dest_lon), unit=Unit.KILOMETERS)
                
                if DEBUG_MODE and i < 3 and j < 3 and i != j:
                    print(f"Raw distance from {origin} {(origin_lat, origin_lon)} to {destination} {(dest_lat, dest_lon)}: {distance_km:.2f} km")
                
                # Multiply by 100 to convert to integer centimeters (preserves precision for the solver)
                # This helps ensure non-zero distances between nearby locations
                distance_matrix[i][j] = max(1, int(distance_km * 100))  # Ensure minimum distance of 1
            except Exception as e:
                print(f"Error calculating distance between {origin} and {destination}: {e}")
                distance_matrix[i][j] = 9999  # Use a large value for errors
    
    print(f"Created distance matrix of size {n}x{n} for all locations and ports")
    
    # Debug: Print a sample of the distance matrix
    if DEBUG_MODE and n > 1:
        print(f"Sample distances from distance matrix:")
        for i in range(min(3, n)):
            for j in range(min(3, n)):
                if i != j:
                    print(f"  Distance from {location_names[i]} to {location_names[j]}: {distance_matrix[i][j]/100.0:.2f} km")
    
    return distance_matrix, location_indices, all_locations

def solve_vrp_gurobi(distance_matrix, demands, location_indices, num_vehicles, vehicle_capacity, depot_index):
    """
    Solve the Vehicle Routing Problem using Gurobi with support for split deliveries.
    
    Parameters:
    - distance_matrix: 2D array of distances between locations
    - demands: Dictionary of demands (volumes) for each location
    - location_indices: Dictionary mapping location names to indices
    - num_vehicles: Number of vehicles available
    - vehicle_capacity: Capacity of each vehicle
    - depot_index: Index of the depot location
    
    Returns:
    - solution: Dictionary containing routes, distances, etc.
    - status: Solution status
    """
    try:
        if not gurobi_available:
            print("Gurobi is not available. Please install it first.")
            return None, "NO_GUROBI"
        
        # Create reverse mapping from indices to location names
        index_to_location = {v: k for k, v in location_indices.items()}
        
        # Create a list of locations (excluding the depot)
        n = len(distance_matrix)
        locations = list(range(n))
        locations.remove(depot_index)  # Remove depot from customer locations
        
        # Create a list of vehicles
        vehicles = list(range(num_vehicles))
        
        # Create the model
        model = gp.Model("VRP")
        
        # Add decision variables
        # x[i,j,k] = 1 if vehicle k travels from location i to location j
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:  # No self-loops
                    for k in vehicles:
                        x[i,j,k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
        
        # y[i,k] = 1 if location i is visited by vehicle k
        y = {}
        for i in locations:  # Only for customer locations, not depot
            for k in vehicles:
                y[i,k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")
        
        # z[i,k] = amount delivered to location i by vehicle k
        z = {}
        for i in locations:
            for k in vehicles:
                z[i,k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"z_{i}_{k}")
        
        # Update the model to include the new variables
        model.update()
        
        # Set objective: minimize total distance
        obj = gp.LinExpr()
        for i in range(n):
            for j in range(n):
                if i != j:
                    for k in vehicles:
                        # Print a sample of the distance matrix for debugging
                        if i < 3 and j < 3 and k == 0:
                            print(f"Distance matrix sample: {distance_matrix[i][j]}")
                        obj += distance_matrix[i][j] * x[i,j,k]
        
        # Add a small penalty for using vehicles to encourage fewer vehicles
        for k in vehicles:
            obj += 800 * x[depot_index,locations[0] if locations else 0,k]  # Penalty for using a vehicle
        
        model.setObjective(obj, GRB.MINIMIZE)
        print("Objective function set with distance and vehicle usage components")
        
        # Add constraints
        
        # Each customer must be visited by at least one vehicle
        for i in locations:
            model.addConstr(gp.quicksum(y[i,k] for k in vehicles) >= 1, f"visit_{i}")
            
        # Add a constraint to ensure we use at least one vehicle
        if locations:  # Only if we have customer locations
            model.addConstr(gp.quicksum(x[depot_index,j,k] for j in locations for k in vehicles) >= 1, "use_vehicle")
        
        # Flow conservation: if a vehicle visits a location, it must also leave
        for i in range(n):
            for k in vehicles:
                # Sum of flows into i equals sum of flows out of i
                in_flow = gp.quicksum(x[j,i,k] for j in range(n) if j != i)
                out_flow = gp.quicksum(x[i,j,k] for j in range(n) if j != i)
                model.addConstr(in_flow == out_flow, f"flow_{i}_{k}")
        
        # Each vehicle starts and ends at the depot
        for k in vehicles:
            # Vehicle must leave depot at most once
            model.addConstr(gp.quicksum(x[depot_index,j,k] for j in range(n) if j != depot_index) <= 1, f"depot_out_{k}")
            # Vehicle must return to depot if it was used
            model.addConstr(gp.quicksum(x[i,depot_index,k] for i in range(n) if i != depot_index) == 
                           gp.quicksum(x[depot_index,j,k] for j in range(n) if j != depot_index), f"depot_in_{k}")
        
        # Link x and y variables: if x[i,j,k] = 1, then y[j,k] = 1
        for i in range(n):
            for j in locations:  # Only for customer locations
                if i != j:
                    for k in vehicles:
                        model.addConstr(x[i,j,k] <= y[j,k], f"link_x_y_{i}_{j}_{k}")
        
        # Capacity constraints
        for k in vehicles:
            model.addConstr(gp.quicksum(z[i,k] for i in locations) <= vehicle_capacity, f"capacity_{k}")
        
        # Demand constraints: total delivery to each customer must meet demand
        for i in locations:
            location_name = index_to_location[i]
            if location_name in demands:
                demand_i = demands[location_name]
                # Ensure demand is satisfied
                model.addConstr(gp.quicksum(z[i,k] for k in vehicles) == demand_i, f"demand_{i}")
                # Force at least one vehicle to visit this location
                model.addConstr(gp.quicksum(y[i,k] for k in vehicles) >= 1, f"force_visit_{i}")
        
        # Link y and z variables: if y[i,k] = 0, then z[i,k] = 0
        for i in locations:
            location_name = index_to_location[i]
            if location_name in demands:
                demand_i = demands[location_name]
                for k in vehicles:
                    model.addConstr(z[i,k] <= demand_i * y[i,k], f"link_y_z_{i}_{k}")
        
        # Only add multi-stop constraints if we have enough locations
        if len(locations) >= 4:  # Only make sense for multiple locations
            # Encourage multi-stop routes by adding constraints
            # For each vehicle, if it's used, it should visit at least 2 locations (besides depot)
            # This prevents the direct depot-to-location-to-depot routes
            for k in vehicles:
                # If vehicle k is used
                is_used = gp.quicksum(x[depot_index,j,k] for j in locations)
                # Then it should visit at least 2 locations
                min_visits = gp.quicksum(y[i,k] for i in locations)
                # Add constraint: if used, visit at least 2 locations
                model.addConstr(min_visits >= 2 * is_used, f"multi_stop_{k}")
        
        # Subtour elimination using MTZ formulation
        # u[i] = position of location i in the route
        u = {}
        for i in locations:
            for k in vehicles:
                u[i,k] = model.addVar(lb=0, ub=len(locations), vtype=GRB.CONTINUOUS, name=f"u_{i}_{k}")
        
        # Update model with new variables
        model.update()
        
        # MTZ constraints
        for i in locations:
            for j in locations:
                if i != j:
                    for k in vehicles:
                        model.addConstr(u[i,k] - u[j,k] + len(locations) * x[i,j,k] <= len(locations) - 1, f"mtz_{i}_{j}_{k}")
        
        # Set time limit
        model.setParam('TimeLimit', TIME_LIMIT_SECONDS)
        # Set MIP gap tolerance
        model.setParam('MIPGap', 0.05)  # 5% gap is acceptable
        
        # Solve the model
        model.optimize()
        
        # Check if a solution was found
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
            # Extract routes
            routes = []
            total_distance = 0
            max_route_distance = 0
            total_load = 0
            
            for k in vehicles:
                # Check if vehicle k is used
                if sum(x[depot_index,j,k].X > 0.5 for j in range(n) if j != depot_index) > 0:
                    route = [depot_index]  # Start at depot
                    current = depot_index
                    route_distance = 0
                    route_load = 0
                    
                    # Follow the route until we return to depot
                    while True:
                        # Find the next location
                        next_loc = None
                        for j in range(n):
                            if j != current and x[current,j,k].X > 0.5:
                                next_loc = j
                                break
                        
                        if next_loc is None or next_loc == depot_index:
                            # Add depot as final destination if not already there
                            if route[-1] != depot_index:
                                route.append(depot_index)
                                route_distance += distance_matrix[current][depot_index]
                            break
                        
                        # Add to route
                        route.append(next_loc)
                        route_distance += distance_matrix[current][next_loc]
                        
                        # Add load if it's a customer location
                        if next_loc in locations:
                            route_load += z[next_loc,k].X if (next_loc,k) in z else 0
                        
                        # Move to next location
                        current = next_loc
                    
                    # Convert indices to location names
                    route_names = [index_to_location[i] for i in route]
                    
                    # Add route to list
                    routes.append({
                        'route': route_names,
                        'distance': route_distance / 100.0,  # Convert back to km
                        'load': route_load
                    })
            
            # Calculate total distance
            total_distance = sum(route['distance'] for route in routes)
            
            # Calculate maximum route distance
            max_route_distance = max(route['distance'] for route in routes) if routes else 0
            
            # Calculate total load
            total_load = sum(route['load'] for route in routes)
            
            # Debug output for distance calculations
            if DEBUG_MODE:
                print("\nDEBUG: Distance Calculation Details")
                print(f"Number of routes: {len(routes)}")
                print(f"Route distances: {[f'{route['distance']:.2f}' for route in routes]}")
                print(f"Total distance sum: {total_distance:.2f} km")
                print(f"Max route distance: {max_route_distance:.2f} km")
                print(f"Average route distance: {total_distance / len(routes) if routes else 0:.2f} km")
            
            # Prepare final solution
            solution = {
                'status': 'OPTIMAL' if model.status == GRB.OPTIMAL else 'FEASIBLE',
                'routes': routes,
                'total_distance': total_distance,
                'max_route_distance': max_route_distance,
                'num_routes': len(routes),
                'total_load': total_load,
                'average_route_distance': total_distance / len(routes) if routes else 0
            }
            
            return solution, solution['status']
        
        else:
            print(f"No solution found. Gurobi status: {model.status}")
            return None, "NO_SOLUTION"
    
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None, "ERROR"
    except Exception as e:
        print(f"Error in VRP model: {e}")
        return None, "ERROR"

def visualize_vrp_solution(locations, solution, scenario_name):
    """Create a visualization of the VRP solution."""
    # Check if we have a valid solution
    if not solution or 'routes' not in solution or not solution['routes']:
        print(f"No solution to visualize for {scenario_name}")
        return None
    
    # Create a folium map centered on the average of all locations
    lat_values = []
    lon_values = []
    for loc_name, coords in locations.items():
        if isinstance(coords, tuple) and len(coords) == 2:
            lat, lon = coords
            lat_values.append(lat)
            lon_values.append(lon)
        elif isinstance(coords, dict) and 'latitude' in coords and 'longitude' in coords:
            lat_values.append(coords['latitude'])
            lon_values.append(coords['longitude'])
    
    if lat_values and lon_values:
        center_lat = sum(lat_values) / len(lat_values)
        center_lon = sum(lon_values) / len(lon_values)
    else:
        # Default to Brazil center if no locations
        center_lat, center_lon = -15.77972, -47.92972
    
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Define colors for routes (up to 10 routes, then cycle)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'cadetblue', 'darkgreen', 'darkpurple', 'pink']
    
    # Add markers for ports
    for port_name, port_data in MAJOR_PORTS.items():
        if port_name in locations:
            folium.Marker(
                location=[port_data['latitude'], port_data['longitude']],
                popup=f"Port: {port_name}<br>Capacity: {port_data['capacity']}",
                icon=folium.Icon(color='darkblue', icon='ship', prefix='fa')
            ).add_to(map_obj)
    
    # Add markers for locations directly to the map (not in a cluster) for better visibility
    for location_name, coords in locations.items():
        # Skip ports, they're already added
        if location_name in MAJOR_PORTS:
            continue
            
        if isinstance(coords, tuple) and len(coords) == 2:
            lat, lon = coords
        elif isinstance(coords, dict) and 'latitude' in coords and 'longitude' in coords:
            lat, lon = coords['latitude'], coords['longitude']
        else:
            continue
        
        # Determine marker color based on type
        if location_name == 'Collective_Hub':
            color = 'green'
            icon = 'users'
            size = (6, 6)  # Make collective hub larger
        else:
            color = 'red'
            icon = 'coffee'
            size = (4, 4)
        
        # Use a more visible marker style
        folium.Marker(
            location=[lat, lon],
            popup=f"{location_name}",
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(map_obj)
    
    # Add polylines for routes
    for i, route_data in enumerate(solution['routes']):
        route = route_data['route']  # Get the route as a list of location names
        route_points = []
        color = colors[i % len(colors)]
        
        for location_name in route:
            # Skip locations without coordinates
            if location_name not in locations:
                continue
                
            # Get coordinates
            coords = locations[location_name]
            if isinstance(coords, tuple) and len(coords) == 2:
                lat, lon = coords
            elif isinstance(coords, dict) and 'latitude' in coords and 'longitude' in coords:
                lat, lon = coords['latitude'], coords['longitude']
            else:
                continue
                
            route_points.append([lat, lon])
        
        # Create the route polyline if we have points
        if len(route_points) > 1:
            folium.PolyLine(
                locations=route_points,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"Route {i+1}: {route_data['distance']} km, {route_data['load']} tonnes"
            ).add_to(map_obj)
    
    # Save the map
    map_file = os.path.join(OUTPUT_DIR, f'{scenario_name}_vrp_map.html')
    map_obj.save(map_file)
    print(f"Created visualization: {map_file}")
    
    return map_file

def create_comparison_visualization(pre_results, post_results):
    """Create visualizations comparing pre and post collective scenarios."""
    # Calculate comparison metrics
    if pre_results and post_results:
        distance_reduction = pre_results['total_distance'] - post_results['total_distance']
        distance_reduction_pct = (distance_reduction / pre_results['total_distance'] * 100) if pre_results['total_distance'] > 0 else 0
        
        # Calculate efficiency (distance per tonne)
        pre_efficiency = pre_results['total_distance'] / pre_results['total_load'] if pre_results['total_load'] > 0 else 0
        post_efficiency = post_results['total_distance'] / post_results['total_load'] if post_results['total_load'] > 0 else 0
        
        efficiency_improvement = pre_efficiency - post_efficiency
        efficiency_pct = (efficiency_improvement / pre_efficiency * 100) if pre_efficiency > 0 else 0
        
        # Add to results for later reference
        pre_results['distance_per_tonne'] = pre_efficiency
        post_results['distance_per_tonne'] = post_efficiency
    else:
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Create bar chart comparing key metrics
    plt.figure(figsize=(12, 10))
    
    # Distance comparison
    plt.subplot(2, 2, 1)
    distances = [pre_results['total_distance'], post_results['total_distance']]
    plt.bar(['Individual Farms', 'Collective'], distances, color=['blue', 'green'])
    plt.title('Total Transportation Distance')
    plt.ylabel('Distance (km)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage change
    plt.annotate(f"-{distance_reduction_pct:.1f}%", 
                 xy=(1, post_results['total_distance']), 
                 xytext=(0.5, (pre_results['total_distance'] + post_results['total_distance'])/2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Efficiency comparison
    plt.subplot(2, 2, 2)
    efficiencies = [pre_results['distance_per_tonne'], post_results['distance_per_tonne']]
    plt.bar(['Individual Farms', 'Collective'], efficiencies, color=['blue', 'green'])
    plt.title('Transportation Efficiency')
    plt.ylabel('Distance per Tonne (km/tonne)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage change
    plt.annotate(f"-{efficiency_pct:.1f}%", 
                 xy=(1, post_results['distance_per_tonne']), 
                 xytext=(0.5, (pre_results['distance_per_tonne'] + post_results['distance_per_tonne'])/2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Number of routes comparison
    plt.subplot(2, 2, 3)
    routes = [pre_results['num_routes'], post_results['num_routes']]
    plt.bar(['Individual Farms', 'Collective'], routes, color=['blue', 'green'])
    plt.title('Number of Routes')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage change
    routes_change_pct = (pre_results['num_routes'] - post_results['num_routes']) / pre_results['num_routes'] * 100 if pre_results['num_routes'] > 0 else 0
    routes_change_text = f"-{routes_change_pct:.1f}%" if routes_change_pct > 0 else f"+{-routes_change_pct:.1f}%"
    plt.annotate(routes_change_text, 
                 xy=(1, post_results['num_routes']), 
                 xytext=(0.5, (pre_results['num_routes'] + post_results['num_routes'])/2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Max route distance comparison
    plt.subplot(2, 2, 4)
    max_routes = [pre_results['max_route_distance'], post_results['max_route_distance']]
    plt.bar(['Individual Farms', 'Collective'], max_routes, color=['blue', 'green'])
    plt.title('Maximum Route Distance')
    plt.ylabel('Distance (km)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage change
    max_route_change_pct = (pre_results['max_route_distance'] - post_results['max_route_distance']) / pre_results['max_route_distance'] * 100 if pre_results['max_route_distance'] > 0 else 0
    max_route_text = f"-{max_route_change_pct:.1f}%" if max_route_change_pct > 0 else f"+{-max_route_change_pct:.1f}%"
    plt.annotate(max_route_text, 
                 xy=(1, post_results['max_route_distance']), 
                 xytext=(0.5, (pre_results['max_route_distance'] + post_results['max_route_distance'])/2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    plt.tight_layout()
    plt.suptitle('Impact of Coffee Farming Collective on Regional Transportation', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'regional_collective_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created comparison visualization: {os.path.join(OUTPUT_DIR, 'regional_collective_impact.png')}")

def run_regional_analysis():
    """Run the VRP analysis for a specific region with and without a collective."""
    # Initialize
    random.seed(RANDOM_SEED)
    create_output_directory()
    
    # Load data
    municipality_locations = load_municipality_locations()
    df, municipality_volumes = load_coffee_data()
    
    if not municipality_locations or not municipality_volumes:
        print("Error: Could not load required data.")
        return
    
    # Parse the target collective
    target_collective = parse_target_collective()
    
    if not target_collective or not target_collective['municipalities']:
        print("Error: Could not find target collective information.")
        return
    
    # Filter locations and volumes to just include the collective municipalities
    collective_municipalities = target_collective['municipalities']
    
    # Check if all municipalities in the collective have locations and volumes
    valid_municipalities = []
    for municipality in collective_municipalities:
        if municipality in municipality_locations and municipality in municipality_volumes:
            valid_municipalities.append(municipality)
        else:
            print(f"Warning: Municipality '{municipality}' doesn't have coordinates or volume data, skipping.")
    
    if not valid_municipalities:
        print("Error: None of the municipalities in the collective have valid data.")
        return
    
    # Select the nearest port as the depot
    # Calculate average position of all municipalities in the collective
    avg_lat = sum(municipality_locations[m][0] for m in valid_municipalities) / len(valid_municipalities)
    avg_lon = sum(municipality_locations[m][1] for m in valid_municipalities) / len(valid_municipalities)
    
    # Find the nearest port
    nearest_port = None
    min_distance = float('inf')
    for port_name, port_data in MAJOR_PORTS.items():
        distance = haversine((avg_lat, avg_lon), (port_data['latitude'], port_data['longitude']), unit=Unit.KILOMETERS)
        if distance < min_distance:
            min_distance = distance
            nearest_port = port_name
    
    if not nearest_port:
        print("Error: Could not determine nearest port.")
        return
    
    print(f"Selected '{nearest_port}' as the nearest port (distance: {min_distance:.1f} km)")
    
    # Scenario 1: Pre-collective (individual municipalities)
    print("\n===== SCENARIO 1: INDIVIDUAL MUNICIPALITIES =====")
    
    # Create a dictionary of municipalities and their locations
    pre_locations = {m: municipality_locations[m] for m in valid_municipalities}
    pre_volumes = {m: municipality_volumes[m] for m in valid_municipalities}
    
    # Add the port
    port_locations = {nearest_port: MAJOR_PORTS[nearest_port]}
    
    # Create distance matrix
    pre_distance_matrix, pre_location_indices, pre_all_locations = create_distance_matrix(pre_locations, port_locations)
    
    # Get depot index
    depot_index = pre_location_indices[nearest_port]
    
    # Solve VRP with Gurobi
    pre_solution, pre_status = solve_vrp_gurobi(
        pre_distance_matrix, pre_volumes, 
        pre_location_indices, NUM_VEHICLES, VEHICLE_CAPACITY, depot_index
    )
    
    if pre_solution:
        # Visualize pre-collective solution
        pre_visualize_file = visualize_vrp_solution(
            pre_all_locations, pre_solution, "individual_farms"
        )
        
        # Print results
        print(f"Individual Farms Results:")
        print(f"  Status: {pre_status}")
        print(f"  Total Distance: {pre_solution['total_distance']} km")
        print(f"  Max Route Distance: {pre_solution['max_route_distance']} km")
        print(f"  Number of Routes: {pre_solution['num_routes']}")
        print(f"  Average Route Distance: {pre_solution['average_route_distance']:.1f} km")
        print(f"  Total Volume: {pre_solution['total_load']} tonnes")
        
        # Calculate efficiency
        distance_per_tonne = pre_solution['total_distance'] / pre_solution['total_load'] if pre_solution['total_load'] > 0 else 0
        print(f"  Distance per Tonne: {distance_per_tonne:.1f} km/tonne")
        
        # Add to solution for comparison later
        pre_solution['distance_per_tonne'] = distance_per_tonne
    else:
        print(f"No solution found for individual farms scenario. Status: {pre_status}")
    
    # Scenario 2: Post-collective (consolidated collection point)
    print("\n===== SCENARIO 2: CONSOLIDATED COLLECTIVE =====")
    
    # Calculate centroid of the collective for the consolidated pickup point
    # Use the suggested hub if available, otherwise calculate the centroid
    if target_collective['suggested_hub'] and target_collective['suggested_hub'] in municipality_locations:
        collective_hub_coords = municipality_locations[target_collective['suggested_hub']]
        print(f"Using suggested hub '{target_collective['suggested_hub']}' as collective center")
    else:
        # Calculate centroid of the collective
        lat_sum = sum(municipality_locations[m][0] for m in valid_municipalities)
        lon_sum = sum(municipality_locations[m][1] for m in valid_municipalities)
        collective_hub_coords = (lat_sum / len(valid_municipalities), lon_sum / len(valid_municipalities))
        print(f"Calculated centroid as collective center: {collective_hub_coords}")
    
    # Create a dictionary with the collective hub as a single point
    post_locations = {'Collective_Hub': collective_hub_coords}
    
    # Calculate total volume for the collective
    collective_volume = sum(municipality_volumes[m] for m in valid_municipalities)
    post_volumes = {'Collective_Hub': collective_volume}
    
    # Create distance matrix
    post_distance_matrix, post_location_indices, post_all_locations = create_distance_matrix(post_locations, port_locations)
    
    # Get depot index
    post_depot_index = post_location_indices[nearest_port]
    
    # For the collective scenario, calculate required vehicles based on volume
    required_vehicles = max(1, int(np.ceil(collective_volume / VEHICLE_CAPACITY)))
    # Add a few extra vehicles for flexibility in the model
    collective_num_vehicles = min(NUM_VEHICLES, required_vehicles + 3)
    
    # Solve with Gurobi
    post_solution, post_status = solve_vrp_gurobi(
        post_distance_matrix, post_volumes, 
        post_location_indices, collective_num_vehicles, VEHICLE_CAPACITY, post_depot_index
    )
    
    if post_solution:
        # Visualize post-collective solution
        post_visualize_file = visualize_vrp_solution(
            post_all_locations, post_solution, "collective"
        )
        
        # Print results
        print(f"Collective Results:")
        print(f"  Status: {post_status}")
        print(f"  Total Distance: {post_solution['total_distance']} km")
        print(f"  Max Route Distance: {post_solution['max_route_distance']} km")
        print(f"  Number of Routes: {post_solution['num_routes']}")
        print(f"  Average Route Distance: {post_solution['average_route_distance']:.1f} km")
        print(f"  Total Volume: {post_solution['total_load']} tonnes")
        
        # Calculate efficiency
        distance_per_tonne = post_solution['total_distance'] / post_solution['total_load'] if post_solution['total_load'] > 0 else 0
        print(f"  Distance per Tonne: {distance_per_tonne:.1f} km/tonne")
        
        # Add to solution for comparison
        post_solution['distance_per_tonne'] = distance_per_tonne
    else:
        print(f"No solution found for collective scenario. Status: {post_status}")
    
    # Compare results
    if pre_solution and post_solution:
        # Calculate impact
        distance_reduction = pre_solution['total_distance'] - post_solution['total_distance']
        distance_reduction_pct = (distance_reduction / pre_solution['total_distance'] * 100) if pre_solution['total_distance'] > 0 else 0
        
        efficiency_improvement = pre_solution['distance_per_tonne'] - post_solution['distance_per_tonne']
        efficiency_pct = (efficiency_improvement / pre_solution['distance_per_tonne'] * 100) if pre_solution['distance_per_tonne'] > 0 else 0
        
        print("\n===== IMPACT OF COLLECTIVE FORMATION =====")
        print(f"Total Distance Reduction: {distance_reduction:.1f} km ({distance_reduction_pct:.1f}%)")
        print(f"Transportation Efficiency Improvement: {efficiency_improvement:.1f} km/tonne ({efficiency_pct:.1f}%)")
        # Handle potential division by zero
        if pre_solution['num_routes'] > 0:
            route_reduction_pct = (pre_solution['num_routes'] - post_solution['num_routes']) / pre_solution['num_routes'] * 100
            print(f"Route Reduction: {pre_solution['num_routes'] - post_solution['num_routes']} routes ({route_reduction_pct:.1f}%)")
        else:
            print("Route Reduction: N/A (no routes in pre-collective solution)")
        
        # Create comparison visualization
        create_comparison_visualization(pre_solution, post_solution)

def main():
    """Main function to run the regional VRP analysis with Gurobi."""
    print("=== REGIONAL VRP ANALYSIS FOR COFFEE COLLECTIVE USING GUROBI ===")
    
    # Check if Gurobi is available
    if not gurobi_available:
        print("Error: Gurobi not available. Please install it first with:")
        print("pip install gurobipy")
        return
    
    # Run the analysis
    run_regional_analysis()
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to {OUTPUT_DIR}/ directory")
    print("\nRecommended next steps:")
    print("1. Review the comparison visualization")
    print("2. Explore the interactive route maps in your browser")
    print("3. Try analyzing different target collectives by changing TARGET_COLLECTIVE_ID")

if __name__ == "__main__":
    main() 