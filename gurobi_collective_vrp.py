"""
Coffee Collective VRP Analysis using Gurobi

This script analyzes two logistics scenarios for a coffee farming collective:
1. Individual pickups from each municipality to Port Santos
2. Hub-based collection where all coffee is first brought to a central hub

Author: Cascade
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

# Define the municipalities and their coordinates
# Coordinates are in (latitude, longitude) format
MUNICIPALITIES = [
    'CHAVANTES',
    'MANDURI',
    'BERNARDINO DE CAMPOS',
    'IPAUSSU',
    'SANTA CRUZ DO RIO PARDO',
    'TIMBURI',
    'OLEO'
]

# Coordinates for each municipality (latitude, longitude)
# Note: These are placeholder coordinates - you should replace with actual values
MUNICIPALITY_COORDS = {
    'CHAVANTES': (-23.0378, -49.7073),
    'MANDURI': (-23.0056, -49.3203),
    'BERNARDINO DE CAMPOS': (-23.0103, -49.4686),
    'IPAUSSU': (-23.0439, -49.6236),
    'SANTA CRUZ DO RIO PARDO': (-22.8975, -49.6314),
    'TIMBURI': (-23.2058, -49.6039),
    'OLEO': (-23.1775, -49.3417)
}

# Port Santos coordinates
PORT_SANTOS_COORDS = (-23.9815, -46.2995)

# Define the hub municipality
HUB_MUNICIPALITY = 'IPAUSSU'

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def calculate_distance(coord1, coord2):
    """Calculate Haversine distance between two coordinates"""
    return haversine(coord1[0], coord1[1], coord2[0], coord2[1])

def scenario1_individual_pickups():
    """
    Scenario 1: Individual pickups from Port Santos to each municipality and back
    """
    total_distance = 0
    distances = {}
    
    for municipality in MUNICIPALITIES:
        coords = MUNICIPALITY_COORDS[municipality]
        # Calculate round-trip distance (Port Santos to municipality and back)
        distance = 2 * calculate_distance(PORT_SANTOS_COORDS, coords)
        distances[municipality] = distance
        total_distance += distance
    
    return total_distance, distances

def scenario2_hub_pickup():
    """
    Scenario 2: Municipalities bring coffee to hub, then one trip from hub to Port Santos
    """
    # Calculate internal collection distance (from each municipality to hub)
    internal_distance = 0
    internal_distances = {}
    hub_coords = MUNICIPALITY_COORDS[HUB_MUNICIPALITY]
    
    for municipality in MUNICIPALITIES:
        if municipality != HUB_MUNICIPALITY:
            coords = MUNICIPALITY_COORDS[municipality]
            distance = calculate_distance(hub_coords, coords)
            internal_distances[municipality] = distance
            internal_distance += distance
    
    # Calculate round-trip distance from hub to Port Santos
    port_distance = 2 * calculate_distance(hub_coords, PORT_SANTOS_COORDS)
    
    total_distance = internal_distance + port_distance
    
    return total_distance, internal_distances, port_distance

def solve_vrp_with_gurobi():
    """
    Solve the Vehicle Routing Problem using Gurobi
    """
    # Create distance matrix
    n = len(MUNICIPALITIES)
    dist_matrix = np.zeros((n+1, n+1))
    
    # Add Port Santos as the first node (index 0)
    for i, muni_i in enumerate(MUNICIPALITIES):
        # Distance from Port Santos to municipality
        dist_matrix[0, i+1] = calculate_distance(PORT_SANTOS_COORDS, MUNICIPALITY_COORDS[muni_i])
        dist_matrix[i+1, 0] = dist_matrix[0, i+1]  # Same distance for return trip
        
        # Distance between municipalities
        for j, muni_j in enumerate(MUNICIPALITIES):
            if i != j:
                dist_matrix[i+1, j+1] = calculate_distance(
                    MUNICIPALITY_COORDS[muni_i], 
                    MUNICIPALITY_COORDS[muni_j]
                )
    
    # Create a new model
    model = gp.Model("Coffee_VRP")
    
    # Create variables
    x = {}
    for i in range(n+1):
        for j in range(n+1):
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
    
    # Set objective: minimize total distance
    obj = gp.quicksum(dist_matrix[i, j] * x[i, j] for i in range(n+1) for j in range(n+1) if i != j)
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    
    # Each municipality must be visited exactly once
    for j in range(1, n+1):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n+1) if i != j) == 1, f"visit_{j}")
    
    # Each municipality must be left exactly once
    for i in range(1, n+1):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n+1) if i != j) == 1, f"leave_{i}")
    
    # Depot (Port Santos) must be visited as many times as there are vehicles
    # For simplicity, we'll use 1 vehicle
    model.addConstr(gp.quicksum(x[0, j] for j in range(1, n+1)) == 1, "depot_out")
    model.addConstr(gp.quicksum(x[i, 0] for i in range(1, n+1)) == 1, "depot_in")
    
    # Subtour elimination using MTZ formulation
    u = {}
    for i in range(1, n+1):
        u[i] = model.addVar(lb=0, ub=n, vtype=GRB.INTEGER, name=f'u_{i}')
    
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, f"mtz_{i}_{j}")
    
    # Optimize the model
    model.optimize()
    
    # Extract results
    if model.status == GRB.OPTIMAL:
        route = []
        current = 0  # Start at depot (Port Santos)
        total_distance = 0
        
        # Reconstruct the route
        while True:
            for j in range(n+1):
                if j != current and x[current, j].x > 0.5:
                    route.append(j)
                    total_distance += dist_matrix[current, j]
                    current = j
                    break
            if current == 0:
                break  # We've returned to the depot
        
        # Convert indices to location names for the output
        location_route = ["Port Santos"]
        for idx in route[:-1]:  # Exclude the last return to depot
            if idx > 0:
                location_route.append(MUNICIPALITIES[idx-1])
        location_route.append("Port Santos")
        
        return total_distance, location_route
    else:
        return None, None

def plot_scenario1(save_path):
    """Visualize Scenario 1: Individual pickups from Port Santos to each municipality"""
    plt.figure(figsize=(12, 10))
    
    # Create coordinates list for scatter plot
    all_coords = list(MUNICIPALITY_COORDS.values()) + [PORT_SANTOS_COORDS]
    lons = [coord[1] for coord in all_coords]
    lats = [coord[0] for coord in all_coords]
    
    # Plot municipalities as scatter points
    for municipality, coords in MUNICIPALITY_COORDS.items():
        plt.plot(coords[1], coords[0], 'bo', markersize=8)
        plt.text(coords[1], coords[0] + 0.02, municipality, fontsize=9, ha='center')
    
    # Plot Port Santos
    plt.plot(PORT_SANTOS_COORDS[1], PORT_SANTOS_COORDS[0], 'r*', markersize=15)
    plt.text(PORT_SANTOS_COORDS[1], PORT_SANTOS_COORDS[0] + 0.02, 'Port Santos', fontsize=12, ha='center', weight='bold')
    
    # Plot lines for each roundtrip
    for municipality, coords in MUNICIPALITY_COORDS.items():
        plt.plot([PORT_SANTOS_COORDS[1], coords[1], PORT_SANTOS_COORDS[1]], 
                 [PORT_SANTOS_COORDS[0], coords[0], PORT_SANTOS_COORDS[0]], 
                 'g-', alpha=0.5)
    
    # Set plot limits with some padding
    x_min, x_max = min(lons) - 0.2, max(lons) + 0.2
    y_min, y_max = min(lats) - 0.2, max(lats) + 0.2
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scenario 1: Individual Pickups (Direct to Port Santos)')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    plt.plot([], [], 'bo', label='Municipalities')
    plt.plot([], [], 'r*', markersize=10, label='Port Santos')
    plt.plot([], [], 'g-', label='Round-trip Route')
    plt.legend(loc='best')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved visualization to {save_path}")

def plot_scenario2(save_path):
    """Visualize Scenario 2: Hub-based collection"""
    plt.figure(figsize=(12, 10))
    
    # Create coordinates list for scatter plot
    all_coords = list(MUNICIPALITY_COORDS.values()) + [PORT_SANTOS_COORDS]
    lons = [coord[1] for coord in all_coords]
    lats = [coord[0] for coord in all_coords]
    
    # Plot municipalities as scatter points
    for municipality, coords in MUNICIPALITY_COORDS.items():
        if municipality == HUB_MUNICIPALITY:
            plt.plot(coords[1], coords[0], 'gs', markersize=12)
            plt.text(coords[1], coords[0] + 0.02, f"{municipality} (HUB)", fontsize=10, ha='center', weight='bold')
        else:
            plt.plot(coords[1], coords[0], 'bo', markersize=8)
            plt.text(coords[1], coords[0] + 0.02, municipality, fontsize=9, ha='center')
    
    # Plot Port Santos
    plt.plot(PORT_SANTOS_COORDS[1], PORT_SANTOS_COORDS[0], 'r*', markersize=15)
    plt.text(PORT_SANTOS_COORDS[1], PORT_SANTOS_COORDS[0] + 0.02, 'Port Santos', fontsize=12, ha='center', weight='bold')
    
    # Get hub coordinates
    hub_coords = MUNICIPALITY_COORDS[HUB_MUNICIPALITY]
    
    # Plot lines from municipalities to hub
    for municipality, coords in MUNICIPALITY_COORDS.items():
        if municipality != HUB_MUNICIPALITY:
            plt.plot([hub_coords[1], coords[1]], 
                     [hub_coords[0], coords[0]], 
                     'b-', alpha=0.5)
    
    # Plot round-trip line from hub to port
    plt.plot([hub_coords[1], PORT_SANTOS_COORDS[1], hub_coords[1]], 
             [hub_coords[0], PORT_SANTOS_COORDS[0], hub_coords[0]], 
             'r-', linewidth=2, alpha=0.7)
    
    # Set plot limits with some padding
    x_min, x_max = min(lons) - 0.2, max(lons) + 0.2
    y_min, y_max = min(lats) - 0.2, max(lats) + 0.2
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Scenario 2: Hub-based Collection (Hub: {HUB_MUNICIPALITY})')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    plt.plot([], [], 'bo', label='Municipalities')
    plt.plot([], [], 'gs', markersize=10, label=f'Hub ({HUB_MUNICIPALITY})')
    plt.plot([], [], 'r*', markersize=10, label='Port Santos')
    plt.plot([], [], 'b-', label='Internal Collection Routes')
    plt.plot([], [], 'r-', label='Hub-Port Round-trip')
    plt.legend(loc='best')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved visualization to {save_path}")

def plot_vrp_solution(route, save_path):
    """Visualize the optimal VRP solution"""
    plt.figure(figsize=(12, 10))
    
    # Create coordinates list for scatter plot
    all_coords = list(MUNICIPALITY_COORDS.values()) + [PORT_SANTOS_COORDS]
    lons = [coord[1] for coord in all_coords]
    lats = [coord[0] for coord in all_coords]
    
    # Map location names to coordinates
    location_to_coords = {
        'Port Santos': PORT_SANTOS_COORDS,
        **MUNICIPALITY_COORDS
    }
    
    # Plot municipalities as scatter points
    for municipality, coords in MUNICIPALITY_COORDS.items():
        plt.plot(coords[1], coords[0], 'bo', markersize=8)
        plt.text(coords[1], coords[0] + 0.02, municipality, fontsize=9, ha='center')
    
    # Plot Port Santos
    plt.plot(PORT_SANTOS_COORDS[1], PORT_SANTOS_COORDS[0], 'r*', markersize=15)
    plt.text(PORT_SANTOS_COORDS[1], PORT_SANTOS_COORDS[0] + 0.02, 'Port Santos', fontsize=12, ha='center', weight='bold')
    
    # Plot the VRP route
    route_coords = [location_to_coords[loc] for loc in route]
    for i in range(len(route_coords) - 1):
        plt.plot([route_coords[i][1], route_coords[i+1][1]],
                 [route_coords[i][0], route_coords[i+1][0]],
                 'g-', linewidth=2, alpha=0.7)
        
        # Add direction arrows
        mid_x = (route_coords[i][1] + route_coords[i+1][1]) / 2
        mid_y = (route_coords[i][0] + route_coords[i+1][0]) / 2
        dx = route_coords[i+1][1] - route_coords[i][1]
        dy = route_coords[i+1][0] - route_coords[i][0]
        
        # Normalize direction and scale
        mag = np.sqrt(dx**2 + dy**2)
        if mag > 0:
            dx, dy = dx/mag * 0.04, dy/mag * 0.04
            plt.arrow(mid_x - dx/2, mid_y - dy/2, dx, dy, 
                     head_width=0.02, head_length=0.03, fc='g', ec='g', alpha=0.7)
    
    # Number the stops in order
    for i, loc in enumerate(route):
        coords = location_to_coords[loc]
        plt.text(coords[1] + 0.03, coords[0], f"{i}", fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='circle'))
    
    # Set plot limits with some padding
    x_min, x_max = min(lons) - 0.2, max(lons) + 0.2
    y_min, y_max = min(lats) - 0.2, max(lats) + 0.2
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Optimal VRP Solution (Gurobi)')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    plt.plot([], [], 'bo', label='Municipalities')
    plt.plot([], [], 'r*', markersize=10, label='Port Santos')
    plt.plot([], [], 'g-', label='Optimal Route')
    plt.legend(loc='best')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved visualization to {save_path}")

def main():
    """Main function to run the analysis"""
    print("Coffee Collective VRP Analysis")
    print("==============================")
    print(f"Collective: {', '.join(MUNICIPALITIES)}")
    print(f"Hub Municipality: {HUB_MUNICIPALITY}")
    print("\n")
    
    # Create visualization output directory if it doesn't exist
    viz_dir = "/Users/aaronwenk/Desktop/CoffeeOptimization/gurobi_vrp_analysis"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        print(f"Created visualization directory: {viz_dir}")
    
    # Scenario 1: Individual Pickups
    s1_distance, s1_details = scenario1_individual_pickups()
    print("Scenario 1: Individual Pickups to Port Santos")
    print("-" * 50)
    for municipality, distance in s1_details.items():
        print(f"{municipality}: {distance:.2f} km (round-trip)")
    print(f"Total Distance (Scenario 1): {s1_distance:.2f} km")
    
    # Generate visualization for Scenario 1
    s1_viz_path = os.path.join(viz_dir, "scenario1_individual_pickups.png")
    plot_scenario1(s1_viz_path)
    print("\n")
    
    # Scenario 2: Hub-based Pickup
    s2_distance, s2_internal, s2_port = scenario2_hub_pickup()
    print("Scenario 2: Hub-based Collection with Central Pickup")
    print("-" * 50)
    for municipality, distance in s2_internal.items():
        print(f"{municipality} to {HUB_MUNICIPALITY}: {distance:.2f} km")
    print(f"Hub to Port Santos (round-trip): {s2_port:.2f} km")
    print(f"Total Distance (Scenario 2): {s2_distance:.2f} km")
    
    # Generate visualization for Scenario 2
    s2_viz_path = os.path.join(viz_dir, "scenario2_hub_based.png")
    plot_scenario2(s2_viz_path)
    print("\n")
    
    # Calculate savings
    savings = s1_distance - s2_distance
    percent_savings = (savings / s1_distance) * 100
    
    print("Comparison Results")
    print("-" * 50)
    print(f"Distance Savings: {savings:.2f} km")
    print(f"Percentage Improvement: {percent_savings:.2f}%")
    print("\n")
    
    # Solve VRP with Gurobi
    print("Optimal VRP Solution (Gurobi)")
    print("-" * 50)
    vrp_distance, vrp_route = solve_vrp_with_gurobi()
    
    if vrp_distance is not None:
        print(f"Optimal Route: {' -> '.join(vrp_route)}")
        print(f"Total Distance (VRP): {vrp_distance:.2f} km")
        
        # Compare with other scenarios
        vrp_savings_s1 = s1_distance - vrp_distance
        vrp_percent_s1 = (vrp_savings_s1 / s1_distance) * 100
        
        vrp_savings_s2 = s2_distance - vrp_distance
        vrp_percent_s2 = (vrp_savings_s2 / s2_distance) * 100
        
        print(f"Savings vs. Scenario 1: {vrp_savings_s1:.2f} km ({vrp_percent_s1:.2f}%)")
        print(f"Savings vs. Scenario 2: {vrp_savings_s2:.2f} km ({vrp_percent_s2:.2f}%)")
        
        # Generate visualization for VRP solution
        vrp_viz_path = os.path.join(viz_dir, "scenario3_optimal_vrp.png")
        plot_vrp_solution(vrp_route, vrp_viz_path)
        
        # Generate combined visualization comparing all scenarios
        combined_viz_path = os.path.join(viz_dir, "scenario_comparison.png")
        plt.figure(figsize=(15, 12))
        
        # Plot all municipalities and Port Santos
        for municipality, coords in MUNICIPALITY_COORDS.items():
            if municipality == HUB_MUNICIPALITY:
                plt.plot(coords[1], coords[0], 'gs', markersize=10)
                plt.text(coords[1], coords[0] + 0.02, f"{municipality} (HUB)", fontsize=9, ha='center')
            else:
                plt.plot(coords[1], coords[0], 'bo', markersize=8)
                plt.text(coords[1], coords[0] + 0.02, municipality, fontsize=8, ha='center')
        
        plt.plot(PORT_SANTOS_COORDS[1], PORT_SANTOS_COORDS[0], 'r*', markersize=15)
        plt.text(PORT_SANTOS_COORDS[1], PORT_SANTOS_COORDS[0] + 0.02, 'Port Santos', fontsize=10, ha='center')
        
        # Plot a summary with total distances
        plt.title('Coffee Collective Logistics Scenarios Comparison', fontsize=14)
        plt.figtext(0.5, 0.01, f"Scenario 1: {s1_distance:.1f} km | Scenario 2: {s2_distance:.1f} km | VRP: {vrp_distance:.1f} km", 
                    ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # Save combined visualization
        plt.tight_layout()
        plt.savefig(combined_viz_path, dpi=300)
        plt.close()
        print(f"Saved combined visualization to {combined_viz_path}")
        
    else:
        print("Failed to find optimal VRP solution.")

if __name__ == "__main__":
    main()
