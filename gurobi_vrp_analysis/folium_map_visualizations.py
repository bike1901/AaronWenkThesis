"""
Coffee Collective VRP Analysis Map Visualizations

This script creates interactive map-based visualizations for the coffee collective routing
analysis using Folium with OpenStreetMap tiles.

Author: Cascade
"""

import folium
import os
from folium.features import CustomIcon
import branca.colormap as cm

# Define the municipalities and their coordinates
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

# VRP optimal route (from Gurobi solution)
OPTIMAL_ROUTE = [
    'Port Santos', 'MANDURI', 'BERNARDINO DE CAMPOS', 
    'SANTA CRUZ DO RIO PARDO', 'CHAVANTES', 'IPAUSSU', 
    'TIMBURI', 'OLEO', 'Port Santos'
]

def create_base_map():
    """Create a base map centered on the region"""
    # Calculate center point for the map
    all_lats = [coords[0] for coords in MUNICIPALITY_COORDS.values()] + [PORT_SANTOS_COORDS[0]]
    all_lons = [coords[1] for coords in MUNICIPALITY_COORDS.values()] + [PORT_SANTOS_COORDS[1]]
    
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap"
    )
    
    return m

def add_municipalities_to_map(m, highlight_hub=True):
    """Add municipality markers to the map"""
    for municipality, coords in MUNICIPALITY_COORDS.items():
        if municipality == HUB_MUNICIPALITY and highlight_hub:
            # Use a different marker for the hub
            folium.Marker(
                location=coords,
                popup=f"{municipality} (HUB)",
                tooltip=f"{municipality} (HUB)",
                icon=folium.Icon(color="green", icon="star")
            ).add_to(m)
        else:
            folium.Marker(
                location=coords,
                popup=municipality,
                tooltip=municipality,
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
    
    # Add Port Santos
    folium.Marker(
        location=PORT_SANTOS_COORDS,
        popup="Port Santos",
        tooltip="Port Santos",
        icon=folium.Icon(color="red", icon="ship", prefix="fa")
    ).add_to(m)
    
    return m

def scenario1_map(save_path):
    """Create interactive map for Scenario 1: Individual Pickups"""
    m = create_base_map()
    m = add_municipalities_to_map(m, highlight_hub=False)
    
    # Add round-trip lines from Port Santos to each municipality
    for municipality, coords in MUNICIPALITY_COORDS.items():
        folium.PolyLine(
            locations=[PORT_SANTOS_COORDS, coords],
            color="orange",
            weight=2,
            opacity=0.7,
            tooltip=f"Port Santos to {municipality}: Round-trip"
        ).add_to(m)
    
    # Add title
    title_html = '''
    <h3 align="center" style="font-size:16px"><b>Scenario 1: Individual Pickups (Direct to Port Santos)</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    m.save(save_path)
    print(f"Saved interactive map to {save_path}")
    
    return m

def scenario2_map(save_path):
    """Create interactive map for Scenario 2: Hub-based Collection"""
    m = create_base_map()
    m = add_municipalities_to_map(m, highlight_hub=True)
    
    # Get hub coordinates
    hub_coords = MUNICIPALITY_COORDS[HUB_MUNICIPALITY]
    
    # Add lines from municipalities to hub
    for municipality, coords in MUNICIPALITY_COORDS.items():
        if municipality != HUB_MUNICIPALITY:
            folium.PolyLine(
                locations=[hub_coords, coords],
                color="blue",
                weight=2,
                opacity=0.7,
                tooltip=f"{municipality} to {HUB_MUNICIPALITY}"
            ).add_to(m)
    
    # Add round-trip line from hub to Port Santos
    folium.PolyLine(
        locations=[hub_coords, PORT_SANTOS_COORDS],
        color="red",
        weight=3,
        opacity=0.8,
        tooltip=f"{HUB_MUNICIPALITY} to Port Santos (Round-trip)"
    ).add_to(m)
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:16px"><b>Scenario 2: Hub-based Collection (Hub: {HUB_MUNICIPALITY})</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    m.save(save_path)
    print(f"Saved interactive map to {save_path}")
    
    return m

def vrp_solution_map(save_path):
    """Create interactive map for the Optimal VRP Solution"""
    m = create_base_map()
    m = add_municipalities_to_map(m, highlight_hub=False)
    
    # Create a color for each segment of the route
    colors = cm.linear.YlOrRd_09.scale(0, len(OPTIMAL_ROUTE) - 2)
    
    # Map location names to coordinates
    location_to_coords = {
        'Port Santos': PORT_SANTOS_COORDS,
        **MUNICIPALITY_COORDS
    }
    
    # Create route segments with directions and labels
    for i in range(len(OPTIMAL_ROUTE) - 1):
        start_loc = OPTIMAL_ROUTE[i]
        end_loc = OPTIMAL_ROUTE[i + 1]
        start_coords = location_to_coords[start_loc]
        end_coords = location_to_coords[end_loc]
        
        # Add a routing line
        folium.PolyLine(
            locations=[start_coords, end_coords],
            color=colors(i),
            weight=4,
            opacity=0.8,
            tooltip=f"Segment {i+1}: {start_loc} to {end_loc}"
        ).add_to(m)
        
        # Add a directional marker
        folium.RegularPolygonMarker(
            location=[(start_coords[0] + end_coords[0])/2, (start_coords[1] + end_coords[1])/2],
            number_of_sides=3,
            rotation=45,
            radius=8,
            color=colors(i),
            fill=True,
            fill_color=colors(i),
            fill_opacity=0.8,
            popup=f"Segment {i+1}: {start_loc} to {end_loc}"
        ).add_to(m)
    
    # Add numbers to show the sequence
    for i, loc in enumerate(OPTIMAL_ROUTE):
        coords = location_to_coords[loc]
        folium.CircleMarker(
            location=coords,
            radius=10,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=0.8,
            tooltip=f"Stop {i}: {loc}"
        ).add_to(m)
        
        folium.map.Marker(
            coords,
            icon=folium.DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=f'<div style="font-size: 10pt; font-weight: bold; text-align: center;">{i}</div>'
            )
        ).add_to(m)
    
    # Add title
    title_html = '''
    <h3 align="center" style="font-size:16px"><b>Optimal VRP Solution (Gurobi)</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    m.save(save_path)
    print(f"Saved interactive map to {save_path}")
    
    return m

def create_maps():
    """Create all map visualizations"""
    # Create visualization output directory if it doesn't exist
    viz_dir = "/Users/aaronwenk/Desktop/CoffeeOptimization/gurobi_vrp_analysis"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        print(f"Created visualization directory: {viz_dir}")
    
    # Generate maps for each scenario
    s1_map_path = os.path.join(viz_dir, "scenario1_map.html")
    s2_map_path = os.path.join(viz_dir, "scenario2_map.html")
    vrp_map_path = os.path.join(viz_dir, "optimal_vrp_map.html")
    
    scenario1_map(s1_map_path)
    scenario2_map(s2_map_path)
    vrp_solution_map(vrp_map_path)
    
    # Create comparison dashboard
    comparison_path = os.path.join(viz_dir, "comparison_dashboard.html")
    
    # Simple summary statistics
    s1_distance = 4831.51
    s2_distance = 832.25
    vrp_distance = 756.42
    
    with open(comparison_path, 'w') as f:
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Coffee Collective VRP Analysis - Map Comparison</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f7f7f7;
                }}
                h1, h2 {{
                    color: #333;
                    text-align: center;
                }}
                .container {{
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .dashboard {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                .stats-container {{
                    display: flex;
                    justify-content: space-around;
                    margin-bottom: 20px;
                }}
                .stat-card {{
                    text-align: center;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
                    flex: 1;
                    margin: 0 10px;
                }}
                .maps-container {{
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 20px;
                }}
                .map-card {{
                    display: flex;
                    flex-direction: column;
                    background-color: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
                }}
                .map-card h3 {{
                    margin: 0;
                    padding: 15px;
                    background-color: #4285f4;
                    color: white;
                }}
                iframe {{
                    border: none;
                    width: 100%;
                    height: 500px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="dashboard">
                    <h1>Coffee Collective VRP Analysis</h1>
                    <h2>Logistics Scenario Comparison</h2>
                    
                    <div class="stats-container">
                        <div class="stat-card" style="background-color: #ffcccc;">
                            <h3>Scenario 1</h3>
                            <p>Individual Pickups</p>
                            <h2>{s1_distance:.2f} km</h2>
                        </div>
                        <div class="stat-card" style="background-color: #ccffcc;">
                            <h3>Scenario 2</h3>
                            <p>Hub-based Collection</p>
                            <h2>{s2_distance:.2f} km</h2>
                            <p>Savings: {(s1_distance - s2_distance):.2f} km ({((s1_distance - s2_distance) / s1_distance * 100):.2f}%)</p>
                        </div>
                        <div class="stat-card" style="background-color: #ccccff;">
                            <h3>Optimal VRP</h3>
                            <p>Gurobi Solution</p>
                            <h2>{vrp_distance:.2f} km</h2>
                            <p>Savings: {(s1_distance - vrp_distance):.2f} km ({((s1_distance - vrp_distance) / s1_distance * 100):.2f}%)</p>
                        </div>
                    </div>
                </div>
                
                <div class="maps-container">
                    <div class="map-card">
                        <h3>Scenario 1: Individual Pickups</h3>
                        <iframe src="scenario1_map.html"></iframe>
                    </div>
                    
                    <div class="map-card">
                        <h3>Scenario 2: Hub-based Collection</h3>
                        <iframe src="scenario2_map.html"></iframe>
                    </div>
                    
                    <div class="map-card">
                        <h3>Optimal VRP Solution</h3>
                        <iframe src="optimal_vrp_map.html"></iframe>
                    </div>
                </div>
            </div>
        </body>
        </html>
        ''')
    
    print(f"Saved comparison dashboard to {comparison_path}")

if __name__ == "__main__":
    create_maps()