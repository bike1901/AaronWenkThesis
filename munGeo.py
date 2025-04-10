import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from haversine import haversine, Unit
from itertools import combinations
from collections import defaultdict
import folium
from folium.plugins import MarkerCluster
import time
import os
import json

# Constants
BRAZIL_DATASET = 'brazil-coffee-v2.5.1-2024-04-26.csv'
DISTANCE_THRESHOLD_KM = 50  # Maximum distance between municipalities to consider for collectives
MIN_SHARED_EXPORTERS = 1  # Minimum number of shared exporters to consider for a collective
GEOCACHE_FILE = 'municipality_geocache.json'  # Cache file to avoid repeated geocoding requests
MUNICIPALITY_MAPPING_FILE = 'brazil_municipalities_mapping.json'  # IBGE municipality mapping file
OUTPUT_DIR = 'collective_analysis'

def load_and_preprocess_data():
    """Load the Brazil coffee dataset and preprocess for analysis."""
    print(f"Loading Brazil coffee dataset: {BRAZIL_DATASET}")
    
    # Load the dataset
    try:
        df = pd.read_csv(BRAZIL_DATASET)
        print(f"Loaded dataset with {len(df)} records")
    except FileNotFoundError:
        print(f"Error: Dataset file {BRAZIL_DATASET} not found.")
        exit(1)
    
    # Basic preprocessing
    print("Preprocessing data...")
    
    # Check if municipality and exporter columns exist
    required_columns = ['municipality', 'exporter']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        # Try to find alternative column names
        possible_alternatives = {
            'municipality': ['municipality_of_origin', 'city_of_origin', 'city', 'origin'],
            'exporter': ['export_company', 'company', 'trader']
        }
        
        for missing_col in missing_columns:
            found = False
            for alt in possible_alternatives.get(missing_col, []):
                if alt in df.columns:
                    print(f"Using '{alt}' column instead of '{missing_col}'")
                    df[missing_col] = df[alt]
                    found = True
                    break
            
            if not found:
                print(f"Error: Required column '{missing_col}' not found in dataset.")
                exit(1)
    
    # Clean municipality names - standardize format
    df['municipality'] = df['municipality'].str.strip().str.upper()
    
    # Add state/region info to municipality names if available
    if 'state' in df.columns:
        # Append state to municipality to help with geocoding accuracy
        df['municipality_full'] = df['municipality'] + ', ' + df['state'] + ', Brazil'
    else:
        df['municipality_full'] = df['municipality'] + ', Brazil'
    
    # Keep only necessary columns
    essential_cols = ['municipality', 'municipality_full', 'exporter']
    
    # Add municipality_trase_id if available
    if 'municipality_trase_id' in df.columns:
        essential_cols.append('municipality_trase_id')
        print("Found municipality_trase_id column - will use for geocoding")
    
    additional_cols = ['volume', 'fob', 'state', 'certification'] 
    keep_cols = essential_cols + [col for col in additional_cols if col in df.columns]
    
    return df[keep_cols]

def geocode_municipalities(df, verify_ssl=True):
    """Geocode municipalities using IBGE data and fall back to online geocoding if needed."""
    print("Geocoding municipalities...")
    
    # Load the IBGE municipality mapping file if it exists
    ibge_mapping = {}
    if os.path.exists(MUNICIPALITY_MAPPING_FILE):
        try:
            with open(MUNICIPALITY_MAPPING_FILE, 'r') as f:
                ibge_mapping = json.load(f)
            print(f"Loaded {len(ibge_mapping)} municipality mappings from IBGE data")
        except Exception as e:
            print(f"Error loading IBGE municipality mapping: {e}")
    else:
        print(f"IBGE municipality mapping file not found: {MUNICIPALITY_MAPPING_FILE}")
        print("Will rely on cached geocoding data and online geocoding")
    
    # First try loading from the cache
    geocache = {}
    if os.path.exists(GEOCACHE_FILE):
        try:
            with open(GEOCACHE_FILE, 'r') as f:
                geocache = json.load(f)
            print(f"Loaded {len(geocache)} cached municipality locations")
        except Exception as e:
            print(f"Error loading geocache: {e}")
    
    # Get unique municipalities to geocode
    unique_municipalities = df[['municipality', 'municipality_full']].drop_duplicates()
    
    # Add trase_id to unique_municipalities if available
    if 'municipality_trase_id' in df.columns:
        # Join the trase_id to the unique_municipalities dataframe
        trase_ids = df[['municipality', 'municipality_trase_id']].drop_duplicates()
        unique_municipalities = unique_municipalities.merge(
            trase_ids, on='municipality', how='left'
        )
        print(f"Added trase_ids to {len(unique_municipalities)} unique municipalities")
    
    # Identify municipalities that need geocoding
    municipalities_to_geocode = [
        m for _, m in unique_municipalities.iterrows() 
        if m['municipality_full'] not in geocache
    ]
    
    if municipalities_to_geocode:
        print(f"Geocoding {len(municipalities_to_geocode)} new municipalities...")
        
        # Try geocoding using IBGE data first
        if ibge_mapping:
            print("Using IBGE municipality mapping data...")
            
            for municipality_data in municipalities_to_geocode:
                municipality_full = municipality_data['municipality_full']
                
                # 1. Try using trase_id if available
                if 'municipality_trase_id' in municipality_data and municipality_data['municipality_trase_id']:
                    trase_id = municipality_data['municipality_trase_id']
                    if trase_id in ibge_mapping:
                        coords = ibge_mapping[trase_id]
                        geocache[municipality_full] = (coords['latitude'], coords['longitude'])
                        print(f"Geocoded using trase_id: {municipality_full} -> {geocache[municipality_full]}")
                        continue
                
                # 2. Try using municipality name
                municipality_name = municipality_data['municipality']
                if municipality_name in ibge_mapping:
                    coords = ibge_mapping[municipality_name]
                    geocache[municipality_full] = (coords['latitude'], coords['longitude'])
                    print(f"Geocoded using municipality name: {municipality_full} -> {geocache[municipality_full]}")
                    continue
                
                # 3. Try fuzzy matching with municipality name
                best_match = None
                for key in ibge_mapping.keys():
                    # Skip trase_ids and numeric codes
                    if key.startswith('BR-') or key.isdigit():
                        continue
                    # Check if either string contains the other
                    if municipality_name in key or key in municipality_name:
                        best_match = key
                        break
                
                if best_match:
                    coords = ibge_mapping[best_match]
                    geocache[municipality_full] = (coords['latitude'], coords['longitude'])
                    print(f"Fuzzy matched: {municipality_full} -> {best_match} -> {geocache[municipality_full]}")
                    continue
                
                # If we get here, we couldn't find a match in IBGE data
                print(f"Could not find in IBGE data: {municipality_full}")
                geocache[municipality_full] = None
                
            # Save geocache after processing with IBGE data
            with open(GEOCACHE_FILE, 'w') as f:
                json.dump(geocache, f)
            print(f"Saved geocache with {len(geocache)} entries after IBGE processing")
        
        # Fall back to online geocoding for any remaining municipalities
        missing_municipalities = [
            m['municipality_full'] for _, m in unique_municipalities.iterrows() 
            if m['municipality_full'] not in geocache or geocache[m['municipality_full']] is None
        ]
        
        if missing_municipalities and verify_ssl:
            print(f"Trying online geocoding for {len(missing_municipalities)} municipalities...")
            try_online_geocoding(missing_municipalities, geocache, verify_ssl)
        
        # Save final geocache
        with open(GEOCACHE_FILE, 'w') as f:
            json.dump(geocache, f)
    
    # Create a dictionary mapping municipality to coordinates
    municipality_locations = {}
    for _, row in unique_municipalities.iterrows():
        coords = geocache.get(row['municipality_full'])
        if coords:
            municipality_locations[row['municipality']] = coords
    
    print(f"Successfully geocoded {len(municipality_locations)} of {len(unique_municipalities)} municipalities")
    return municipality_locations

def create_brazil_municipality_database():
    """Create a database of Brazilian municipalities with coordinates."""
    print("Creating a database of Brazilian municipalities...")
    
    # Hard-coded list of major Brazilian coffee-producing municipalities with coordinates
    # This is a starter list that can be expanded
    municipalities = [
        {"name": "ARAGUARI", "state": "MINAS GERAIS", "latitude": -18.6456, "longitude": -48.1934},
        {"name": "PATROCINIO", "state": "MINAS GERAIS", "latitude": -18.9379, "longitude": -46.9934},
        {"name": "MONTE CARMELO", "state": "MINAS GERAIS", "latitude": -18.7278, "longitude": -47.4983},
        {"name": "SAO GOTARDO", "state": "MINAS GERAIS", "latitude": -19.3089, "longitude": -46.0465},
        {"name": "CAMPOS ALTOS", "state": "MINAS GERAIS", "latitude": -19.6916, "longitude": -46.1727},
        {"name": "CARMO DO PARANAIBA", "state": "MINAS GERAIS", "latitude": -18.9911, "longitude": -46.3124},
        {"name": "IBIA", "state": "MINAS GERAIS", "latitude": -19.4747, "longitude": -46.5465},
        {"name": "MANHUACU", "state": "MINAS GERAIS", "latitude": -20.2572, "longitude": -42.0281},
        {"name": "TRES PONTAS", "state": "MINAS GERAIS", "latitude": -21.3692, "longitude": -45.5093},
        {"name": "NEPOMUCENO", "state": "MINAS GERAIS", "latitude": -21.2324, "longitude": -45.2352},
        {"name": "VARGINHA", "state": "MINAS GERAIS", "latitude": -21.5513, "longitude": -45.4295},
        {"name": "GUAXUPE", "state": "MINAS GERAIS", "latitude": -21.3049, "longitude": -46.7081},
        {"name": "SAO SEBASTIAO DO PARAISO", "state": "MINAS GERAIS", "latitude": -20.9167, "longitude": -46.9833},
        {"name": "ALFENAS", "state": "MINAS GERAIS", "latitude": -21.4291, "longitude": -45.9474},
        {"name": "COROMANDEL", "state": "MINAS GERAIS", "latitude": -18.4734, "longitude": -47.2001},
        {"name": "FRANCA", "state": "SAO PAULO", "latitude": -20.5386, "longitude": -47.4008},
        {"name": "CRISTAIS PAULISTA", "state": "SAO PAULO", "latitude": -20.4036, "longitude": -47.4211},
        {"name": "PEDREGULHO", "state": "SAO PAULO", "latitude": -20.2533, "longitude": -47.4775},
        {"name": "ESPIRITO SANTO DO PINHAL", "state": "SAO PAULO", "latitude": -22.1911, "longitude": -46.7477},
        {"name": "MARILIA", "state": "SAO PAULO", "latitude": -22.2171, "longitude": -49.9501},
        {"name": "GARÇA", "state": "SAO PAULO", "latitude": -22.2128, "longitude": -49.6541},
        {"name": "LINHARES", "state": "ESPIRITO SANTO", "latitude": -19.3946, "longitude": -40.0643},
        {"name": "COLATINA", "state": "ESPIRITO SANTO", "latitude": -19.5399, "longitude": -40.6272},
        {"name": "NOVA VENECIA", "state": "ESPIRITO SANTO", "latitude": -18.7153, "longitude": -40.4053}
    ]
    
    # Convert to DataFrame and save
    import pandas as pd
    db = pd.DataFrame(municipalities)
    db.to_csv('brazil_municipalities.csv', index=False)
    print("Created starter database with 24 major coffee-producing municipalities")
    
    # You could add code here to scrape more municipalities from Wikipedia or other sources
    # Or even download from a public dataset if available
    
    print("Municipality database created successfully.")
    return True

def try_online_geocoding(municipalities, geocache, verify_ssl=True):
    """Try to geocode municipalities using online services as a fallback."""
    try:
        # If SSL verification is explicitly disabled
        if not verify_ssl:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            print("SSL verification explicitly disabled for online geocoding.")
        
        # Initialize geocoder with rate limiting
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
        
        geolocator = Nominatim(user_agent="coffee_municipalities_geocoder")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        # Try a test geocode
        test_location = geocode("Sao Paulo, Brazil")
        if test_location:
            print("Online geocoding connection successful")
            
            for municipality_full in municipalities:
                try:
                    location = geocode(municipality_full)
                    
                    if location:
                        geocache[municipality_full] = (location.latitude, location.longitude)
                        print(f"Online geocoded: {municipality_full} -> {geocache[municipality_full]}")
                    else:
                        # Try with just municipality name and Brazil
                        municipality_simple = municipality_full.split(',')[0].strip() + ", Brazil"
                        location = geocode(municipality_simple)
                        
                        if location:
                            geocache[municipality_full] = (location.latitude, location.longitude)
                            print(f"Online geocoded with simplified name: {municipality_full} -> {geocache[municipality_full]}")
                        else:
                            print(f"Could not geocode online: {municipality_full}")
                            geocache[municipality_full] = None
                    
                    # Save periodically
                    if len(geocache) % 5 == 0:
                        with open(GEOCACHE_FILE, 'w') as f:
                            json.dump(geocache, f)
                    
                    # Be nice to the API
                    time.sleep(1)
                
                except Exception as e:
                    print(f"Error geocoding {municipality_full}: {e}")
                    geocache[municipality_full] = None
        else:
            print("Test geocoding failed, online geocoding not available")
    
    except Exception as e:
        print(f"Error setting up online geocoder: {e}")
        print("Online geocoding failed. Using only local database results.")

def find_nearby_municipalities(locations):
    """Find pairs of municipalities that are within the distance threshold."""
    print(f"Finding nearby municipality pairs (within {DISTANCE_THRESHOLD_KM} km)...")
    
    nearby_pairs = []
    municipalities = list(locations.keys())
    total_comparisons = len(municipalities) * (len(municipalities) - 1) // 2
    
    print(f"Comparing {total_comparisons} municipality pairs...")
    
    for i, (mun1, mun2) in enumerate(combinations(municipalities, 2)):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{total_comparisons} pairs...")
            
        if locations[mun1] and locations[mun2]:  # Both municipalities have valid coordinates
            try:
                dist = haversine(locations[mun1], locations[mun2], unit=Unit.KILOMETERS)
                if dist <= DISTANCE_THRESHOLD_KM:
                    nearby_pairs.append((mun1, mun2, dist))
            except Exception as e:
                print(f"Error calculating distance between {mun1} and {mun2}: {e}")
    
    print(f"Found {len(nearby_pairs)} nearby municipality pairs")
    return nearby_pairs

def analyze_shared_exporters(df, nearby_pairs):
    """Analyze which nearby municipalities share exporters."""
    print("Analyzing shared exporters between nearby municipalities...")
    
    # Create dictionary mapping municipalities to their exporters
    mun_exporters = df.groupby('municipality')['exporter'].apply(set).to_dict()
    
    # Calculate export volumes by municipality if volume data is available
    if 'volume' in df.columns:
        mun_volumes = df.groupby('municipality')['volume'].sum().to_dict()
    else:
        mun_volumes = {mun: 1 for mun in mun_exporters.keys()}  # Default to 1 if no volume data
    
    # Find shared exporters for each nearby pair
    collective_suggestions = []
    
    for mun1, mun2, dist in nearby_pairs:
        if mun1 in mun_exporters and mun2 in mun_exporters:
            shared_exporters = mun_exporters[mun1].intersection(mun_exporters[mun2])
            
            if len(shared_exporters) >= MIN_SHARED_EXPORTERS:
                # Calculate the combined volume
                combined_volume = mun_volumes.get(mun1, 0) + mun_volumes.get(mun2, 0)
                
                # Calculate export diversity (number of unique exporters)
                total_exporters = len(mun_exporters[mun1].union(mun_exporters[mun2]))
                
                collective_suggestions.append({
                    'municipalities': [mun1, mun2],
                    'distance_km': dist,
                    'shared_exporters': list(shared_exporters),
                    'num_shared_exporters': len(shared_exporters),
                    'total_exporters': total_exporters,
                    'combined_volume': combined_volume
                })
    
    # Sort suggestions by number of shared exporters (primary) and distance (secondary)
    collective_suggestions.sort(key=lambda x: (x['num_shared_exporters'], -x['distance_km']), reverse=True)
    
    print(f"Found {len(collective_suggestions)} potential farming collectives")
    return collective_suggestions

def expand_collectives(suggestions, locations, threshold=DISTANCE_THRESHOLD_KM):
    """Expand collective suggestions to include more than two municipalities when possible."""
    print("Expanding collective suggestions to include more municipalities...")
    
    # Create a graph of connected municipalities
    connections = defaultdict(set)
    for suggestion in suggestions:
        mun1, mun2 = suggestion['municipalities']
        connections[mun1].add(mun2)
        connections[mun2].add(mun1)
    
    # Find larger collectives using a greedy approach
    visited = set()
    expanded_collectives = []
    
    # Sort municipalities by their number of connections (most connected first)
    sorted_municipalities = sorted(connections.keys(), 
                                   key=lambda m: len(connections[m]), 
                                   reverse=True)
    
    for start_mun in sorted_municipalities:
        if start_mun in visited:
            continue
            
        # Start a new collective
        collective = {start_mun}
        visited.add(start_mun)
        
        # Find nearby connected municipalities
        candidates = connections[start_mun].copy()
        
        # Greedily add municipalities that are within threshold of all existing members
        while candidates:
            next_mun = candidates.pop()
            
            if next_mun in visited:
                continue
                
            # Check if the candidate is within threshold of all existing members
            can_add = True
            for mun in collective:
                if mun != next_mun:
                    try:
                        dist = haversine(locations[mun], locations[next_mun], unit=Unit.KILOMETERS)
                        if dist > threshold:
                            can_add = False
                            break
                    except:
                        can_add = False
                        break
            
            if can_add:
                collective.add(next_mun)
                visited.add(next_mun)
                
                # Add new candidates from this municipality
                for new_candidate in connections[next_mun]:
                    if new_candidate not in visited and new_candidate not in candidates:
                        candidates.add(new_candidate)
        
        if len(collective) > 1:
            expanded_collectives.append(list(collective))
    
    print(f"Created {len(expanded_collectives)} expanded collectives")
    
    # Calculate collective stats
    final_collectives = []
    for municipalities in expanded_collectives:
        if len(municipalities) >= 2:
            # Calculate maximum distance between any two municipalities
            max_dist = 0
            for i, mun1 in enumerate(municipalities):
                for mun2 in municipalities[i+1:]:
                    try:
                        dist = haversine(locations[mun1], locations[mun2], unit=Unit.KILOMETERS)
                        max_dist = max(max_dist, dist)
                    except:
                        pass
            
            final_collectives.append({
                'municipalities': municipalities,
                'size': len(municipalities),
                'max_distance_km': max_dist
            })
    
    # Sort by size (descending) and max distance (ascending)
    final_collectives.sort(key=lambda x: (x['size'], -x['max_distance_km']), reverse=True)
    
    return final_collectives

def create_collective_visualizations(df, suggestions, expanded_collectives, locations):
    """Create visualizations of the suggested collectives."""
    print("Creating visualizations...")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. Create a map with all municipalities and suggested collectives
    brazil_map = folium.Map(location=[-15.77972, -47.92972], zoom_start=5)  # Center on Brazil
    
    # Add municipality markers
    marker_cluster = MarkerCluster().add_to(brazil_map)
    
    for municipality, coords in locations.items():
        if coords:  # Skip municipalities without coordinates
            folium.Marker(
                location=coords,
                tooltip=municipality,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)
    
    # Add lines for collectives
    for i, suggestion in enumerate(suggestions[:20]):  # Show top 20 pairs
        mun1, mun2 = suggestion['municipalities']
        coords1 = locations.get(mun1)
        coords2 = locations.get(mun2)
        
        if coords1 and coords2:
            folium.PolyLine(
                locations=[coords1, coords2],
                color='red',
                weight=2,
                opacity=0.7,
                tooltip=f"Collective: {mun1} - {mun2} ({suggestion['distance_km']:.1f} km)"
            ).add_to(brazil_map)
    
    # Save the map
    brazil_map.save(os.path.join(OUTPUT_DIR, 'municipality_collectives_map.html'))
    
    # 2. Create a map for expanded collectives
    expanded_map = folium.Map(location=[-15.77972, -47.92972], zoom_start=5)  # Center on Brazil
    
    # Add municipality markers (clustered)
    marker_cluster = MarkerCluster().add_to(expanded_map)
    for municipality, coords in locations.items():
        if coords:
            folium.Marker(
                location=coords,
                tooltip=municipality,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)
    
    # Add polygons for expanded collectives with different colors
    colors = ['red', 'green', 'purple', 'orange', 'darkblue', 'darkred', 'cadetblue', 
              'darkgreen', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black']
    
    for i, collective in enumerate(expanded_collectives[:20]):  # Top 20 collectives
        color = colors[i % len(colors)]
        municipalities = collective['municipalities']
        
        # Get coordinates for all municipalities in the collective
        coords = [locations.get(mun) for mun in municipalities if locations.get(mun)]
        
        if len(coords) >= 3:
            # Create a polygon for collectives with 3+ municipalities
            folium.Polygon(
                locations=coords,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.2,
                tooltip=f"Collective: {', '.join(municipalities)} (Size: {len(municipalities)})"
            ).add_to(expanded_map)
        elif len(coords) == 2:
            # Create a line for collectives with 2 municipalities
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f"Collective: {', '.join(municipalities)}"
            ).add_to(expanded_map)
    
    # Save the expanded collectives map
    expanded_map.save(os.path.join(OUTPUT_DIR, 'expanded_collectives_map.html'))
    
    # 3. Create network graph of municipalities and shared exporters
    try:
        import networkx as nx
        
        # Create network graph
        G = nx.Graph()
        
        # Add municipality nodes
        for municipality in locations.keys():
            G.add_node(municipality, type='municipality')
        
        # Add edges between municipalities that share exporters
        for suggestion in suggestions:
            mun1, mun2 = suggestion['municipalities']
            G.add_edge(mun1, mun2, weight=suggestion['num_shared_exporters'])
        
        # Draw the graph
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw municipality nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='skyblue', 
                              node_size=100, 
                              alpha=0.8)
        
        # Draw edges based on number of shared exporters
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, 
                              width=[w/2 for w in edge_weights], 
                              alpha=0.5, 
                              edge_color='gray')
        
        # Add labels to nodes
        municipality_labels = {node: node.split()[0] for node in G.nodes()}  # First word only for clarity
        nx.draw_networkx_labels(G, pos, 
                               labels=municipality_labels, 
                               font_size=8, 
                               font_family='sans-serif')
        
        plt.title('Municipality Network: Connections represent shared exporters', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'municipality_network.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Note: NetworkX not available for network visualization. Install with 'pip install networkx'.")

def generate_collective_report(suggestions, expanded_collectives, df, locations):
    """Generate a detailed report of collective suggestions."""
    print("Generating collective recommendations report...")
    
    report_file = os.path.join(OUTPUT_DIR, 'collective_recommendations.txt')
    
    with open(report_file, 'w') as f:
        f.write("=== COFFEE FARMING COLLECTIVE RECOMMENDATIONS ===\n\n")
        f.write(f"Analysis based on {BRAZIL_DATASET}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        
        f.write("PARAMETERS:\n")
        f.write(f"- Maximum distance between municipalities: {DISTANCE_THRESHOLD_KM} km\n")
        f.write(f"- Minimum shared exporters: {MIN_SHARED_EXPORTERS}\n\n")
        
        f.write("1. TOP MUNICIPALITY PAIR RECOMMENDATIONS\n")
        f.write("======================================\n\n")
        
        # Write detailed information for top pairs
        for i, suggestion in enumerate(suggestions[:20], 1):
            mun1, mun2 = suggestion['municipalities']
            f.write(f"Recommendation #{i}: {mun1} + {mun2}\n")
            f.write(f"  Distance: {suggestion['distance_km']:.1f} km\n")
            f.write(f"  Shared Exporters: {suggestion['num_shared_exporters']} of {suggestion['total_exporters']} total\n")
            
            # List shared exporters
            f.write(f"  Shared Exporter List: {', '.join(suggestion['shared_exporters'][:5])}")
            if len(suggestion['shared_exporters']) > 5:
                f.write(f" and {len(suggestion['shared_exporters']) - 5} more")
            f.write("\n")
            
            # Add volume information if available
            f.write(f"  Combined Volume: {suggestion['combined_volume']:.2f}\n")
            
            # Calculate potential benefits
            f.write("  Potential Benefits:\n")
            f.write("    - Shared transportation to the same exporters\n")
            f.write("    - Combined volume for better negotiating power\n")
            f.write("    - Knowledge sharing between nearby communities\n")
            
            f.write("\n")
        
        f.write("\n2. EXPANDED COLLECTIVE RECOMMENDATIONS\n")
        f.write("======================================\n\n")
        
        # Write information for expanded collectives
        for i, collective in enumerate(expanded_collectives[:10], 1):
            municipalities = collective['municipalities']
            
            f.write(f"Expanded Collective #{i}\n")
            f.write(f"  Size: {collective['size']} municipalities\n")
            f.write(f"  Municipalities: {', '.join(municipalities)}\n")
            f.write(f"  Maximum Distance: {collective['max_distance_km']:.1f} km\n")
            
            # Calculate total volume if available
            if 'volume' in df.columns:
                total_volume = df[df['municipality'].isin(municipalities)]['volume'].sum()
                f.write(f"  Total Volume: {total_volume:.2f}\n")
            
            # Calculate centroid for suggested central location
            if all(locations.get(mun) for mun in municipalities):
                coords = [locations[mun] for mun in municipalities]
                centroid_lat = sum(c[0] for c in coords) / len(coords)
                centroid_lon = sum(c[1] for c in coords) / len(coords)
                
                # Find municipality closest to centroid
                closest_mun = None
                min_dist = float('inf')
                
                for mun in municipalities:
                    mun_coords = locations[mun]
                    dist = haversine((centroid_lat, centroid_lon), mun_coords, unit=Unit.KILOMETERS)
                    if dist < min_dist:
                        min_dist = dist
                        closest_mun = mun
                
                f.write(f"  Suggested Central Hub: {closest_mun}\n")
            
            f.write("\n")
        
        f.write("\n3. SUMMARY STATISTICS\n")
        f.write("====================\n\n")
        
        f.write(f"Total Municipalities Analyzed: {len(locations)}\n")
        f.write(f"Potential Pair Collectives Identified: {len(suggestions)}\n")
        f.write(f"Expanded Collectives Identified: {len(expanded_collectives)}\n\n")
        
        f.write("The recommended collectives are visualized in the following files:\n")
        f.write("- municipality_collectives_map.html: Interactive map of suggested municipality pairs\n")
        f.write("- expanded_collectives_map.html: Interactive map of expanded collectives\n")
        f.write("- municipality_network.png: Network visualization of municipality connections\n\n")
        
        f.write("These recommendations are based on geographic proximity and shared exporter relationships.\n")
        f.write("Forming these collectives could lead to economies of scale, improved logistics, and better negotiating power.\n")
    
    print(f"Report generated: {report_file}")

def main():
    """Main function to run the complete municipality geolocation and collective recommendation process."""
    print("=== MUNICIPALITY GEOLOCATION AND COLLECTIVE ANALYSIS ===")
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    
    try:
        # Step 2: Geocode municipalities - force SSL verification off on macOS
        import platform
        if platform.system() == 'Darwin':
            print("macOS detected - disabling SSL verification by default.")
            municipality_locations = geocode_municipalities(df, verify_ssl=False)
        else:
            municipality_locations = geocode_municipalities(df)
        
        if not municipality_locations:
            print("Warning: No municipalities were successfully geocoded. Trying again with SSL verification disabled...")
            municipality_locations = geocode_municipalities(df, verify_ssl=False)
            
            if not municipality_locations:
                print("Error: Unable to geocode any municipalities. Analysis cannot continue.")
                print("Please try installing the certifi package: pip install certifi")
                exit(1)
        
        # Step 3: Find nearby municipality pairs
        nearby_pairs = find_nearby_municipalities(municipality_locations)
        
        # Step 4: Analyze shared exporters and create collective suggestions
        collective_suggestions = analyze_shared_exporters(df, nearby_pairs)
        
        # Step 5: Expand collectives to include more municipalities
        expanded_collectives = expand_collectives(collective_suggestions, municipality_locations)
        
        # Step 6: Create visualizations
        create_collective_visualizations(df, collective_suggestions, expanded_collectives, municipality_locations)
        
        # Step 7: Generate comprehensive report
        generate_collective_report(collective_suggestions, expanded_collectives, df, municipality_locations)
        
        # Step 8: Print summary
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Found {len(collective_suggestions)} potential municipality pairs for collectives")
        print(f"Created {len(expanded_collectives)} expanded collectives")
        print(f"Results saved to {OUTPUT_DIR}/ directory")
        print("\nRecommended next steps:")
        print("1. Review the collective_recommendations.txt report")
        print("2. Explore the interactive maps in your browser")
        print("3. Share findings with relevant stakeholders")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("\nSuggestions to fix the issue:")
        print("1. Install certifi package: pip install certifi")
        print("2. On macOS, run the Python certificate install script:")
        print("   /Applications/Python 3.x/Install Certificates.command")
        print("3. If using a virtual environment, make sure it has access to system certificates")
        print("4. Try running the script again")

if __name__ == "__main__":
    main()