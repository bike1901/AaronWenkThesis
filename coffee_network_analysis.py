#!/usr/bin/env python3
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
try:
    import community as community_louvain  # may need to install python-louvain
except ImportError:
    print("Please install python-louvain: pip install python-louvain")
from networkx.algorithms import bipartite
import warnings
warnings.filterwarnings('ignore')

# Define file paths
DATA_FILE = 'brazil-coffee-v2.5.1-2024-04-26.csv'  # Changed to Brazilian dataset
OUTPUT_DIR = 'analysis_results'

def load_data():
    """Load and preprocess the coffee trade data."""
    print("Loading coffee trade data...")
    df = pd.read_csv(DATA_FILE)
    
    # Map 'municipality' to 'municipality_of_export' for Brazilian data
    if 'municipality_of_export' not in df.columns and 'municipality' in df.columns:
        df['municipality_of_export'] = df['municipality']
    
    # Basic preprocessing
    # Fill missing values appropriately
    # For simplification, we'll group unnamed exporters and importers
    df['exporter'] = df['exporter'].fillna('Unknown Exporter')
    df['importer'] = df['importer'].fillna('Unknown Importer')
    df['exporter_group'] = df['exporter_group'].fillna('Unknown Group')
    df['importer_group'] = df['importer_group'].fillna('Unknown Group')
    df['municipality_of_export'] = df['municipality_of_export'].fillna('Unknown Municipality')
    
    # Filter out rows with missing critical values
    df = df.dropna(subset=['volume'])
    
    # Convert volume to numeric if it's not already
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['fob'] = pd.to_numeric(df['fob'], errors='coerce')
    
    # Calculate FOB per unit if not present
    if 'fob_per_unit' not in df.columns:
        df['fob_per_unit'] = df['fob'] / df['volume'].where(df['volume'] > 0, np.nan)
    
    return df

def create_multilayer_network(df):
    """
    Create a multi-layer supply chain network with:
    Layer 1: Municipalities (supply)
    Layer 2: Exporters (intermediaries)
    Layer 3: Importers (demand)
    Layer 4: Destination countries (final markets)
    
    Returns:
    - G: A directed network with all layers
    - node_layers: Dictionary mapping nodes to their layer
    - path_weights: Dictionary with weights for each complete path
    """
    print("Creating multi-layer supply chain network...")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Track which layer each node belongs to
    node_layers = {}
    
    # Dictionary to track complete paths and their weights
    path_weights = defaultdict(lambda: {'volume': 0, 'fob': 0})
    
    # Process each trade record
    for idx, row in df.iterrows():
        # Skip if any essential component is missing
        if pd.isna(row['municipality_of_export']) or pd.isna(row['exporter']) or \
           pd.isna(row['importer']) or pd.isna(row['country_of_destination']):
            continue
            
        # Identify nodes in each layer
        municipality = f"M:{row['municipality_of_export']}"
        exporter = f"E:{row['exporter']}"
        importer = f"I:{row['importer']}"
        destination = f"D:{row['country_of_destination']}"
        
        # Add nodes to graph if they don't exist
        for node, layer in [(municipality, 1), (exporter, 2), (importer, 3), (destination, 4)]:
            if node not in G:
                G.add_node(node)
                node_layers[node] = layer
        
        # Add edges with weights
        volume = row['volume']
        fob = row['fob'] if not pd.isna(row['fob']) else 0
                
        # Add edges between layers
        G.add_edge(municipality, exporter, weight=volume, fob=fob)
        G.add_edge(exporter, importer, weight=volume, fob=fob)
        G.add_edge(importer, destination, weight=volume, fob=fob)
        
        # Track complete path weights
        path_key = (municipality, exporter, importer, destination)
        path_weights[path_key]['volume'] += volume
        path_weights[path_key]['fob'] += fob
    
    print(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, node_layers, path_weights

def analyze_dominant_paths(G, path_weights):
    """Identify and analyze dominant trade paths in the network."""
    print("Analyzing dominant trade paths...")
    
    # Sort paths by volume and FOB (still needed for later analysis)
    paths_by_volume = sorted(path_weights.items(), key=lambda x: x[1]['volume'], reverse=True)
    paths_by_fob = sorted(path_weights.items(), key=lambda x: x[1]['fob'], reverse=True)
    
    # We're removing reporting of top paths by volume and FOB as they highlight outliers
    # with limited improvement potential
    
    # Calculate and return FOB per tonne for each path where possible
    path_fob_per_tonne = {}
    for path, weights in path_weights.items():
        if weights['volume'] > 0 and weights['fob'] > 0:
            path_fob_per_tonne[path] = weights['fob'] / weights['volume']
    
    # Sort by FOB per tonne (still needed for later analysis)
    paths_by_fob_per_tonne = sorted(path_fob_per_tonne.items(), key=lambda x: x[1], reverse=True)
    
    # REMOVED: Inaccurate market segmentation analysis based on percentiles
    # Instead, use fixed price thresholds based on industry standards
    
    # Simple placeholder for compatibility with rest of the code
    # Create segments with fixed price thresholds
    market_segments = {
        'commodity': (0, 3000),           # Standard commodity coffee
        'premium': (3000, 4500),          # Premium commercial coffee
        'commercial_specialty': (4500, 6000), # Lower specialty coffee (80-84 pts)
        'high_specialty': (6000, 600000)      # High specialty (85+ pts)
    }
    
    # Categorize paths into segments
    segment_paths = {
        'commodity': [],
        'premium': [],
        'commercial_specialty': [],
        'high_specialty': []
    }
    
    segment_volumes = {segment: 0 for segment in segment_paths}
    segment_revenues = {segment: 0 for segment in segment_paths}
    
    # Track municipalities, exporters, importers, and destinations in each segment
    segment_municipalities = {segment: set() for segment in segment_paths}
    segment_exporters = {segment: set() for segment in segment_paths}
    segment_importers = {segment: set() for segment in segment_paths}
    segment_destinations = {segment: set() for segment in segment_paths}
    
    # Track municipality segment participation and average prices
    muni_segment_participation = defaultdict(lambda: defaultdict(float))  # muni -> segment -> volume
    muni_segment_revenues = defaultdict(lambda: defaultdict(float))      # muni -> segment -> fob
    
    for path, fob_pt in path_fob_per_tonne.items():
        weights = path_weights[path]
        muni, exp, imp, dest = path
        
        # Determine which segment this path belongs to
        if fob_pt <= market_segments['commodity'][1]:
            segment = 'commodity'
        elif fob_pt <= market_segments['premium'][1]:
            segment = 'premium'
        elif fob_pt <= market_segments['commercial_specialty'][1]:
            segment = 'commercial_specialty'
        else:
            segment = 'high_specialty'
        
        segment_paths[segment].append((path, fob_pt))
        segment_volumes[segment] += weights['volume']
        segment_revenues[segment] += weights['fob']
        
        # Track entities involved in each segment
        segment_municipalities[segment].add(muni)
        segment_exporters[segment].add(exp)
        segment_importers[segment].add(imp)
        segment_destinations[segment].add(dest)
        
        # Track municipality participation in each segment
        muni_segment_participation[muni][segment] += weights['volume']
        muni_segment_revenues[muni][segment] += weights['fob']
    
    # Calculate municipality dominant segment and average price by segment (moved outside the loop)
    muni_dominant_segment = {}
    muni_avg_prices_by_segment = defaultdict(dict)
    
    for muni in muni_segment_participation:
        # Determine dominant segment for this municipality
        dominant_segment = max(muni_segment_participation[muni].items(), 
                              key=lambda x: x[1], default=(None, 0))[0]
        if dominant_segment:
            muni_dominant_segment[muni] = dominant_segment
        
        # Calculate average prices in each segment
        for segment, volume in muni_segment_participation[muni].items():
            if volume > 0 and segment in muni_segment_revenues[muni]:
                muni_avg_prices_by_segment[muni][segment] = muni_segment_revenues[muni][segment] / volume
        
    # Report market segmentation
    print("\n== Coffee Market Segmentation Analysis ==")
    print(f"Segmentation based on FOB per tonne price ranges:")
    
    total_volume = sum(segment_volumes.values())
    total_revenue = sum(segment_revenues.values())
    
    print(f"== Note on Coffee Market Segmentation ==")
    print(f"Market segmentation has been modified to use fixed price thresholds")
    print(f"instead of percentile-based thresholds to better reflect industry standards.\n")
    
    print(f"Fixed price thresholds used:")
    print(f"  Commodity: ${market_segments['commodity'][0]:.2f} - ${market_segments['commodity'][1]:.2f}/tonne")
    print(f"  Premium: ${market_segments['premium'][0]:.2f} - ${market_segments['premium'][1]:.2f}/tonne")
    print(f"  Commercial Specialty: ${market_segments['commercial_specialty'][0]:.2f} - ${market_segments['commercial_specialty'][1]:.2f}/tonne")
    print(f"  High Specialty: ${market_segments['high_specialty'][0]:.2f} - ${market_segments['high_specialty'][1]:.2f}/tonne\n")
    
    # Analyze municipality participation across segments
    print("\n== Municipality Market Segment Participation ==")
        
    # Calculate how many municipalities participate predominantly in each segment
    segment_dominant_counts = defaultdict(int)
    for muni, segment in muni_dominant_segment.items():
        segment_dominant_counts[segment] += 1
    
    print("\nMunicipality dominant segment distribution:")
    for segment, count in segment_dominant_counts.items():
        pct = (count / len(muni_dominant_segment) * 100) if muni_dominant_segment else 0
        print(f"  {segment.replace('_', ' ').title()}: {count} municipalities ({pct:.1f}%)")
        
    # Identify municipalities that are exclusively in commodity segment
    commodity_only_munis = [muni for muni in muni_segment_participation 
                          if len(muni_segment_participation[muni]) == 1 
                          and 'commodity' in muni_segment_participation[muni]]
    
    # Identify municipalities with presence in multiple segments
    multi_segment_munis = [muni for muni in muni_segment_participation 
                         if len(muni_segment_participation[muni]) > 1]
    
    # Calculate percentage of municipalities stuck in commodity-only vs. those with diversification
    total_munis = len(muni_segment_participation)
    commodity_only_pct = (len(commodity_only_munis) / total_munis * 100) if total_munis > 0 else 0
    multi_segment_pct = (len(multi_segment_munis) / total_munis * 100) if total_munis > 0 else 0
        
    print(f"\nMarket diversification analysis:")
    print(f"  Municipalities selling exclusively commodity coffee: {len(commodity_only_munis)} ({commodity_only_pct:.1f}%)")
    print(f"  Municipalities selling in multiple market segments: {len(multi_segment_munis)} ({multi_segment_pct:.1f}%)")
        
    # Analyze price differences by segment for multi-segment municipalities
    if multi_segment_munis:
        avg_premium_pct = []
        for muni in multi_segment_munis:
            prices = muni_avg_prices_by_segment[muni]
            if 'commodity' in prices and any(s in prices for s in ['premium', 'commercial_specialty', 'high_specialty']):
                commodity_price = prices['commodity']
                higher_segments = {s: p for s, p in prices.items() 
                                  if s in ['premium', 'commercial_specialty', 'high_specialty']}
                if higher_segments:
                    avg_higher_price = sum(higher_segments.values()) / len(higher_segments)
                    premium_pct = ((avg_higher_price / commodity_price) - 1) * 100
                    avg_premium_pct.append(premium_pct)
        
        if avg_premium_pct:
            overall_premium = sum(avg_premium_pct) / len(avg_premium_pct)
            print(f"\nFor municipalities selling in both commodity and higher segments:")
            print(f"  Average price premium for higher segments: +{overall_premium:.1f}% above commodity prices")
            print(f"  This represents the typical price improvement potential for upgrading from commodity to higher segments")
        
        # Create a structure to pass segment data to other functions
        municipality_segment_data = {}
        for muni in muni_segment_participation:
            municipality_segment_data[muni] = {
                'segments': list(muni_segment_participation[muni].keys()),
                'dominant_segment': muni_dominant_segment.get(muni, None),
                'avg_prices': muni_avg_prices_by_segment[muni],
                'volumes': {s: v for s, v in muni_segment_participation[muni].items()},
                'revenues': {s: r for s, r in muni_segment_revenues[muni].items()},
                'is_commodity_only': muni in commodity_only_munis,
                'is_multi_segment': muni in multi_segment_munis
            }
    else:
        segment_paths = {}
        municipality_segment_data = {}
        
    return paths_by_volume, paths_by_fob, paths_by_fob_per_tonne, segment_paths, municipality_segment_data

def analyze_centrality(G, node_layers):
    """
    Analyze the centrality of nodes to identify potential bottlenecks and 
    positions of power within the supply chain.
    """
    print("Analyzing network centrality metrics...")
    
    # Create a copy of the graph with weights as a simple number for centrality calcs
    G_centrality = G.copy()
    for u, v, data in G.edges(data=True):
        G_centrality[u][v]['weight'] = data['weight']
    
    # Calculate different centrality measures
    # Betweenness centrality identifies nodes that bridge different parts of the network
    betweenness = nx.betweenness_centrality(G_centrality, weight='weight', normalized=True)
    
    # Eigenvector centrality identifies nodes connected to other important nodes
    eigenvector = nx.eigenvector_centrality(G_centrality, weight='weight', max_iter=1000)
    
    # Degree centrality (in and out) shows direct connections
    in_degree = nx.in_degree_centrality(G_centrality)
    out_degree = nx.out_degree_centrality(G_centrality)
    
    # Store results by node layer
    centrality_by_layer = defaultdict(lambda: defaultdict(list))
    for node in G.nodes():
        layer = node_layers[node]
        prefix = node[:2]  # M:, E:, I:, or D:
        
        centrality_by_layer[layer]['betweenness'].append((node, betweenness[node]))
        centrality_by_layer[layer]['eigenvector'].append((node, eigenvector[node]))
        centrality_by_layer[layer]['in_degree'].append((node, in_degree[node]))
        centrality_by_layer[layer]['out_degree'].append((node, out_degree[node]))
    
    # Display top entities by centrality for each layer
    layer_names = {
        1: "Municipalities", 
        2: "Exporters", 
        3: "Importers", 
        4: "Destinations"
    }
    
    # Define centrality measures to analyze
    centrality_measures = ['betweenness', 'eigenvector']
    
    results = {}
    
    for layer in [1, 2, 3, 4]:
        results[layer] = {}
        print(f"\n== Top {layer_names[layer]} by Centrality ==")
        
        for measure in centrality_measures:
            nodes = sorted(centrality_by_layer[layer][measure], key=lambda x: x[1], reverse=True)
            results[layer][measure] = nodes
            
            print(f"\nTop 10 {layer_names[layer]} by {measure} centrality:")
            for i, (node, score) in enumerate(nodes[:10]):
                print(f"{i+1}. {node[2:]}: {score:.4f}")
    
    return results

def identify_bottlenecks(G, centrality_results, node_layers, municipality_segment_data=None):
    """
    Identify potential bottlenecks in the supply chain based on centrality metrics
    and connection patterns.
    """
    print("\nIdentifying potential bottlenecks in the supply chain...")
    
    # Focus on exporters and importers (layers 2 and 3)
    bottlenecks = []
    
    # Look for exporters with high betweenness centrality
    top_exporters = centrality_results[2]['betweenness'][:15]  # Examine more exporters
    
    # For each potential bottleneck, analyze their influence
    for node, score in top_exporters:
        # Get the connections of this exporter
        suppliers = list(G.predecessors(node))  # municipalities
        buyers = list(G.successors(node))       # importers
        
        # Calculate volume and FOB metrics for this exporter
        total_volume = 0
        total_fob = 0
        for u, v, data in G.edges(node, data=True):
            if u == node:  # Outgoing edge
                total_volume += data['weight']
                total_fob += data.get('fob', 0)
        
        # Check for bottleneck conditions - either high supplier:buyer ratio or high market concentration
        if (len(suppliers) > 3 * len(buyers) and score > 0.01) or (len(suppliers) > 10 and len(buyers) <= 3):
            bottlenecks.append({
                'node': node,
                'centrality': score,
                'suppliers': len(suppliers),
                'buyers': len(buyers),
                'supplier_to_buyer_ratio': len(suppliers) / max(1, len(buyers)),
                'type': 'exporter',
                'volume': total_volume,
                'fob': total_fob,
                'fob_per_tonne': total_fob / total_volume if total_volume > 0 else 0
            })
    
    # Similar analysis for importers
    top_importers = centrality_results[3]['betweenness'][:15]  # Examine more importers
    for node, score in top_importers:
        suppliers = list(G.predecessors(node))  # exporters
        buyers = list(G.successors(node))       # destinations
        
        # Calculate volume and FOB metrics for this importer
        total_volume = 0
        total_fob = 0
        for u, v, data in G.in_edges(node, data=True):
            total_volume += data['weight']
            total_fob += data.get('fob', 0)
            
        if (len(suppliers) > 2 * len(buyers) and score > 0.01) or (len(suppliers) > 8 and len(buyers) <= 2):
            bottlenecks.append({
                'node': node,
                'centrality': score,
                'suppliers': len(suppliers),
                'buyers': len(buyers),
                'supplier_to_buyer_ratio': len(suppliers) / max(1, len(buyers)),
                'type': 'importer',
                'volume': total_volume,
                'fob': total_fob,
                'fob_per_tonne': total_fob / total_volume if total_volume > 0 else 0
            })
    
    # Sort bottlenecks by influence (centrality * supplier_to_buyer_ratio)
    for b in bottlenecks:
        b['influence'] = b['centrality'] * b['supplier_to_buyer_ratio']
    
    bottlenecks = sorted(bottlenecks, key=lambda x: x['influence'], reverse=True)
    
    # Report findings
    print("\nTop potential bottlenecks in the supply chain:")
    for i, b in enumerate(bottlenecks[:10]):
        print(f"{i+1}. {b['type'].title()} {b['node'][2:]}")
        print(f"   Centrality: {b['centrality']:.4f}, Suppliers: {b['suppliers']}, Buyers: {b['buyers']}")
        print(f"   Supplier-to-buyer ratio: {b['supplier_to_buyer_ratio']:.2f}")
        print(f"   Estimated market influence: {b['influence']:.4f}")
        print(f"   Volume handled: {b['volume']:.2f} tonnes, Avg. FOB: ${b['fob_per_tonne']:.2f}/tonne")
    
    # Analyze market concentration
    # Calculate municipality distribution across exporters
    municipalities = [node for node in G.nodes() if node_layers[node] == 1]
    exporters = [node for node in G.nodes() if node_layers[node] == 2]
    
    # Count how many municipalities each exporter buys from
    exporter_supplier_counts = {}
    for exporter in exporters:
        suppliers = list(G.predecessors(exporter))
        exporter_supplier_counts[exporter] = len(suppliers)
    
    # Calculate what percentage of municipalities are connected to at least N exporters
    muni_exporter_counts = {}
    for muni in municipalities:
        buyers = list(G.successors(muni))
        muni_exporter_counts[muni] = len(buyers)
    
    muni_exporter_distribution = defaultdict(int)
    for muni, count in muni_exporter_counts.items():
        muni_exporter_distribution[count] += 1
    
    # Print market concentration analysis
    print("\n== Market Concentration Analysis ==")
    
    # Municipality to exporter connection distribution
    print("\nMunicipality-Exporter connection distribution:")
    total_munis = len(municipalities)
    
    muni_connectedness_levels = {
        'monopoly': 0,        # Connected to 1 exporter
        'duopoly': 0,         # Connected to 2 exporters
        'limited_choice': 0,  # Connected to 3-4 exporters
        'competitive': 0      # Connected to 5+ exporters
    }
    
    # Track municipalities in each connectedness category
    muni_by_connectedness = {
        'monopoly': [],
        'duopoly': [],
        'limited_choice': [],
        'competitive': []
    }
    
    for muni, count in muni_exporter_counts.items():
        if count == 1:
            muni_connectedness_levels['monopoly'] += 1
            muni_by_connectedness['monopoly'].append(muni)
        elif count == 2:
            muni_connectedness_levels['duopoly'] += 1
            muni_by_connectedness['duopoly'].append(muni)
        elif count <= 4:
            muni_connectedness_levels['limited_choice'] += 1
            muni_by_connectedness['limited_choice'].append(muni)
        else:
            muni_connectedness_levels['competitive'] += 1
            muni_by_connectedness['competitive'].append(muni)
    
    for level, count in muni_connectedness_levels.items():
        percentage = (count / total_munis * 100) if total_munis > 0 else 0
        print(f"  {level.replace('_', ' ').title()}: {count} municipalities ({percentage:.1f}%)")
    
    # Identify most concentrated municipalities (those with fewest options)
    most_concentrated = sorted(muni_exporter_counts.items(), key=lambda x: x[1])[:10]
    
    print("\nMost concentrated municipalities (limited market access):")
    for i, (muni, count) in enumerate(most_concentrated):
        print(f"  {i+1}. {muni[2:]}: Connected to only {count} exporters")
    
    # Analyze the relationship between market access and prices
    print("\n== Market Concentration and Price Correlation ==")
    
    # Calculate average FOB per tonne for each municipality
    muni_volumes = defaultdict(float)
    muni_fob = defaultdict(float)
    
    for u, v, data in G.edges(data=True):
        if node_layers.get(u) == 1:  # Municipality node
            muni_volumes[u] += data['weight']
            if 'fob' in data:
                muni_fob[u] += data['fob']
    
    # Calculate average FOB per municipality and group by connectedness level
    connectedness_prices = {
        'monopoly': [],
        'duopoly': [],
        'limited_choice': [],
        'competitive': []
    }
    
    for connectedness, munis in muni_by_connectedness.items():
        for muni in munis:
            if muni in muni_volumes and muni_volumes[muni] > 0 and muni in muni_fob and muni_fob[muni] > 0:
                avg_price = muni_fob[muni] / muni_volumes[muni]
                connectedness_prices[connectedness].append(avg_price)
    
    # Calculate average, median and price statistics by market concentration level
    price_stats = {}
    for level, prices in connectedness_prices.items():
        if prices:
            price_stats[level] = {
                'count': len(prices),
                'mean': np.mean(prices),
                'median': np.median(prices),
                'min': min(prices),
                'max': max(prices),
                'p25': np.percentile(prices, 25),
                'p75': np.percentile(prices, 75)
            }
    
    # Report price statistics by market concentration level
    print("\nAverage FOB per tonne by market access level:")
    
    # Define comparison base for percentage calculations
    if 'competitive' in price_stats:
        base_price = price_stats['competitive']['median']
    else:
        # If no competitive municipalities, use overall median
        all_prices = []
        for prices in connectedness_prices.values():
            all_prices.extend(prices)
        base_price = np.median(all_prices) if all_prices else 0
    
    # Calculate price penalties for less competitive markets
    penalties = {}
    for level in ['monopoly', 'duopoly', 'limited_choice']:
        if level in price_stats and base_price > 0:
            price_diff = price_stats[level]['median'] - base_price
            pct_diff = (price_diff / base_price) * 100
            penalties[level] = (price_diff, pct_diff)
    
    # Print detailed price stats by concentration level
    for level in ['monopoly', 'duopoly', 'limited_choice', 'competitive']:
        if level in price_stats:
            stats = price_stats[level]
            print(f"\n{level.replace('_', ' ').title()} markets ({stats['count']} municipalities):")
            print(f"  Median price: ${stats['median']:.2f}/tonne")
            print(f"  Price range: ${stats['min']:.2f} - ${stats['max']:.2f}/tonne")
            print(f"  Middle 50% range: ${stats['p25']:.2f} - ${stats['p75']:.2f}/tonne")
            
            # Print penalty for less competitive markets
            if level in penalties:
                price_diff, pct_diff = penalties[level]
                diff_sign = "+" if price_diff >= 0 else ""
                print(f"  Price differential vs. competitive markets: {diff_sign}${price_diff:.2f}/tonne ({diff_sign}{pct_diff:.1f}%)")
                if price_diff < 0:
                    print(f"  This suggests a market concentration penalty of {abs(pct_diff):.1f}%")
    
    # Analyze segment participation by market concentration level
    if municipality_segment_data:
        print("\n== Market Segment Distribution by Concentration Level ==")
        
        muni_dominant_segment = municipality_segment_data.get('muni_dominant_segment', {})
        segment_by_concentration = {
            'monopoly': defaultdict(int),
            'duopoly': defaultdict(int),
            'limited_choice': defaultdict(int),
            'competitive': defaultdict(int)
        }
        
        # Count municipalities in each segment by concentration level
        for level, munis in muni_by_connectedness.items():
            for muni in munis:
                if muni in muni_dominant_segment:
                    segment = muni_dominant_segment[muni]
                    segment_by_concentration[level][segment] += 1
        
        # Report segment distribution by concentration level
        for level, segment_counts in segment_by_concentration.items():
            if segment_counts:
                total = sum(segment_counts.values())
                print(f"\n{level.replace('_', ' ').title()} markets segment distribution:")
                
                for segment, count in sorted(segment_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = (count / total * 100) if total > 0 else 0
                    print(f"  {segment.replace('_', ' ').title()}: {count} municipalities ({pct:.1f}%)")
    
    # Recommendations for reducing bottlenecks
    print("\n== Recommendations for Reducing Bottlenecks and Improving Farmer Income ==")
    
    if len(bottlenecks) > 0:
        print("\n1. Address Key Bottlenecks:")
        for i, b in enumerate(bottlenecks[:3]):
            if b['type'] == 'exporter':
                print(f"   • {b['node'][2:]}: This exporter connects {b['suppliers']} municipalities to only {b['buyers']} importers")
                print(f"     Recommendation: Facilitate direct connections between these municipalities and more importers")
                print(f"     or establish producer cooperatives to increase bargaining power")
            else:
                print(f"   • {b['node'][2:]}: This importer sources from {b['suppliers']} exporters for only {b['buyers']} destinations")
                print(f"     Recommendation: Diversify market access or establish direct trade relationships")
    
    # Recommendations based on market concentration analysis
    if muni_connectedness_levels['monopoly'] > 0 or muni_connectedness_levels['duopoly'] > 0:
        print("\n2. Improve Market Access in Concentrated Regions:")
        
        # Count total affected municipalities
        affected = muni_connectedness_levels['monopoly'] + muni_connectedness_levels['duopoly']
        percentage = (affected / total_munis * 100) if total_munis > 0 else 0
        
        print(f"   • {affected} municipalities ({percentage:.1f}%) have access to 2 or fewer exporters")
        
        # Calculate estimated economic impact if price penalty exists
        if 'monopoly' in penalties and 'duopoly' in penalties:
            mono_diff, _ = penalties['monopoly']
            duo_diff, _ = penalties['duopoly']
            
            # Calculate weighted average penalty
            avg_penalty = ((mono_diff * muni_connectedness_levels['monopoly']) + 
                          (duo_diff * muni_connectedness_levels['duopoly'])) / affected if affected > 0 else 0
            
            if avg_penalty < 0:
                # Calculate total volume from concentrated municipalities
                concentrated_volume = sum(muni_volumes[m] for m in muni_by_connectedness['monopoly'] + muni_by_connectedness['duopoly'] 
                                         if m in muni_volumes)
                
                # Calculate potential economic impact of addressing concentration
                potential_impact = abs(avg_penalty) * concentrated_volume
                
                print(f"   • Estimated price penalty: ${abs(avg_penalty):.2f}/tonne")
                print(f"   • Potential economic impact of addressing market concentration: ${potential_impact:.2f}")
        
        print("     Recommendation: Target these regions for market access interventions such as:")
        print("       - Establish regional collection centers connected to multiple exporters")
        print("       - Develop digital platforms connecting farmers to diverse buyers")
        print("       - Strengthen producer organizations to increase collective bargaining power")
        
    # Add value chain integration recommendations
    print("\n3. Strengthen Value Chain Integration:")
    print("   • Promote vertical integration where appropriate:")
    print("     - Support farmer groups in acquiring processing equipment to capture more value")
    print("     - Develop training programs for quality improvement at the municipal level")
    print("     - Establish certification programs to access premium markets")
    
    # Transparency recommendations
    print("\n4. Increase Price Transparency:")
    print("   • Implement price information systems accessible to farmers")
    print("   • Support fair trade and direct trade initiatives to shorten the supply chain")
    print("   • Develop blockchain or other traceability systems to increase transparency")
    
    # Organize priority recommendations based on concentration analysis
    priority_recommendations = []
    
    # If we have identified clear market concentration penalties
    if 'monopoly' in penalties and penalties['monopoly'][0] < 0:
        # Priority 1: Address monopoly situations first as they have highest penalties
        monopoly_munis = muni_by_connectedness['monopoly']
        if monopoly_munis:
            # Find most affected municipalities (highest volume with monopoly)
            monopoly_volumes = [(m, muni_volumes.get(m, 0)) for m in monopoly_munis]
            top_monopoly = sorted(monopoly_volumes, key=lambda x: x[1], reverse=True)[:5]
            
            priority_recommendations.append({
                'type': 'market_access',
                'target': 'monopoly',
                'municipalities': [m[0] for m in top_monopoly],
                'potential_impact': sum(m[1] * abs(penalties['monopoly'][0]) for m in top_monopoly),
                'recommendation': "Establish additional market channels in monopoly regions"
            })
    
    # If there are more diverse opportunities in the high-volume segment
    if municipality_segment_data and 'market_segments' in municipality_segment_data:
        # Find commodity municipalities with competitive markets
        commodity_munis = []
        if 'muni_dominant_segment' in municipality_segment_data:
            for muni, segment in municipality_segment_data['muni_dominant_segment'].items():
                if segment == 'commodity' and muni in muni_by_connectedness['competitive']:
                    commodity_munis.append(muni)
        
        if commodity_munis:
            # Calculate potential from moving to premium segment
            if 'market_segments' in municipality_segment_data:
                market_segments = municipality_segment_data['market_segments']
                commodity_max = market_segments['commodity'][1] if 'commodity' in market_segments else 0
                premium_min = market_segments['premium'][0] if 'premium' in market_segments else 0
                
                if premium_min > commodity_max:
                    potential_gain = premium_min - commodity_max
                    
                    # Find volumes for these municipalities
                    commodity_competitive_vols = [(m, muni_volumes.get(m, 0)) for m in commodity_munis]
                    top_opportunities = sorted(commodity_competitive_vols, key=lambda x: x[1], reverse=True)[:5]
                    
                    priority_recommendations.append({
                        'type': 'quality_improvement',
                        'target': 'commodity_to_premium',
                        'municipalities': [m[0] for m in top_opportunities],
                        'potential_impact': sum(m[1] * potential_gain for m in top_opportunities),
                        'recommendation': "Quality improvement programs for competitive commodity producers"
                    })
    
    # Print priority recommendations
    if priority_recommendations:
        print("\n== Priority Recommendations Based on Data Analysis ==")
        
        # Sort by potential impact
        priority_recommendations.sort(key=lambda x: x.get('potential_impact', 0), reverse=True)
        
        for i, rec in enumerate(priority_recommendations):
            print(f"\n{i+1}. {rec['recommendation']}")
            
            target_munis = [m[2:] for m in rec['municipalities']]
            print(f"   Target municipalities: {', '.join(target_munis[:3])}" + 
                 (f" and {len(target_munis)-3} more" if len(target_munis) > 3 else ""))
            
            if 'potential_impact' in rec:
                print(f"   Estimated potential impact: ${rec['potential_impact']:.2f}")
    
    return bottlenecks, muni_connectedness_levels, most_concentrated, price_stats if 'price_stats' in locals() else None, muni_by_connectedness

def detect_communities(G, node_layers):
    """
    Detect communities in the network to identify natural groupings
    of trade participants.
    """
    print("\nDetecting communities in the trade network...")
    
    # We need to convert the directed graph to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Add weights to the undirected graph
    for u, v in G_undirected.edges():
        if G.has_edge(u, v):
            G_undirected[u][v]['weight'] = G[u][v]['weight']
        else:
            G_undirected[u][v]['weight'] = G[v][u]['weight']
    
    # Try to detect communities using the Louvain method
    try:
        partition = community_louvain.best_partition(G_undirected, weight='weight')
        
        # Count nodes in each community
        community_sizes = defaultdict(int)
        community_layers = defaultdict(lambda: defaultdict(int))
        
        for node, community_id in partition.items():
            community_sizes[community_id] += 1
            layer = node_layers[node]
            community_layers[community_id][layer] += 1
        
        # Sort communities by size
        sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Report findings
        print(f"\nDetected {len(sorted_communities)} distinct trade communities")
        print("\nTop communities by size:")
        
        for i, (community_id, size) in enumerate(sorted_communities[:10]):
            print(f"\nCommunity #{i+1} (ID: {community_id}): {size} members")
            print(f"  Municipalities: {community_layers[community_id][1]}")
            print(f"  Exporters: {community_layers[community_id][2]}")
            print(f"  Importers: {community_layers[community_id][3]}")
            print(f"  Destinations: {community_layers[community_id][4]}")
            
            # Report some key members
            members = [node for node, comm_id in partition.items() if comm_id == community_id]
            
            # Group members by layer
            community_members = defaultdict(list)
            for member in members:
                layer = node_layers[member]
                community_members[layer].append(member)
            
            # Display a few key members from each layer
            for layer in [1, 2, 3, 4]:
                if community_members[layer]:
                    layer_name = {1: "Municipalities", 2: "Exporters", 
                                  3: "Importers", 4: "Destinations"}[layer]
                    print(f"  {layer_name}: ", end="")
                    members_to_show = min(5, len(community_members[layer]))
                    print(", ".join([m[2:] for m in community_members[layer][:members_to_show]]))
                    if len(community_members[layer]) > members_to_show:
                        print(f"    ... and {len(community_members[layer]) - members_to_show} more")
        
        return partition, sorted_communities, community_layers
    
    except Exception as e:
        print(f"Error detecting communities: {e}")
        print("You may need to install the python-louvain package: pip install python-louvain")
        return None, None, None

def create_bipartite_network(df):
    """
    Create a bipartite network of municipalities and importers, 
    with exporters as edge attributes.
    """
    print("\nCreating bipartite network of municipalities and importers...")
    
    # Create a weighted bipartite graph
    B = nx.Graph()
    
    # Add attributes to track node types
    B.graph['top'] = set()  # municipalities
    B.graph['bottom'] = set()  # importers
    
    # Process each trade record
    muni_importer_pairs = defaultdict(lambda: {'volume': 0, 'fob': 0, 'exporters': set()})
    
    for idx, row in df.iterrows():
        # Skip if essential components are missing
        if pd.isna(row['municipality_of_export']) or pd.isna(row['importer']):
            continue
        
        # Create node names
        municipality = f"M:{row['municipality_of_export']}"
        importer = f"I:{row['importer']}"
        exporter = f"E:{row['exporter']}" if not pd.isna(row['exporter']) else "E:Unknown"
        
        # Add nodes if they don't exist
        if municipality not in B:
            B.add_node(municipality, bipartite=0)
            B.graph['top'].add(municipality)
        
        if importer not in B:
            B.add_node(importer, bipartite=1)
            B.graph['bottom'].add(importer)
        
        # Aggregate trade between this municipality and importer
        pair_key = (municipality, importer)
        muni_importer_pairs[pair_key]['volume'] += row['volume']
        
        if not pd.isna(row['fob']):
            muni_importer_pairs[pair_key]['fob'] += row['fob']
        
        muni_importer_pairs[pair_key]['exporters'].add(exporter)
    
    # Add edges with weights and exporter information
    for (municipality, importer), data in muni_importer_pairs.items():
        B.add_edge(
            municipality, 
            importer, 
            weight=data['volume'],
            fob=data['fob'],
            fob_per_unit=data['fob']/data['volume'] if data['volume'] > 0 else 0,
            exporters=list(data['exporters']),
            exporter_count=len(data['exporters'])
        )
    
    print(f"Created bipartite network with {len(B.graph['top'])} municipalities, "
          f"{len(B.graph['bottom'])} importers, and {len(B.edges())} trade relationships")
    
    return B

def analyze_bipartite_matching(B, municipality_segment_data=None, muni_connectedness=None):
    """
    Analyze the bipartite network to understand potential for optimizing
    municipality-importer pairings.
    """
    print("\nAnalyzing bipartite matching patterns...")
    
    # Extract the top (municipalities) and bottom (importers) node sets
    municipalities = B.graph['top']
    importers = B.graph['bottom']
    
    # Compute projections to see relationships within each set
    print("Computing municipality projection...")
    muni_projection = bipartite.weighted_projected_graph(B, municipalities)
    
    print("Computing importer projection...")
    importer_projection = bipartite.weighted_projected_graph(B, importers)
    
    # Find municipalities with highest diversity of importers
    muni_diversity = {node: len(list(B.neighbors(node))) for node in municipalities}
    top_diverse_munis = sorted(muni_diversity.items(), key=lambda x: x[1], reverse=True)
    
    # Find importers with highest diversity of municipalities
    importer_diversity = {node: len(list(B.neighbors(node))) for node in importers}
    top_diverse_importers = sorted(importer_diversity.items(), key=lambda x: x[1], reverse=True)
    
    # Find edges with highest FOB per unit
    high_value_edges = []
    for u, v, data in B.edges(data=True):
        if 'fob_per_unit' in data and data['fob_per_unit'] > 0:
            high_value_edges.append((u, v, data['fob_per_unit'], data['weight']))
    
    high_value_edges.sort(key=lambda x: x[2], reverse=True)
    
    # Find potential opportunities: Municipalities selling at low FOB that could
    # connect to importers buying at high FOB
    muni_avg_fob = {}
    for muni in municipalities:
        fob_values = []
        volumes = []
        for _, importer, data in B.edges(muni, data=True):
            if 'fob_per_unit' in data and data['fob_per_unit'] > 0:
                fob_values.append(data['fob_per_unit'] * data['weight'])
                volumes.append(data['weight'])
        
        if sum(volumes) > 0:
            muni_avg_fob[muni] = sum(fob_values) / sum(volumes)
    
    importer_avg_fob = {}
    for imp in importers:
        fob_values = []
        volumes = []
        for muni, _, data in B.edges(imp, data=True):
            if 'fob_per_unit' in data and data['fob_per_unit'] > 0:
                fob_values.append(data['fob_per_unit'] * data['weight'])
                volumes.append(data['weight'])
        
        if sum(volumes) > 0:
            importer_avg_fob[imp] = sum(fob_values) / sum(volumes)
    
    # Find municipalities with low average FOB
    low_fob_munis = sorted(muni_avg_fob.items(), key=lambda x: x[1])
    
    # Find importers with high average FOB
    high_fob_importers = sorted(importer_avg_fob.items(), key=lambda x: x[1], reverse=True)
    
    # Output results
    print("\n== Municipality Trade Diversity ==")
    print("\nTop municipalities by importer diversity:")
    for i, (muni, count) in enumerate(top_diverse_munis[:10]):
        print(f"{i+1}. {muni[2:]}: Trading with {count} importers")
    
    print("\n== Importer Trade Diversity ==")
    print("\nTop importers by municipality diversity:")
    for i, (imp, count) in enumerate(top_diverse_importers[:10]):
        print(f"{i+1}. {imp[2:]}: Buying from {count} municipalities")
    
    print("\n== Highest Value Trade Relationships ==")
    print("\nTop trade relationships by FOB per tonne:")
    for i, (muni, imp, fob_per_unit, volume) in enumerate(high_value_edges[:10]):
        print(f"{i+1}. {muni[2:]} → {imp[2:]}: ${fob_per_unit:.2f}/tonne, "
              f"Volume: {volume:.2f} tonnes")
    
    print("\n== Optimization Opportunities ==")
    print("\nMunicipalities with lowest average FOB per tonne:")
    for i, (muni, avg_fob) in enumerate(low_fob_munis[:10]):
        print(f"{i+1}. {muni[2:]}: ${avg_fob:.2f}/tonne average")
    
    print("\nImporters with highest average FOB per tonne:")
    for i, (imp, avg_fob) in enumerate(high_fob_importers[:10]):
        print(f"{i+1}. {imp[2:]}: ${avg_fob:.2f}/tonne average")
    
    print("\nSkipping potential high-value connections analysis to improve performance")
    
    # Return basic results without calculating potential connections
    return {
        'top_diverse_munis': top_diverse_munis,
        'top_diverse_importers': top_diverse_importers,
        'high_value_edges': high_value_edges,
        'low_fob_munis': low_fob_munis,
        'high_fob_importers': high_fob_importers,
        'potential_connections': [],      # Empty placeholder
        'top_opportunities': []           # Empty placeholder
    }

def analyze_broader_market_trends(G, node_layers, path_weights, segment_paths=None):
    """
    Analyze broader market trends with a focus on volume metrics rather than just
    high-value specialty coffee paths. This provides insights into the mainstream coffee market.
    """
    print("\n== Broader Market Analysis ==")
    
    # Identify key municipalities, exporters, and importers by volume
    municipality_volumes = defaultdict(float)
    exporter_volumes = defaultdict(float)
    importer_volumes = defaultdict(float)
    destination_volumes = defaultdict(float)
    
    # Calculate total volume for each node
    for (muni, exp, imp, dest), weights in path_weights.items():
        volume = weights['volume']
        municipality_volumes[muni] += volume
        exporter_volumes[exp] += volume
        importer_volumes[imp] += volume
        destination_volumes[dest] += volume
    
    # Sort by volume
    top_municipalities = sorted(municipality_volumes.items(), key=lambda x: x[1], reverse=True)
    top_exporters = sorted(exporter_volumes.items(), key=lambda x: x[1], reverse=True)
    top_importers = sorted(importer_volumes.items(), key=lambda x: x[1], reverse=True)
    top_destinations = sorted(destination_volumes.items(), key=lambda x: x[1], reverse=True)
    
    # Report findings on volume distribution
    total_volume = sum(municipality_volumes.values())
    
    print("\nVolume concentration analysis:")
    print(f"Total volume in dataset: {total_volume:.2f} tonnes")
    
    # Calculate and print volume concentration for each layer
    for layer_name, data in [
        ("Municipalities", top_municipalities), 
        ("Exporters", top_exporters), 
        ("Importers", top_importers), 
        ("Destinations", top_destinations)
    ]:
        # Calculate percentage of total volume handled by top 5, 10, and 20 entities
        top_5_volume = sum([vol for _, vol in data[:5]])
        top_10_volume = sum([vol for _, vol in data[:10]])
        top_20_volume = sum([vol for _, vol in data[:20]])
        
        print(f"\n{layer_name} concentration:")
        print(f"  Top 5 {layer_name.lower()} handle {top_5_volume:.2f} tonnes ({top_5_volume/total_volume*100:.1f}% of total)")
        print(f"  Top 10 {layer_name.lower()} handle {top_10_volume:.2f} tonnes ({top_10_volume/total_volume*100:.1f}% of total)")
        print(f"  Top 20 {layer_name.lower()} handle {top_20_volume:.2f} tonnes ({top_20_volume/total_volume*100:.1f}% of total)")
    
    # Calculate price-volume relationships
    # This helps understand if higher volumes correlate with lower prices
    price_volume_correlations = {
        'municipalities': [],
        'exporters': [],
        'importers': []
    }
    
    # Calculate average FOB per municipality, exporter, and importer
    municipality_fob = defaultdict(float)
    exporter_fob = defaultdict(float)
    importer_fob = defaultdict(float)
    
    # Sum up FOB values
    for (muni, exp, imp, _), weights in path_weights.items():
        if weights['volume'] > 0 and weights['fob'] > 0:
            municipality_fob[muni] += weights['fob']
            exporter_fob[exp] += weights['fob']
            importer_fob[imp] += weights['fob']
    
    # Calculate FOB per tonne
    for muni in municipality_volumes:
        if municipality_volumes[muni] > 0 and municipality_fob[muni] > 0:
            fob_per_tonne = municipality_fob[muni] / municipality_volumes[muni]
            price_volume_correlations['municipalities'].append((muni, municipality_volumes[muni], fob_per_tonne))
    
    for exp in exporter_volumes:
        if exporter_volumes[exp] > 0 and exporter_fob[exp] > 0:
            fob_per_tonne = exporter_fob[exp] / exporter_volumes[exp]
            price_volume_correlations['exporters'].append((exp, exporter_volumes[exp], fob_per_tonne))
    
    for imp in importer_volumes:
        if importer_volumes[imp] > 0 and importer_fob[imp] > 0:
            fob_per_tonne = importer_fob[imp] / importer_volumes[imp]
            price_volume_correlations['importers'].append((imp, importer_volumes[imp], fob_per_tonne))
    
    # Calculate correlation coefficients
    correlation_results = {}
    
    print("\nPrice-Volume relationship analysis:")
    for entity_type, data in price_volume_correlations.items():
        if len(data) > 5:  # Need enough data for meaningful correlation
            volumes = [vol for _, vol, _ in data]
            prices = [price for _, _, price in data]
            
            if len(volumes) == len(prices) and len(volumes) > 0:
                correlation = np.corrcoef(volumes, prices)[0, 1]
                correlation_results[entity_type] = correlation
                
                print(f"  {entity_type.title()}: Correlation between volume and price: {correlation:.4f}")
                if correlation < -0.2:
                    print(f"    Negative correlation suggests higher volumes are associated with lower prices")
                    print(f"    This may indicate limited bargaining power for high-volume producers")
                elif correlation > 0.2:
                    print(f"    Positive correlation suggests higher volumes are associated with higher prices")
                    print(f"    This may indicate economies of scale or market power benefits")
                else:
                    print(f"    No strong correlation between volume and price")
    
    # Volume-based market segmentation
    # Group municipalities by volume to identify different scales of production
    if len(municipality_volumes) > 0:
        volume_thresholds = {
            'small_scale': np.percentile([vol for _, vol in top_municipalities], 25),
            'medium_scale': np.percentile([vol for _, vol in top_municipalities], 50),
            'large_scale': np.percentile([vol for _, vol in top_municipalities], 75)
        }
        
        volume_segments = {
            'micro_scale': [],      # Below small_scale threshold
            'small_scale': [],      # Between small and medium
            'medium_scale': [],     # Between medium and large
            'large_scale': []       # Above large_scale threshold
        }
        
        # Categorize municipalities by volume
        for muni, volume in top_municipalities:
            if volume < volume_thresholds['small_scale']:
                volume_segments['micro_scale'].append((muni, volume))
            elif volume < volume_thresholds['medium_scale']:
                volume_segments['small_scale'].append((muni, volume))
            elif volume < volume_thresholds['large_scale']:
                volume_segments['medium_scale'].append((muni, volume))
            else:
                volume_segments['large_scale'].append((muni, volume))
        
        # Calculate average prices for each segment
        segment_avg_prices = {}
        
        for segment, municipalities in volume_segments.items():
            if municipalities:
                total_fob = 0
                total_vol = 0
                
                for muni, vol in municipalities:
                    if muni in municipality_fob:
                        total_fob += municipality_fob[muni]
                        total_vol += vol
                
                if total_vol > 0:
                    segment_avg_prices[segment] = total_fob / total_vol
        
        # Report volume-based segmentation
        print("\nMunicipality volume-based segmentation:")
        
        for segment, municipalities in volume_segments.items():
            segment_volume = sum([vol for _, vol in municipalities])
            segment_count = len(municipalities)
            volume_pct = (segment_volume / total_volume * 100) if total_volume > 0 else 0
            count_pct = (segment_count / len(municipality_volumes) * 100) if len(municipality_volumes) > 0 else 0
            
            print(f"\n{segment.replace('_', ' ').title()} Producers:")
            
            # Show volume range for this segment
            if segment == 'micro_scale':
                volume_range = f"Below {volume_thresholds['small_scale']:.2f} tonnes"
            elif segment == 'small_scale':
                volume_range = f"{volume_thresholds['small_scale']:.2f} - {volume_thresholds['medium_scale']:.2f} tonnes"
            elif segment == 'medium_scale':
                volume_range = f"{volume_thresholds['medium_scale']:.2f} - {volume_thresholds['large_scale']:.2f} tonnes"
            else:
                volume_range = f"Above {volume_thresholds['large_scale']:.2f} tonnes"
                
            print(f"  Volume range: {volume_range}")
            print(f"  Count: {segment_count} municipalities ({count_pct:.1f}% of total)")
            print(f"  Total volume: {segment_volume:.2f} tonnes ({volume_pct:.1f}% of total)")
            
            if segment in segment_avg_prices:
                print(f"  Average FOB: ${segment_avg_prices[segment]:.2f}/tonne")
            
            # List top 3 municipalities in this segment
            if municipalities:
                sorted_munis = sorted(municipalities, key=lambda x: x[1], reverse=True)
                print("  Top municipalities in this segment:")
                for i, (muni, vol) in enumerate(sorted_munis[:3]):
                    avg_price = municipality_fob[muni] / vol if vol > 0 and muni in municipality_fob else 0
                    print(f"    {i+1}. {muni[2:]}: {vol:.2f} tonnes, ${avg_price:.2f}/tonne")
    
    # Broader market improvement opportunities
    print("\n== Broader Market Improvement Opportunities ==")
    
    # Identify municipalities with high volumes but low prices
    high_vol_low_price = []
    for muni, vol in top_municipalities:
        if vol > np.percentile([v for _, v in top_municipalities], 60):  # Top 40% by volume
            if muni in municipality_fob and municipality_fob[muni] > 0:
                price = municipality_fob[muni] / vol
                if price < np.percentile([municipality_fob[m] / municipality_volumes[m] 
                                         for m in municipality_volumes 
                                         if municipality_volumes[m] > 0 and m in municipality_fob 
                                         and municipality_fob[m] > 0], 40):  # Bottom 40% by price
                    high_vol_low_price.append((muni, vol, price))
    
    high_vol_low_price.sort(key=lambda x: x[1], reverse=True)  # Sort by volume, highest first
    
    if high_vol_low_price:
        print("\n1. High-Volume, Low-Price Municipalities:")
        print("   These municipalities produce significant volumes but receive below-average prices.")
        print("   They represent prime opportunities for value chain improvements.")
        
        for i, (muni, vol, price) in enumerate(high_vol_low_price[:5]):
            print(f"   {i+1}. {muni[2:]}: {vol:.2f} tonnes at ${price:.2f}/tonne")
        
        print("\n   Recommendations for these municipalities:")
        print("   • Establish quality improvement programs to increase base coffee value")
        print("   • Create producer cooperatives to achieve economies of scale in processing and marketing")
        print("   • Develop direct relationships with specialty roasters for portions of their production")
        print("   • Implement certification programs (organic, fair trade) for price premiums")
    
    # Identify diversification opportunities in the high-volume segment
    if correlation_results.get('municipalities') and correlation_results['municipalities'] < -0.1:
        print("\n2. Diversification Strategies for High-Volume Regions:")
        print("   The negative correlation between volume and price suggests that high-volume")
        print("   regions may benefit from market diversification strategies:")
        print("   • Reserve a percentage of production for specialty market development")
        print("   • Invest in post-harvest processing to differentiate products")
        print("   • Establish regional brands based on origin characteristics")
        print("   • Develop multi-channel sales strategies spanning commodity to specialty markets")
    
    # Identify infrastructure and logistics improvement opportunities
    print("\n3. Infrastructure and Logistics Improvements:")
    print("   • Invest in improved storage facilities to maintain quality and allow for strategic selling")
    print("   • Develop regional processing centers to achieve economies of scale")
    print("   • Improve transportation infrastructure to reduce costs and preserve quality")
    print("   • Implement digital tracking systems to improve supply chain efficiency")
    
    return {
        'top_municipalities': top_municipalities,
        'top_exporters': top_exporters,
        'top_importers': top_importers,
        'top_destinations': top_destinations,
        'correlation_results': correlation_results,
        'high_vol_low_price': high_vol_low_price,
        'volume_segments': volume_segments if 'volume_segments' in locals() else None
    }

def visualize_broader_market(G, node_layers, broader_market_results, bottlenecks, output_dir):
    """Create visualizations for broader market analysis results."""
    print("\nGenerating broader market visualizations...")
    
    # 1. Create a municipality volume distribution visualization
    plt.figure(figsize=(14, 10))
    
    # Get municipality volumes
    muni_volumes = []
    for muni in [node for node in G.nodes() if node_layers[node] == 1]:
        vol = sum([G[muni][succ]['weight'] for succ in G.successors(muni)]) if G.successors(muni) else 0
        if vol > 0:  # Only include municipalities with volume
            muni_volumes.append((muni, vol))
    
    # Sort by volume
    muni_volumes.sort(key=lambda x: x[1], reverse=True)
    
    # Make a distribution histogram
    volumes = [vol for _, vol in muni_volumes]
    
    # Use a log scale to better visualize the distribution
    plt.hist(volumes, bins=50, alpha=0.7, color='#5DA637')
    plt.xscale('log')
    plt.xlabel('Volume (tonnes, log scale)', fontsize=12)
    plt.ylabel('Number of Municipalities', fontsize=12)
    plt.title('Municipality Coffee Production Volume Distribution', fontsize=16)
    
    # Add annotation about concentration
    total_volume = sum(volumes)
    top_10_volume = sum([vol for _, vol in muni_volumes[:10]])
    plt.figtext(0.7, 0.8, f"Top 10 municipalities: {top_10_volume/total_volume*100:.1f}% of volume", 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    volume_dist_path = os.path.join(output_dir, 'municipality_volume_distribution.png')
    plt.savefig(volume_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create a visualization of market concentration
    if 'high_vol_low_price' in broader_market_results:
        plt.figure(figsize=(14, 10))
        
        # Extract key municipalities for highlighting
        high_vol_low_price = broader_market_results['high_vol_low_price']
        hvlp_munis = [muni for muni, _, _ in high_vol_low_price]
        
        # Prepare data for scatter plot
        x_values = []  # volumes
        y_values = []  # prices
        colors = []    # for highlighting key municipalities
        sizes = []     # point sizes
        labels = []    # for annotation
        
        # Calculate average prices
        for muni, vol in muni_volumes:
            muni_edges = [(u, v, data) for u, v, data in G.out_edges(muni, data=True)]
            total_fob = sum([data.get('fob', 0) for _, _, data in muni_edges])
            
            if vol > 0 and total_fob > 0:
                price = total_fob / vol
                x_values.append(vol)
                y_values.append(price)
                
                # Color high-volume, low-price municipalities differently
                if muni in hvlp_munis:
                    colors.append('#D4523E')  # Red for high-vol, low-price
                    sizes.append(100)
                    labels.append(muni[2:])  # Store label for annotation
                else:
                    colors.append('#4A7BC9')  # Blue for others
                    sizes.append(30)
                    labels.append(None)
        
        # Create scatter plot
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(x_values, y_values, c=colors, s=sizes, alpha=0.6)
        
        # Use log scale for volume
        plt.xscale('log')
        
        # Add annotations for highlighted municipalities
        for i, label in enumerate(labels):
            if label:
                plt.annotate(label, (x_values[i], y_values[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        
        plt.xlabel('Volume (tonnes, log scale)', fontsize=12)
        plt.ylabel('Price (FOB/tonne in USD)', fontsize=12)
        plt.title('Municipality Volume vs. Price - Highlighting High-Volume, Low-Price Regions', fontsize=16)
        
        # Add a line showing the trend
        if len(x_values) > 1 and len(y_values) > 1:
            try:
                z = np.polyfit(np.log10(x_values), y_values, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(x_values), max(x_values), 100)
                plt.plot(x_range, p(np.log10(x_range)), "r--", alpha=0.5, 
                         label=f"Trend: {z[0]:.1f}*log10(volume) + {z[1]:.1f}")
                plt.legend()
            except:
                print("Could not fit trend line")
        
        plt.grid(True, alpha=0.3)
        
        # Add explanatory note
        plt.figtext(0.5, 0.01, 
                   "Red dots indicate high-volume municipalities with below-average prices - key targets for intervention",
                   ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        
        price_volume_path = os.path.join(output_dir, 'price_volume_relationship.png')
        plt.savefig(price_volume_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Create a visualization of market concentration and bottlenecks
    plt.figure(figsize=(14, 10))
    
    # Create a graph showing municipalities connected to exporters
    # Focus on most concentrated municipalities and bottleneck exporters
    
    # Get most concentrated municipalities
    concentrated_munis = [muni for muni, count in broader_market_results.get('most_concentrated', [])[:10]]
    
    # Get bottleneck exporters
    bottleneck_exporters = [b['node'] for b in bottlenecks if b['type'] == 'exporter'][:5]
    
    # Create a subgraph with these nodes and their immediate connections
    important_nodes = set(concentrated_munis + bottleneck_exporters)
    
    # Add immediate connections 
    for node in list(important_nodes):
        if node in G:
            # Add predecessors for exporters (municipalities)
            if node[:2] == 'E:':
                for pred in G.predecessors(node):
                    important_nodes.add(pred)
            
            # Add successors for municipalities (exporters)
            if node[:2] == 'M:':
                for succ in G.successors(node):
                    important_nodes.add(succ)
    
    # Create the subgraph
    G_market = G.subgraph(important_nodes).copy()
    
    if len(G_market) > 0:
        # Create a visualization
        plt.figure(figsize=(14, 12))
        
        # Use a bipartite layout
        # Separate nodes by type
        municipalities = [n for n in G_market.nodes() if n[:2] == 'M:']
        exporters = [n for n in G_market.nodes() if n[:2] == 'E:']
        
        # Position municipalities on the left, exporters on the right
        pos = {}
        
        # Position municipalities vertically on the left
        for i, node in enumerate(municipalities):
            pos[node] = (0, i - len(municipalities)/2)
        
        # Position exporters vertically on the right
        for i, node in enumerate(exporters):
            pos[node] = (10, i - len(exporters)/2)
        
        # Draw edges with width based on volume
        max_weight = max([G_market[u][v]['weight'] for u, v in G_market.edges()]) if G_market.edges() else 1
        
        edge_colors = []
        edge_widths = []
        
        for u, v in G_market.edges():
            weight = G_market[u][v]['weight']
            normalized_weight = weight / max_weight
            
            edge_colors.append('gray')
            edge_widths.append(1 + 5 * normalized_weight)
        
        nx.draw_networkx_edges(G_market, pos, 
                               width=edge_widths, 
                               edge_color=edge_colors, 
                               alpha=0.6,
                               arrows=True,
                               arrowsize=10)
        
        # Draw nodes with different colors and sizes
        muni_colors = []
        muni_sizes = []
        
        for node in municipalities:
            if node in concentrated_munis:
                muni_colors.append('#D4523E')  # Red for concentrated
                muni_sizes.append(200)
            else:
                muni_colors.append('#5DA637')  # Green for normal
                muni_sizes.append(100)
        
        exp_colors = []
        exp_sizes = []
        
        for node in exporters:
            if node in bottleneck_exporters:
                exp_colors.append('#D4523E')  # Red for bottlenecks
                exp_sizes.append(300)
            else:
                exp_colors.append('#4A7BC9')  # Blue for normal
                exp_sizes.append(150)
        
        # Draw municipality nodes
        nx.draw_networkx_nodes(G_market, pos, 
                              nodelist=municipalities,
                              node_color=muni_colors,
                              node_size=muni_sizes,
                              alpha=0.8)
        
        # Draw exporter nodes
        nx.draw_networkx_nodes(G_market, pos, 
                              nodelist=exporters,
                              node_color=exp_colors,
                              node_size=exp_sizes,
                              alpha=0.8)
        
        # Add labels
        labels = {}
        for node in G_market.nodes():
            if ((node in concentrated_munis) or 
                (node in bottleneck_exporters) or 
                (node in exporters and len(list(G_market.predecessors(node))) > 5)):
                labels[node] = node[2:]  # Full label for important nodes
            else:
                # Shortened labels for less important nodes
                name = node[2:]
                labels[node] = name[:10] + "..." if len(name) > 10 else name
        
        nx.draw_networkx_labels(G_market, pos, labels=labels, font_size=8,
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Add a title and legend
        plt.title('Market Concentration and Bottlenecks', fontsize=16)
        
        # Create a custom legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#5DA637', label='Municipality', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D4523E', label='Concentrated Municipality', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A7BC9', label='Exporter', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D4523E', label='Bottleneck Exporter', markersize=15)
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.figtext(0.5, 0.01, 
                   "Red nodes indicate market bottlenecks or concentrated municipalities with limited market access",
                   ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        
        plt.axis('off')
        
        market_concentration_path = os.path.join(output_dir, 'market_concentration.png')
        plt.savefig(market_concentration_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'volume_distribution': volume_dist_path,
        'price_volume': price_volume_path if 'price_volume_path' in locals() else None,
        'market_concentration': market_concentration_path if 'market_concentration_path' in locals() else None
    }

def export_results(results, output_dir):
    """Export analysis results to CSV files for further investigation."""
    print("\nExporting analysis results to CSV files...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Format bottlenecks data for export
    if results.get('bottlenecks'):
        bottlenecks_data = []
        for b in results['bottlenecks']:
            bottlenecks_data.append({
                'node_name': b['node'][2:],  # Remove prefix (M:, E:, I:, D:)
                'type': b['type'],
                'centrality': b['centrality'],
                'suppliers': b['suppliers'],
                'buyers': b['buyers'],
                'supplier_to_buyer_ratio': b['supplier_to_buyer_ratio'],
                'influence': b['influence'],
                'volume': b.get('volume', 0),
                'fob_per_tonne': b.get('fob_per_tonne', 0)
            })
        
        bottlenecks_df = pd.DataFrame(bottlenecks_data)
        bottlenecks_path = os.path.join(output_dir, 'bottlenecks.csv')
        bottlenecks_df.to_csv(bottlenecks_path, index=False)
        print(f"Bottlenecks exported to {bottlenecks_path}")
    else:
        bottlenecks_path = None
        print("No bottlenecks data to export.")
    
    # Export high value paths
    high_value_paths = []
    if 'paths_by_fob_per_tonne' in results:
        for (path, fob_per_tonne) in results['paths_by_fob_per_tonne'][:30]:
            muni, exp, imp, dest = path
            weights = results['path_weights'][path]
            high_value_paths.append({
                'municipality': muni[2:],
                'exporter': exp[2:],
                'importer': imp[2:],
                'destination': dest[2:],
                'volume': weights['volume'],
                'fob': weights['fob'],
                'fob_per_tonne': fob_per_tonne
        })
    
    paths_df = pd.DataFrame(high_value_paths)
    paths_path = os.path.join(output_dir, 'high_value_paths.csv')
    paths_df.to_csv(paths_path, index=False)
    print(f"High value paths exported to {paths_path}")
    
    # Export potential new connections
    potential_connections = []
    if 'bipartite_results' in results and 'potential_connections' in results['bipartite_results']:
        for muni, imp, gain in results['bipartite_results']['potential_connections'][:30]:
            potential_connections.append({
                'municipality': muni[2:],
                'importer': imp[2:],
                'potential_gain_per_tonne': gain
            })
        
        connections_df = pd.DataFrame(potential_connections)
        connections_path = os.path.join(output_dir, 'potential_connections.csv')
        connections_df.to_csv(connections_path, index=False)
        print(f"Potential new connections exported to {connections_path}")
    else:
        connections_path = None
        print("No potential connections data to export.")
    
    # Export top opportunities
    top_opportunities = []
    if 'bipartite_results' in results and 'top_opportunities' in results['bipartite_results']:
        for muni, imp, score, gain, opp_type in results['bipartite_results']['top_opportunities'][:30]:
            top_opportunities.append({
                'municipality': muni[2:],
                'importer': imp[2:],
                'potential_gain_per_tonne': gain,
                'score': score,
                'opportunity_type': opp_type
            })
        
        opportunities_df = pd.DataFrame(top_opportunities)
        opportunities_path = os.path.join(output_dir, 'top_opportunities.csv')
        opportunities_df.to_csv(opportunities_path, index=False)
        print(f"Top opportunities exported to {opportunities_path}")
    else:
        opportunities_path = None
    
    # Export broader market analysis results
    broader_market_exports = {}
    
    # Export volume data by entity type
    if 'broader_market_results' in results:
        # Top municipalities by volume
        if 'top_municipalities' in results['broader_market_results']:
            muni_data = []
            for muni, vol in results['broader_market_results']['top_municipalities']:
                avg_price = 0
                if 'municipality_fob' in results and muni in results['municipality_fob'] and vol > 0:
                    avg_price = results['municipality_fob'][muni] / vol
                
                muni_data.append({
                    'municipality': muni[2:],
                    'volume': vol,
                    'avg_price': avg_price
                })
            
            muni_df = pd.DataFrame(muni_data)
            muni_path = os.path.join(output_dir, 'municipality_volumes.csv')
            muni_df.to_csv(muni_path, index=False)
            broader_market_exports['municipality_volumes'] = muni_path
            print(f"Municipality volumes exported to {muni_path}")
        
        # Export high volume, low price municipalities
        if 'high_vol_low_price' in results['broader_market_results']:
            hvlp_data = []
            for muni, vol, price in results['broader_market_results']['high_vol_low_price']:
                hvlp_data.append({
                    'municipality': muni[2:],
                    'volume': vol,
                    'price': price
                })
            
            if hvlp_data:
                hvlp_df = pd.DataFrame(hvlp_data)
                hvlp_path = os.path.join(output_dir, 'high_vol_low_price.csv')
                hvlp_df.to_csv(hvlp_path, index=False)
                broader_market_exports['high_vol_low_price'] = hvlp_path
                print(f"High-volume, low-price municipalities exported to {hvlp_path}")
        
        # Export market concentration data
        if 'muni_connectedness_levels' in results:
            concentration_data = []
            for level, count in results['muni_connectedness_levels'].items():
                concentration_data.append({
                    'market_type': level,
                    'municipality_count': count
                })
            
            if concentration_data:
                concentration_df = pd.DataFrame(concentration_data)
                concentration_path = os.path.join(output_dir, 'market_concentration.csv')
                concentration_df.to_csv(concentration_path, index=False)
                broader_market_exports['market_concentration'] = concentration_path
                print(f"Market concentration data exported to {concentration_path}")
    
    # Export segment analysis
    if 'municipality_segment_data' in results and results['municipality_segment_data']:
        # Export municipality dominant segment data
        if 'muni_dominant_segment' in results['municipality_segment_data']:
            seg_data = []
            muni_dom_segment = results['municipality_segment_data']['muni_dominant_segment']
            
            for muni, segment in muni_dom_segment.items():
                seg_data.append({
                    'municipality': muni[2:],
                    'dominant_segment': segment
                })
            
            if seg_data:
                seg_df = pd.DataFrame(seg_data)
                seg_path = os.path.join(output_dir, 'municipality_segments.csv')
                seg_df.to_csv(seg_path, index=False)
                print(f"Municipality segment data exported to {seg_path}")
    
    # Export tariff analysis results
    tariff_exports = {}
    
    # 1. Export US market impact data
    if 'tariff_analysis' in results and 'us_market_impacts' in results['tariff_analysis']:
        us_impact = results['tariff_analysis']['us_market_impacts']
        us_impact_df = pd.DataFrame([us_impact])
        us_impact_path = os.path.join(output_dir, 'us_tariff_impact.csv')
        us_impact_df.to_csv(us_impact_path, index=False)
        tariff_exports['us_impact'] = us_impact_path
        print(f"US tariff impact exported to {us_impact_path}")
    
    # 2. Export farmer income impact data
    if 'tariff_analysis' in results and 'farmer_income_impacts' in results['tariff_analysis']:
        farmer_impacts = []
        for muni, impact in results['tariff_analysis']['farmer_income_impacts'].items():
            impact_data = {
                'municipality': muni,
                'original_fob': impact['original_fob'],
                'tariff_fob': impact['tariff_fob'],
                'volume': impact['volume'],
                'pct_change': impact['pct_change'],
                'income_change_per_tonne': impact['income_change_per_tonne']
            }
            farmer_impacts.append(impact_data)
        
        if farmer_impacts:
            farmer_impact_df = pd.DataFrame(farmer_impacts)
            farmer_impact_path = os.path.join(output_dir, 'farmer_tariff_impacts.csv')
            farmer_impact_df.to_csv(farmer_impact_path, index=False)
            tariff_exports['farmer_impacts'] = farmer_impact_path
            print(f"Farmer tariff impacts exported to {farmer_impact_path}")
    
    # 3. Export market segment impact data
    if ('tariff_analysis' in results and 'market_segment_impacts' in results['tariff_analysis'] and
            results['tariff_analysis']['market_segment_impacts']):
        segment_impacts = []
        for segment, impact in results['tariff_analysis']['market_segment_impacts'].items():
            segment_impacts.append({
                'segment': segment,
                'original_fob': impact['original_fob'],
                'tariff_fob': impact['tariff_fob'],
                'pct_change': impact['pct_change']
            })
        
        if segment_impacts:
            segment_impact_df = pd.DataFrame(segment_impacts)
            segment_impact_path = os.path.join(output_dir, 'segment_tariff_impacts.csv')
            segment_impact_df.to_csv(segment_impact_path, index=False)
            tariff_exports['segment_impacts'] = segment_impact_path
            print(f"Segment tariff impacts exported to {segment_impact_path}")
    
    # 4. Export summary of tariff impact
    if 'tariff_impact' in results:
        tariff_summary = {
            'affected_municipalities': len(results['tariff_impact']['affected_municipalities']),
            'affected_exporters': len(results['tariff_impact']['affected_exporters']),
            'affected_importers': len(results['tariff_impact']['affected_importers']),
            'total_original_fob': results['tariff_impact']['total_original_fob'],
            'total_tariff_fob': results['tariff_impact']['total_tariff_fob'],
            'total_tariff_amount': results['tariff_impact']['total_tariff_amount']
        }
        
        tariff_summary_df = pd.DataFrame([tariff_summary])
        tariff_summary_path = os.path.join(output_dir, 'tariff_summary.csv')
        tariff_summary_df.to_csv(tariff_summary_path, index=False)
        tariff_exports['tariff_summary'] = tariff_summary_path
        print(f"Tariff summary exported to {tariff_summary_path}")
    
    # Return paths to the exported files
    return {
        'bottlenecks_path': bottlenecks_path,
        'paths_path': paths_path,
        'connections_path': connections_path,
        'opportunities_path': opportunities_path if 'opportunities_path' in locals() else None,
        'broader_market_exports': broader_market_exports,
        'tariff_exports': tariff_exports
    }

def visualize_network(G, node_layers, bottlenecks, partition):
    """
    Create a visualization of the network with bottlenecks highlighted.
    """
    print("\nGenerating network visualization...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Create a simplified graph for visualization (too many nodes makes it unreadable)
    # Focus on high-volume paths
    G_vis = nx.DiGraph()
    
    # Add nodes with significant volume or that are bottlenecks
    bottleneck_nodes = [b['node'] for b in bottlenecks] if bottlenecks else []
    
    # Add nodes
    for node in G.nodes():
        layer = node_layers[node]
        
        # Add node if it's a bottleneck or a key player
        if node in bottleneck_nodes or node in G and sum([G[node][succ]['weight'] for succ in G.successors(node)]) > 10:
            G_vis.add_node(node, layer=layer)
    
    # Add edges with significant weight
    for u, v, data in G.edges(data=True):
        if u in G_vis and v in G_vis and data['weight'] > 5:  # Only significant connections
            G_vis.add_edge(u, v, weight=data['weight'])
    
    # Check if graph is too large for visualization
    if len(G_vis) > 100:
        # Further simplify by keeping only top nodes by volume
        nodes_by_layer = defaultdict(list)
        for node in G_vis.nodes():
            layer = node_layers[node]
            vol = 0
            if layer < 4:  # For all except destinations
                vol = sum([G_vis[node][succ]['weight'] for succ in G_vis.successors(node)]) if G_vis.successors(node) else 0
            else:  # For destinations
                vol = sum([G_vis[pred][node]['weight'] for pred in G_vis.predecessors(node)]) if G_vis.predecessors(node) else 0
            
            nodes_by_layer[layer].append((node, vol))
        
        # Keep top nodes in each layer
        keep_nodes = set(bottleneck_nodes)  # Always keep bottlenecks
        for layer, nodes in nodes_by_layer.items():
            top_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)[:15]  # Keep top 15 per layer
            keep_nodes.update([n[0] for n in top_nodes])
        
        # Create new simplified graph
        G_vis_simple = nx.DiGraph()
        for node in keep_nodes:
            G_vis_simple.add_node(node, layer=node_layers[node])
        
        # Add edges between kept nodes
        for u, v, data in G_vis.edges(data=True):
            if u in keep_nodes and v in keep_nodes:
                G_vis_simple.add_edge(u, v, weight=data['weight'])
        
        G_vis = G_vis_simple
    
    # Set up the visualization
    plt.figure(figsize=(18, 14))
    
    # Define layout
    pos = nx.multipartite_layout(G_vis, subset_key="layer")
    
    # Define node sizes based on importance
    node_sizes = []
    for node in G_vis.nodes():
        layer = node_layers[node]
        if node in bottleneck_nodes:
            node_sizes.append(300)  # Bottlenecks are larger
        else:
            node_sizes.append(100)  # Regular nodes
    
    # Define node colors by layer
    layer_colors = {
        1: '#5DA637',  # Green for municipalities
        2: '#D95F02',  # Orange for exporters
        3: '#7570B3',  # Purple for importers
        4: '#1B9E77'   # Teal for destinations
    }
    
    node_colors = [layer_colors[node_layers[node]] for node in G_vis.nodes()]
    
    # Define edge colors and widths
    edge_colors = []
    edge_widths = []
    
    for u, v in G_vis.edges():
        if u in bottleneck_nodes or v in bottleneck_nodes:
            edge_colors.append('red')  # Highlight bottleneck connections
            edge_widths.append(2.0)
        else:
            edge_colors.append('gray')
            edge_widths.append(0.8)
    
    # Draw the network
    nx.draw_networkx_nodes(G_vis, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    
    # Draw edges with alpha for better visibility
    nx.draw_networkx_edges(G_vis, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6, 
                          connectionstyle='arc3,rad=0.1')  # Curved edges
    
    # Add labels to bottlenecks and key nodes
    labels = {}
    for node in G_vis.nodes():
        if node in bottleneck_nodes or node_layers[node] == 2 and G_vis.degree(node) > 5:  # Bottlenecks and key exporters
            labels[node] = node[2:]  # Remove the prefix
    
    nx.draw_networkx_labels(G_vis, pos, labels=labels, font_size=8, font_weight='bold')
    
    # Add legend for node types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=layer_colors[1], markersize=10, label='Municipality'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=layer_colors[2], markersize=10, label='Exporter'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=layer_colors[3], markersize=10, label='Importer'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=layer_colors[4], markersize=10, label='Destination')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add title and adjust layout
    plt.title('Coffee Supply Chain Network', fontsize=16)
    plt.tight_layout()
    
    # Save the visualization
    vis_path = os.path.join(OUTPUT_DIR, 'network_visualization.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network visualization saved to {vis_path}")
    return vis_path

def apply_tariff(G, node_layers, path_weights, tariff_rate=0.1, source_countries=["Brazil", "Colombia"], target_country="United States"):
    """
    Apply a tariff to coffee imports from specified source countries to a target country.
    
    Parameters:
    - G: The network graph
    - node_layers: Dictionary mapping nodes to their layer
    - path_weights: Dictionary with weights for each path
    - tariff_rate: The tariff rate to apply (default: 0.1 for 10%)
    - source_countries: List of countries to apply the tariff to (default: Brazil and Colombia)
    - target_country: Country applying the tariff (default: United States)
    
    Returns:
    - Modified G: The modified network with tariff applied
    - Modified path_weights: Updated path weights with tariff effects
    - tariff_impact: Dictionary with analysis of tariff impact
    """
    print(f"\n== Applying {tariff_rate*100:.0f}% tariff by {target_country} on coffee from {', '.join(source_countries)} ==")
    
    # Track changes for impact analysis
    tariff_impact = {
        'affected_paths': [],
        'total_original_fob': 0,
        'total_tariff_fob': 0,
        'total_tariff_amount': 0,
        'affected_municipalities': set(),
        'affected_exporters': set(),
        'affected_importers': set(),
        'price_changes': defaultdict(float)  # FOB change per municipality
    }
    
    # Create copies of the input data to modify
    G_with_tariff = G.copy()
    path_weights_with_tariff = defaultdict(lambda: {'volume': 0, 'fob': 0})
    for path, weights in path_weights.items():
        path_weights_with_tariff[path]['volume'] = weights['volume']
        path_weights_with_tariff[path]['fob'] = weights['fob']
    
    # Process each path to apply tariffs
    for path, weights in path_weights.items():
        municipality, exporter, importer, destination = path
        
        # Check if this path involves the target country and source countries
        source_country = None
        for src_country in source_countries:
            if src_country in exporter:  # Assuming exporter name contains country
                source_country = src_country
                break
        
        if source_country and target_country in destination:
            # This path is affected by the tariff
            original_fob = weights['fob']
            # Apply tariff - increase the FOB cost by tariff rate
            tariff_fob = original_fob * (1 + tariff_rate)
            tariff_amount = original_fob * tariff_rate
            
            # Update the path weights
            path_weights_with_tariff[path]['fob'] = tariff_fob
            
            # Update edges in the graph
            # Edge from importer to destination
            if G_with_tariff.has_edge(importer, destination):
                G_with_tariff[importer][destination]['fob'] = G_with_tariff[importer][destination].get('fob', 0) * (1 + tariff_rate)
            
            # Track for impact analysis
            tariff_impact['affected_paths'].append(path)
            tariff_impact['total_original_fob'] += original_fob
            tariff_impact['total_tariff_fob'] += tariff_fob
            tariff_impact['total_tariff_amount'] += tariff_amount
            tariff_impact['affected_municipalities'].add(municipality)
            tariff_impact['affected_exporters'].add(exporter)
            tariff_impact['affected_importers'].add(importer)
            
            # Track price impact on municipality
            muni_name = municipality[2:]  # Remove the "M:" prefix
            tariff_impact['price_changes'][muni_name] -= tariff_amount  # Negative because farmers likely bear some cost
    
    # Summarize tariff impact
    print(f"Tariff affects {len(tariff_impact['affected_paths'])} trade paths")
    print(f"Affected municipalities: {len(tariff_impact['affected_municipalities'])}")
    print(f"Affected exporters: {len(tariff_impact['affected_exporters'])}")
    print(f"Affected importers: {len(tariff_impact['affected_importers'])}")
    print(f"Total original FOB value: ${tariff_impact['total_original_fob']:,.2f}")
    print(f"Total FOB value with tariff: ${tariff_impact['total_tariff_fob']:,.2f}")
    print(f"Total tariff amount: ${tariff_impact['total_tariff_amount']:,.2f}")
    
    return G_with_tariff, path_weights_with_tariff, tariff_impact

def analyze_tariff_impact(G, G_with_tariff, node_layers, path_weights, path_weights_with_tariff, tariff_impact, municipality_segment_data=None):
    """
    Analyze how the applied tariff impacts coffee prices and farmer incomes.
    
    Parameters:
    - G: Original network before tariff
    - G_with_tariff: Network after tariff application
    - node_layers: Dictionary mapping nodes to their layer
    - path_weights: Original path weights
    - path_weights_with_tariff: Path weights after tariff
    - tariff_impact: Tariff impact data from apply_tariff
    - municipality_segment_data: Optional data about municipalities by market segment
    
    Returns:
    - Analysis results dictionary
    """
    print("\n== Analyzing Tariff Impact ==")
    
    results = {
        'price_impacts': {},
        'farmer_income_impacts': {},
        'market_segment_impacts': {},
        'us_market_impacts': {},
        'exporter_impacts': {},
        'importer_impacts': {}
    }
    
    # Calculate impact on US coffee prices
    us_paths = [(path, weights) for path, weights in path_weights.items() 
                if "United States" in path[3]]  # path[3] is the destination
    
    us_tariff_paths = [(path, path_weights_with_tariff[path]) for path, _ in us_paths]
    
    if us_paths:
        original_us_total_fob = sum(weights['fob'] for _, weights in us_paths)
        original_us_total_volume = sum(weights['volume'] for _, weights in us_paths)
        
        tariff_us_total_fob = sum(weights['fob'] for _, weights in us_tariff_paths)
        tariff_us_total_volume = sum(weights['volume'] for _, weights in us_tariff_paths)
        
        # Calculate average FOB per tonne
        original_avg_fob = original_us_total_fob / original_us_total_volume if original_us_total_volume > 0 else 0
        tariff_avg_fob = tariff_us_total_fob / tariff_us_total_volume if tariff_us_total_volume > 0 else 0
        
        price_increase_pct = ((tariff_avg_fob - original_avg_fob) / original_avg_fob * 100) if original_avg_fob > 0 else 0
        
        results['us_market_impacts'] = {
            'original_total_fob': original_us_total_fob,
            'tariff_total_fob': tariff_us_total_fob,
            'original_total_volume': original_us_total_volume,
            'tariff_total_volume': tariff_us_total_volume,
            'original_avg_fob': original_avg_fob,
            'tariff_avg_fob': tariff_avg_fob,
            'price_increase_pct': price_increase_pct
        }
        
        print(f"US Coffee Market Impact:")
        print(f"  Original average FOB: ${original_avg_fob:.2f}/tonne")
        print(f"  Tariff average FOB: ${tariff_avg_fob:.2f}/tonne")
        print(f"  Price increase: {price_increase_pct:.2f}%")
    
    # Calculate impact on farmer income
    municipality_original_fob = defaultdict(float)
    municipality_tariff_fob = defaultdict(float)
    municipality_original_volume = defaultdict(float)
    
    # Calculate original FOB and volume per municipality
    for (muni, _, _, _), weights in path_weights.items():
        municipality_original_fob[muni] += weights['fob']
        municipality_original_volume[muni] += weights['volume']
    
    # Calculate tariff FOB per municipality
    for (muni, _, _, _), weights in path_weights_with_tariff.items():
        municipality_tariff_fob[muni] += weights['fob']
    
    # Calculate income impacts for affected municipalities
    farmer_income_impacts = {}
    for muni in tariff_impact['affected_municipalities']:
        original_fob = municipality_original_fob[muni]
        tariff_fob = municipality_tariff_fob[muni]
        volume = municipality_original_volume[muni]
        
        # Calculate percentage change in income
        pct_change = ((tariff_fob - original_fob) / original_fob * 100) if original_fob > 0 else 0
        
        # Calculate income change per tonne
        income_change_per_tonne = (tariff_fob - original_fob) / volume if volume > 0 else 0
        
        farmer_income_impacts[muni[2:]] = {  # Remove the "M:" prefix
            'original_fob': original_fob,
            'tariff_fob': tariff_fob,
            'volume': volume,
            'pct_change': pct_change,
            'income_change_per_tonne': income_change_per_tonne
        }
    
    # Sort municipalities by percentage income change (most negative first)
    sorted_income_impacts = sorted(farmer_income_impacts.items(), 
                                 key=lambda x: x[1]['pct_change'])
    
    results['farmer_income_impacts'] = farmer_income_impacts
    
    print("\nFarmer Income Impact:")
    print(f"  Number of affected municipalities: {len(farmer_income_impacts)}")
    
    # Report the most affected municipalities
    if sorted_income_impacts:
        most_affected = sorted_income_impacts[:5]
        print("  Most negatively affected municipalities:")
        for muni, impact in most_affected:
            print(f"    {muni}: {impact['pct_change']:.2f}% income change, ${impact['income_change_per_tonne']:.2f}/tonne")
    
    # Analyze impact by market segment if segment data is available
    if municipality_segment_data:
        segment_impacts = {
            'commodity': {'original_fob': 0, 'tariff_fob': 0, 'pct_change': 0},
            'premium': {'original_fob': 0, 'tariff_fob': 0, 'pct_change': 0},
            'commercial_specialty': {'original_fob': 0, 'tariff_fob': 0, 'pct_change': 0},
            'high_specialty': {'original_fob': 0, 'tariff_fob': 0, 'pct_change': 0}
        }
        
        # Sum up FOB by segment
        for muni, muni_data in municipality_segment_data.items():
            muni_node = f"M:{muni}"
            dominant_segment = muni_data.get('dominant_segment')
            
            if dominant_segment and muni_node in municipality_original_fob and muni_node in municipality_tariff_fob:
                segment_impacts[dominant_segment]['original_fob'] += municipality_original_fob[muni_node]
                segment_impacts[dominant_segment]['tariff_fob'] += municipality_tariff_fob[muni_node]
        
        # Calculate percentage changes
        for segment, impact in segment_impacts.items():
            if impact['original_fob'] > 0:
                impact['pct_change'] = (impact['tariff_fob'] - impact['original_fob']) / impact['original_fob'] * 100
        
        results['market_segment_impacts'] = segment_impacts
        
        print("\nImpact by Market Segment:")
        for segment, impact in segment_impacts.items():
            if impact['original_fob'] > 0:
                print(f"  {segment.replace('_', ' ').title()}: {impact['pct_change']:.2f}% income change")
    
    return results

def visualize_tariff_impact(G, G_with_tariff, node_layers, tariff_impact, farmer_income_impacts, output_dir):
    """
    Visualize the impact of tariffs on the coffee trade network.
    
    Parameters:
    - G: Original network graph
    - G_with_tariff: Network graph with tariff applied
    - node_layers: Dictionary mapping nodes to their layer
    - tariff_impact: Dictionary with tariff impact data
    - farmer_income_impacts: Dictionary with income impacts per municipality
    - output_dir: Directory to save visualizations
    
    Returns:
    - Dictionary with paths to generated visualizations
    """
    print("\nCreating tariff impact visualizations...")
    
    output_paths = {}
    
    # 1. Plot affected trade routes on a map-like visualization
    plt.figure(figsize=(12, 10))
    
    # Create positions for network nodes based on layer
    pos = {}
    
    # Define layer positions (x-axis)
    layer_positions = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}
    
    # Sort nodes by volume to position higher volume nodes more centrally
    nodes_by_layer = {}
    for node, layer in node_layers.items():
        if node not in nodes_by_layer:
            nodes_by_layer[layer] = []
        nodes_by_layer[layer].append(node)
    
    # Position nodes within each layer
    for layer, nodes in nodes_by_layer.items():
        if layer in layer_positions:
            x = layer_positions[layer]
            # Space nodes evenly along y-axis
            num_nodes = len(nodes)
            for i, node in enumerate(nodes):
                y = 0.1 + (i / max(1, num_nodes - 1)) * 0.8  # Distribute between 0.1 and 0.9
                pos[node] = (x, y)
    
    # Filter nodes to only include those with positions
    nodes_with_pos = set(pos.keys())
    
    # Filter node lists to only include nodes with positions
    unaffected_nodes = [n for n in G.nodes() if n not in tariff_impact['affected_municipalities'] and 
                        n not in tariff_impact['affected_exporters'] and 
                        n not in tariff_impact['affected_importers'] and
                        n in nodes_with_pos]
    
    affected_municipalities = [n for n in tariff_impact['affected_municipalities'] if n in nodes_with_pos]
    affected_exporters = [n for n in tariff_impact['affected_exporters'] if n in nodes_with_pos]
    affected_importers = [n for n in tariff_impact['affected_importers'] if n in nodes_with_pos]
    
    # Draw the original network (gray)
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=unaffected_nodes,
                          node_color='lightgray', 
                          node_size=50, 
                          alpha=0.5)
    
    # Draw affected municipalities
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=affected_municipalities,
                          node_color='green', 
                          node_size=100, 
                          alpha=0.7)
    
    # Draw affected exporters
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=affected_exporters,
                          node_color='blue', 
                          node_size=100, 
                          alpha=0.7)
    
    # Draw affected importers
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=affected_importers,
                          node_color='red', 
                          node_size=100, 
                          alpha=0.7)
    
    # Draw edges for affected paths (only if both ends have positions)
    affected_edges = []
    for path in tariff_impact['affected_paths']:
        muni, exp, imp, dest = path
        if muni in nodes_with_pos and exp in nodes_with_pos:
            affected_edges.append((muni, exp))
        if exp in nodes_with_pos and imp in nodes_with_pos:
            affected_edges.append((exp, imp))
        if imp in nodes_with_pos and dest in nodes_with_pos:
            affected_edges.append((imp, dest))
    
    nx.draw_networkx_edges(G, pos, 
                          edgelist=affected_edges,
                          width=1.5, 
                          edge_color='red',
                          alpha=0.6)
    
    # Draw other edges in light gray (only if both ends have positions)
    other_edges = [(u, v) for u, v in G.edges() if (u, v) not in affected_edges 
                   and u in nodes_with_pos and v in nodes_with_pos]
    
    nx.draw_networkx_edges(G, pos, 
                          edgelist=other_edges,
                          width=0.5, 
                          edge_color='lightgray',
                          alpha=0.3)
    
    # Add labels for key nodes (only those with positions)
    key_nodes = set()
    key_nodes.update(affected_municipalities[:5])
    key_nodes.update(affected_exporters[:5])
    key_nodes.update(affected_importers[:5])
    
    # Create labels (remove the prefixes for readability)
    labels = {node: node[2:] for node in key_nodes if node in nodes_with_pos}
    
    nx.draw_networkx_labels(G, pos, 
                           labels=labels, 
                           font_size=8, 
                           font_color='black')
    
    plt.title('Coffee Trade Network - Paths Affected by 10% US Tariff on Brazil/Colombia')
    plt.axis('off')
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Affected Municipalities'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Affected Exporters'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Affected Importers'),
        plt.Line2D([0], [0], color='red', lw=2, label='Affected Trade Routes')
    ]
    
    plt.legend(handles=legend_elements, loc='lower right')
    
    network_path = os.path.join(output_dir, 'tariff_impact_network.png')
    plt.savefig(network_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_paths['network'] = network_path
    
    # 2. Plot income impact on farmers
    if farmer_income_impacts:
        # Extract data for visualization
        municipalities = []
        pct_changes = []
        volume_sizes = []
        
        for muni, impact in farmer_income_impacts.items():
            if impact['volume'] > 0:  # Only include municipalities with volume
                municipalities.append(muni)
                pct_changes.append(impact['pct_change'])
                # Scale the volume for visualization
                volume_sizes.append(max(20, min(500, impact['volume'] / 10)))
        
        # Sort by percentage change
        sorted_indices = np.argsort(pct_changes)
        sorted_municipalities = [municipalities[i] for i in sorted_indices]
        sorted_pct_changes = [pct_changes[i] for i in sorted_indices]
        sorted_volume_sizes = [volume_sizes[i] for i in sorted_indices]
        
        # Take top and bottom 15 for visibility
        top_indices = sorted_indices[-15:]
        bottom_indices = sorted_indices[:15]
        
        # Combine the indices
        plot_indices = sorted(list(top_indices) + list(bottom_indices))
        
        plot_municipalities = [municipalities[i] for i in plot_indices]
        plot_pct_changes = [pct_changes[i] for i in plot_indices]
        plot_volume_sizes = [volume_sizes[i] for i in plot_indices]
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Use colors to indicate positive or negative impact
        colors = ['red' if pct < 0 else 'green' for pct in plot_pct_changes]
        
        y_pos = np.arange(len(plot_municipalities))
        bars = plt.barh(y_pos, plot_pct_changes, color=colors, alpha=0.7)
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.yticks(y_pos, plot_municipalities, fontsize=10)
        plt.xlabel('Percentage Change in Income (%)', fontsize=12)
        plt.title('Impact of 10% US Tariff on Brazilian/Colombian Coffee - Farmer Income Changes', fontsize=14)
        
        # Add value labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width + 0.5 if width >= 0 else width - 2
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{plot_pct_changes[i]:.1f}%', 
                    va='center', fontsize=9)
        
        farmers_path = os.path.join(output_dir, 'tariff_impact_farmers.png')
        plt.savefig(farmers_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths['farmers'] = farmers_path
    
    # 3. Create a summary visualization showing the overall impact
    plt.figure(figsize=(10, 6))
    
    # Summary metrics
    metrics = ['Affected Municipalities', 'Affected Exporters', 'Affected Importers']
    values = [
        len(tariff_impact['affected_municipalities']),
        len(tariff_impact['affected_exporters']),
        len(tariff_impact['affected_importers'])
    ]
    
    # Plot summary bar chart
    plt.bar(metrics, values, color=['green', 'blue', 'red'], alpha=0.7)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.ylabel('Count', fontsize=12)
    plt.title('Entities Affected by 10% US Tariff on Brazilian/Colombian Coffee', fontsize=14)
    
    summary_path = os.path.join(output_dir, 'tariff_impact_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_paths['summary'] = summary_path
    
    print(f"Created tariff impact visualizations in {output_dir}")
    return output_paths

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load the data
    coffee_data = load_data()
    print(f"Loaded {len(coffee_data)} trade records.")
    
    # Create multi-layer network
    G, node_layers, path_weights = create_multilayer_network(coffee_data)
    
    # Analyze dominant paths but don't display top paths - we'll focus only on market segments
    paths_by_volume, paths_by_fob, paths_by_fob_per_tonne, segment_paths, municipality_segment_data = analyze_dominant_paths(G, path_weights)
    
    # Skip centrality analysis and community detection - not directly tied to farmer income
    # centrality_results = analyze_centrality(G, node_layers)
    # partition, sorted_communities, community_layers = detect_communities(G, node_layers)
    
    # Skip detailed bottleneck identification but get connectedness information for later analysis
    print("\nIdentifying bottlenecks and connectedness...(simplified)")
    muni_connectedness_levels = {}
    most_concentrated = []
    price_stats = {}
    muni_by_connectedness = {}
    bottlenecks = []
    
    # Calculate basic municipality connectivity
    for node in G.nodes():
        if node.startswith('M:'):  # Municipality node
            muni_connectedness_levels[node] = len(list(G.neighbors(node)))
    
    # Calculate FOB per municipality for basic price analysis
    municipality_fob = defaultdict(float)
    municipality_volumes = defaultdict(float)
    for (muni, exp, imp, _), weights in path_weights.items():
        if weights['volume'] > 0:
            municipality_volumes[muni] += weights['volume']
            municipality_fob[muni] += weights['fob']
    
    # Create and analyze bipartite network (but skip diversity metrics)
    B = create_bipartite_network(coffee_data)
    bipartite_results = analyze_bipartite_matching(B, municipality_segment_data, muni_by_connectedness)
    
    # Analyze broader market trends with a focus on improvement opportunities
    print("\n== Simplified Broader Market Analysis ==")
    print("Focusing on actionable opportunities for improvement...")
    
    # Calculate average FOB per municipality
    muni_avg_fob = {}
    for muni in municipality_volumes:
        if municipality_volumes[muni] > 0 and municipality_fob[muni] > 0:
            muni_avg_fob[muni] = municipality_fob[muni] / municipality_volumes[muni]
    
    # Find municipalities with low average FOB
    low_fob_munis = sorted(muni_avg_fob.items(), key=lambda x: x[1])[:20] if muni_avg_fob else []
    
    print("\nMunicipalities with lowest FOB prices (priority intervention targets):")
    for i, (muni, avg_fob) in enumerate(low_fob_munis[:10]):
        volume = municipality_volumes[muni]
        print(f"{i+1}. {muni[2:]}: ${avg_fob:.2f}/tonne average, Volume: {volume:.2f} tonnes")
    
    # Focus on price improvement opportunities from market segment analysis
    if hasattr(municipality_segment_data, 'items'):  # Check if data exists
        print("\n== Market Segment Price Improvement Potential ==")
        
        # Note on removed market segment participation analysis
        print("== NOTE: Municipality Market Segment Analysis Modified ==\n")
        print("Market segment participation analysis has been modified")
        print("to use fixed price thresholds instead of percentile-based segmentation.")
        print("This provides a more accurate representation of coffee market segments based on industry standards.\n")
        
        # Simplified analysis based on fixed price thresholds
        print("Market diversification analysis:")
        
        # Count diversification with fixed thresholds
        commodity_only = 0
        multi_segment = 0
        total_munis = len(municipality_segment_data)
        
        for muni, data in municipality_segment_data.items():
            if 'segments' in data:
                if len(data['segments']) == 1 and 'commodity' in data['segments']:
                    commodity_only += 1
                elif len(data['segments']) > 1:
                    multi_segment += 1
        
        if total_munis > 0:
            pct_commodity = (commodity_only / total_munis) * 100
            pct_multi = (multi_segment / total_munis) * 100
            print(f"  Municipalities selling exclusively commodity coffee: {commodity_only} ({pct_commodity:.1f}%)")
            print(f"  Municipalities selling in multiple market segments: {multi_segment} ({pct_multi:.1f}%)\n")
            
            # Calculate average price premium for higher segments vs commodity
            premium_pcts = []
            for muni, data in municipality_segment_data.items():
                if 'avg_prices' in data and 'commodity' in data['avg_prices']:
                    commodity_price = data['avg_prices']['commodity']
                    higher_segments = {s: p for s, p in data['avg_prices'].items() 
                                     if s in ['premium', 'commercial_specialty', 'high_specialty']}
                    if higher_segments:
                        avg_higher_price = sum(higher_segments.values()) / len(higher_segments)
                        premium_pct = ((avg_higher_price / commodity_price) - 1) * 100
                        premium_pcts.append(premium_pct)
            
            if premium_pcts:
                overall_premium = sum(premium_pcts) / len(premium_pcts)
                print(f"\nPrice improvement potential:")
                print(f"Average price premium for higher segments: +{overall_premium:.1f}% above commodity prices")
                print(f"This represents the typical price improvement from upgrading from commodity to higher segments")
    
    # Calculate and export basic results
    results = {
        'low_fob_municipalities': low_fob_munis,
        'municipality_fob': municipality_fob,
        'municipality_volumes': municipality_volumes,
        'municipality_segment_data': municipality_segment_data,
        'path_weights': path_weights,
        'paths_by_fob_per_tonne': paths_by_fob_per_tonne
    }
    
    export_paths = export_results(results, OUTPUT_DIR)
    
    print("\nStreamlined network analysis complete. Key findings:")
    print("1. Identified municipalities with lowest coffee prices as priority intervention targets")
    print("2. Analyzed market segment participation and price improvement potential")
    print("3. Calculated potential price gains from upgrading market segments")
    
    print(f"\nCheck {OUTPUT_DIR} directory for detailed results.")