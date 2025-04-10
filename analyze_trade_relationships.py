import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Load optimization results
print("Loading optimization results...")
try:
    results_df = pd.read_csv('optimized_coffee_trade.csv')
    print(f"Loaded {len(results_df)} trade relationships")
except FileNotFoundError:
    print("Error: optimized_coffee_trade.csv not found. Run coffee_trade_optimization.py first.")
    exit(1)

# Load original dataset for comparison
try:
    original_df = pd.read_csv('brazil-coffee-v2.5.1-2024-04-26.csv')
    print(f"Loaded original dataset with {len(original_df)} records")
except FileNotFoundError:
    print("Warning: Original dataset not found. Some analyses will be limited.")
    original_df = None

# Create output directory for visualizations
import os
if not os.path.exists('analysis_results'):
    os.makedirs('analysis_results')

# Format values as dollars
def format_dollars(x, pos):
    return f'${x:,.0f}'

dollar_formatter = FuncFormatter(format_dollars)

# 1. Analyze relationship stability
print("\nAnalyzing relationship stability...")

# Count relationships by type
active_relationships = results_df[results_df['relationship_active']].shape[0]
total_relationships = results_df.shape[0]
existing_relationships = results_df[results_df['is_existing_relationship']].shape[0]
new_relationships = results_df[results_df['is_new_relationship']].shape[0]
maintained_relationships = results_df[(results_df['is_existing_relationship']) & (results_df['relationship_active'])].shape[0]
dropped_relationships = results_df[(results_df['is_existing_relationship']) & (~results_df['relationship_active'])].shape[0]

print(f"Original trade relationships: {existing_relationships}")
print(f"Relationships maintained after optimization: {maintained_relationships} ({100*maintained_relationships/existing_relationships:.2f}%)")
print(f"Relationships dropped after optimization: {dropped_relationships} ({100*dropped_relationships/existing_relationships:.2f}%)")
print(f"New relationships created: {new_relationships}")
print(f"Total active relationships after optimization: {active_relationships}")

# Create pie chart of relationship status
plt.figure(figsize=(12, 8))
plt.pie([maintained_relationships, dropped_relationships, new_relationships], 
        labels=['Maintained', 'Dropped', 'New'],
        autopct='%1.1f%%',
        colors=['#4CAF50', '#F44336', '#2196F3'],
        explode=(0.05, 0.05, 0.05),
        startangle=90,
        shadow=True)
plt.title('Trade Relationship Status After Optimization')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.savefig('analysis_results/relationship_status_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Analyze FOB value changes
print("\nAnalyzing FOB value changes...")

# Create a cleaner subset for calculating total values
existing_relationships_df = results_df[results_df['is_existing_relationship']]
active_relationships_df = results_df[results_df['relationship_active']]
new_relationships_df = results_df[results_df['is_new_relationship']]
dropped_relationships_df = results_df[(results_df['is_existing_relationship']) & (~results_df['relationship_active'])]

# Calculate total FOB before and after
total_original_fob = existing_relationships_df['original_fob'].sum()
total_optimized_fob = active_relationships_df['optimized_fob'].sum()
optimized_existing_fob = active_relationships_df[active_relationships_df['is_existing_relationship']]['optimized_fob'].sum()
optimized_new_fob = new_relationships_df['optimized_fob'].sum()

# Calculate total switching costs
total_switching_costs = results_df['switching_cost'].sum() if 'switching_cost' in results_df.columns else 0
new_relationship_costs = new_relationships_df['switching_cost'].sum() if 'switching_cost' in results_df.columns else 0
dropped_relationship_costs = dropped_relationships_df['switching_cost'].sum() if 'switching_cost' in results_df.columns else 0

# Calculate net benefit (FOB improvement minus switching costs)
net_benefit = total_optimized_fob - total_switching_costs
gross_improvement = total_optimized_fob - total_original_fob
net_improvement = net_benefit - total_original_fob

print(f"Total original FOB: ${total_original_fob:,.2f}")
print(f"Total optimized FOB (gross): ${total_optimized_fob:,.2f}")
print(f"  - From maintained relationships: ${optimized_existing_fob:,.2f} ({100*optimized_existing_fob/total_optimized_fob:.2f}%)")
print(f"  - From new relationships: ${optimized_new_fob:,.2f} ({100*optimized_new_fob/total_optimized_fob:.2f}%)")
print(f"Total switching costs: ${total_switching_costs:,.2f}")
print(f"  - New relationship costs: ${new_relationship_costs:,.2f}")
print(f"  - Dropped relationship costs: ${dropped_relationship_costs:,.2f}")
print(f"Net FOB after switching costs: ${net_benefit:,.2f}")
print(f"Gross improvement: ${gross_improvement:,.2f} ({100*gross_improvement/total_original_fob:.2f}%)")
print(f"Net improvement: ${net_improvement:,.2f} ({100*net_improvement/total_original_fob:.2f}%)")

# Create stacked bar chart of FOB values before and after optimization, including switching costs
plt.figure(figsize=(14, 10))  # Increased figure size for better spacing
x = ['Original', 'Optimized\n(Gross)', 'Optimized\n(Net)']
existing_y = [total_original_fob, optimized_existing_fob, optimized_existing_fob]
new_y = [0, optimized_new_fob, optimized_new_fob]
switching_costs_y = [0, 0, -total_switching_costs]

# Create stacked bar chart
plt.bar(x, existing_y, color='#4CAF50', label='Maintained Relationships')
plt.bar(x, new_y, bottom=existing_y, color='#2196F3', label='New Relationships')

# Add switching costs bar for the net column
if total_switching_costs > 0:
    # Plot switching costs as negative bars starting from total optimized FOB
    base_for_switching = [0, 0, total_optimized_fob]
    plt.bar(x, switching_costs_y, bottom=base_for_switching, color='#F44336', label='Switching Costs')

# Add labels and formatting
plt.title('Total FOB Value Before and After Optimization, Including Switching Costs', fontsize=16)
plt.ylabel('FOB Value (USD)', fontsize=14)
plt.gca().yaxis.set_major_formatter(dollar_formatter)
plt.legend(loc='best', fontsize=12)

# Calculate maximum value for setting y-axis limits with extra space
max_value = max(total_original_fob, total_optimized_fob) * 1.3  # Add 30% extra space at the top
plt.ylim(0, max_value)

# Add value labels on bars
for i, v in enumerate(existing_y):
    if i != 2:  # Don't label the "existing" part of the Net column separately
        plt.text(i, v/2, f'${v:,.0f}', ha='center', va='center', color='white', fontweight='bold')

for i, v in enumerate(new_y):
    if v > 0 and i != 2:  # Don't label the "new" part of the Net column separately
        plt.text(i, existing_y[i] + v/2, f'${v:,.0f}', ha='center', va='center', color='white', fontweight='bold')

# Add total values with better spacing
plt.text(0, total_original_fob + 0.05 * max_value, f'${total_original_fob:,.0f}', ha='center', va='bottom')
plt.text(1, total_optimized_fob + 0.05 * max_value, f'${total_optimized_fob:,.0f}', ha='center', va='bottom')
plt.text(2, net_benefit + 0.05 * max_value, f'${net_benefit:,.0f}', ha='center', va='bottom')

# Add percentage change annotations with better spacing
plt.annotate(f'+{100*gross_improvement/total_original_fob:.2f}%',
             xy=(1, total_optimized_fob), 
             xytext=(1.2, total_optimized_fob + 0.15 * max_value),  # Increased vertical space
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=14, ha='center')

plt.annotate(f'+{100*net_improvement/total_original_fob:.2f}%',
             xy=(2, net_benefit), 
             xytext=(2.2, net_benefit + 0.22 * max_value),  # More vertical space than the first annotation
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=14, ha='center')

if total_switching_costs > 0:
    plt.annotate(f'Switching Costs: ${total_switching_costs:,.0f}',
                 xy=(2, total_optimized_fob - total_switching_costs/2), 
                 xytext=(2.4, total_optimized_fob - total_switching_costs/2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12, ha='center')

plt.tight_layout()
plt.savefig('analysis_results/fob_value_comparison_with_switching_costs.png', dpi=300, bbox_inches='tight')
plt.close()

# Create direct comparison scatter plots for FOB and volume
print("\nCreating direct relationship comparison scatter plots...")

# Add status column for coloring
results_df['status'] = 'Not Activated'
results_df.loc[results_df['is_existing_relationship'] & results_df['relationship_active'], 'status'] = 'Maintained'
results_df.loc[results_df['is_existing_relationship'] & ~results_df['relationship_active'], 'status'] = 'Dropped'
results_df.loc[results_df['is_new_relationship'], 'status'] = 'New'

# Create scatter plot for FOB comparison
plt.figure(figsize=(12, 8))

# Define colors and sizes for different status categories
colors = {'Maintained': '#4CAF50', 'Dropped': '#F44336', 'New': '#2196F3', 'Not Activated': '#9E9E9E'}
sizes = {'Maintained': 50, 'Dropped': 50, 'New': 80, 'Not Activated': 30}

# Plot each category separately for better legend control
for status in ['Maintained', 'Dropped', 'New']:
    subset = results_df[results_df['status'] == status]
    if status == 'New':
        # New relationships have original_fob = 0
        plt.scatter(subset['original_fob'], subset['optimized_fob'], 
                   c=colors[status], s=sizes[status], alpha=0.7, label=status)
    elif status == 'Dropped':
        # Dropped relationships have optimized_fob = 0
        plt.scatter(subset['original_fob'], subset['optimized_fob'], 
                   c=colors[status], s=sizes[status], alpha=0.7, label=status)
    else:
        plt.scatter(subset['original_fob'], subset['optimized_fob'], 
                   c=colors[status], s=sizes[status], alpha=0.7, label=status)

# Add diagonal line (y=x) to show where original = optimized
max_val = max(results_df['original_fob'].max(), results_df['optimized_fob'].max())
plt.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.5, label='No Change')

plt.xlabel('Original FOB Value (USD)')
plt.ylabel('Optimized FOB Value (USD)')
plt.title('Direct Comparison of Original vs. Optimized FOB Values')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('analysis_results/fob_direct_comparison_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# Create scatter plot for volume comparison
plt.figure(figsize=(12, 8))

# Plot each category separately
for status in ['Maintained', 'Dropped', 'New']:
    subset = results_df[results_df['status'] == status]
    if status == 'New':
        # New relationships have original_volume = 0
        plt.scatter(subset['original_volume'], subset['optimized_volume'], 
                   c=colors[status], s=sizes[status], alpha=0.7, label=status)
    elif status == 'Dropped':
        # Dropped relationships have optimized_volume = 0
        plt.scatter(subset['original_volume'], subset['optimized_volume'], 
                   c=colors[status], s=sizes[status], alpha=0.7, label=status)
    else:
        plt.scatter(subset['original_volume'], subset['optimized_volume'], 
                   c=colors[status], s=sizes[status], alpha=0.7, label=status)

# Add diagonal line (y=x) to show where original = optimized
max_val = max(results_df['original_volume'].max(), results_df['optimized_volume'].max())
plt.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.5, label='No Change')

plt.xlabel('Original Volume')
plt.ylabel('Optimized Volume')
plt.title('Direct Comparison of Original vs. Optimized Volumes')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('analysis_results/volume_direct_comparison_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Analyze distribution of FOB changes for maintained relationships
print("\nAnalyzing distribution of FOB changes...")

# Only consider maintained relationships for percent change
maintained_relationships_df = results_df[(results_df['is_existing_relationship']) & (results_df['relationship_active'])]
maintained_relationships_df['percent_fob_change'] = maintained_relationships_df['percent_change']

plt.figure(figsize=(12, 6))
sns.histplot(maintained_relationships_df['percent_fob_change'], bins=30, kde=True, color='#2196F3')
plt.title('Distribution of FOB Value Changes in Maintained Relationships')
plt.xlabel('Percent Change in FOB Value')
plt.ylabel('Number of Trade Relationships')
plt.axvline(x=0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('analysis_results/fob_change_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Analyze changes in trade partner counts
print("\nAnalyzing changes in trade partner counts...")

# Analyze exporter partner changes
# Original counts
exporter_original_partners = existing_relationships_df.groupby('exporter').size().reset_index(name='original_count')
# Optimized counts
exporter_optimized_partners = active_relationships_df.groupby('exporter').size().reset_index(name='optimized_count')

# Merge the data
exporter_partner_changes = pd.merge(exporter_original_partners, exporter_optimized_partners, 
                                   on='exporter', how='outer').fillna(0)
exporter_partner_changes['change'] = exporter_partner_changes['optimized_count'] - exporter_partner_changes['original_count']
exporter_partner_changes['percent_change'] = 100 * exporter_partner_changes['change'] / exporter_partner_changes['original_count'].replace(0, np.nan)

# Sort by percent change
exporter_partner_changes = exporter_partner_changes.sort_values('percent_change')

# Importer partner changes
# Original counts
importer_original_partners = existing_relationships_df.groupby('importer').size().reset_index(name='original_count')
# Optimized counts
importer_optimized_partners = active_relationships_df.groupby('importer').size().reset_index(name='optimized_count')

# Merge the data
importer_partner_changes = pd.merge(importer_original_partners, importer_optimized_partners, 
                                   on='importer', how='outer').fillna(0)
importer_partner_changes['change'] = importer_partner_changes['optimized_count'] - importer_partner_changes['original_count']
importer_partner_changes['percent_change'] = 100 * importer_partner_changes['change'] / importer_partner_changes['original_count'].replace(0, np.nan)

# Sort by percent change
importer_partner_changes = importer_partner_changes.sort_values('percent_change')

# Plot the top 10 exporters with most significant partner reductions
plt.figure(figsize=(12, 8))
top_reduced_exporters = exporter_partner_changes.sort_values('percent_change').head(10)
bars = plt.barh(top_reduced_exporters['exporter'], top_reduced_exporters['percent_change'], color='#F44336')
plt.title('Top 10 Exporters with Largest Reduction in Trade Partners')
plt.xlabel('Percent Change in Number of Trade Partners')
plt.ylabel('Exporter')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_results/top_reduced_exporters.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot the top 10 exporters with most significant partner increases
plt.figure(figsize=(12, 8))
top_increased_exporters = exporter_partner_changes.sort_values('percent_change', ascending=False).head(10)
bars = plt.barh(top_increased_exporters['exporter'], top_increased_exporters['percent_change'], color='#4CAF50')
plt.title('Top 10 Exporters with Largest Increase in Trade Partners')
plt.xlabel('Percent Change in Number of Trade Partners')
plt.ylabel('Exporter')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_results/top_increased_exporters.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot the top 10 importers with most significant partner reductions
plt.figure(figsize=(12, 8))
top_reduced_importers = importer_partner_changes.sort_values('percent_change').head(10)
bars = plt.barh(top_reduced_importers['importer'], top_reduced_importers['percent_change'], color='#F44336')
plt.title('Top 10 Importers with Largest Reduction in Trade Partners')
plt.xlabel('Percent Change in Number of Trade Partners')
plt.ylabel('Importer')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_results/top_reduced_importers.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot the top 10 importers with most significant partner increases
plt.figure(figsize=(12, 8))
top_increased_importers = importer_partner_changes.sort_values('percent_change', ascending=False).head(10)
bars = plt.barh(top_increased_importers['importer'], top_increased_importers['percent_change'], color='#4CAF50')
plt.title('Top 10 Importers with Largest Increase in Trade Partners')
plt.xlabel('Percent Change in Number of Trade Partners')
plt.ylabel('Importer')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_results/top_increased_importers.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Analyze trade concentration
print("\nAnalyzing trade concentration...")

# Define function to calculate Herfindahl-Hirschman Index (HHI)
def calculate_hhi(df, entity_column, value_column):
    # Calculate total value
    total_value = df[value_column].sum()
    
    # Group by entity and calculate market shares
    market_shares = df.groupby(entity_column)[value_column].sum() / total_value
    
    # Calculate HHI (sum of squared market shares)
    hhi = (market_shares ** 2).sum()
    
    return hhi

# Calculate HHI for original and optimized trade
exporter_original_hhi = calculate_hhi(existing_relationships_df, 'exporter', 'original_fob')
exporter_optimized_hhi = calculate_hhi(active_relationships_df, 'exporter', 'optimized_fob')

importer_original_hhi = calculate_hhi(existing_relationships_df, 'importer', 'original_fob')
importer_optimized_hhi = calculate_hhi(active_relationships_df, 'importer', 'optimized_fob')

print(f"Exporter market concentration (HHI) - Original: {exporter_original_hhi:.4f}, Optimized: {exporter_optimized_hhi:.4f}")
print(f"Change in exporter concentration: {100 * (exporter_optimized_hhi - exporter_original_hhi) / exporter_original_hhi:+.2f}%")

print(f"Importer market concentration (HHI) - Original: {importer_original_hhi:.4f}, Optimized: {importer_optimized_hhi:.4f}")
print(f"Change in importer concentration: {100 * (importer_optimized_hhi - importer_original_hhi) / importer_original_hhi:+.2f}%")

# Create bar chart for HHI comparison
plt.figure(figsize=(10, 6))
x = ['Exporter\nOriginal', 'Exporter\nOptimized', 'Importer\nOriginal', 'Importer\nOptimized']
y = [exporter_original_hhi, exporter_optimized_hhi, importer_original_hhi, importer_optimized_hhi]
colors = ['#2196F3', '#4CAF50', '#2196F3', '#4CAF50']
bars = plt.bar(x, y, color=colors)
plt.title('Market Concentration (HHI) Before and After Optimization')
plt.ylabel('Herfindahl-Hirschman Index')
plt.ylim(0, max(y) * 1.2)  # Add some space for annotations

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(y),
             f'{height:.4f}',
             ha='center', va='bottom', rotation=0, fontsize=11)

# Add percentage change annotations
plt.annotate(f'{100 * (exporter_optimized_hhi - exporter_original_hhi) / exporter_original_hhi:+.2f}%',
             xy=(1, exporter_optimized_hhi), 
             xytext=(1, exporter_optimized_hhi + 0.05*max(y)),
             ha='center', fontsize=11)

plt.annotate(f'{100 * (importer_optimized_hhi - importer_original_hhi) / importer_original_hhi:+.2f}%',
             xy=(3, importer_optimized_hhi), 
             xytext=(3, importer_optimized_hhi + 0.05*max(y)),
             ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('analysis_results/market_concentration_hhi.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Analyze volume changes
print("\nAnalyzing volume changes...")

maintained_relationships_df['volume_percent_change'] = 100 * (maintained_relationships_df['optimized_volume'] - maintained_relationships_df['original_volume']) / maintained_relationships_df['original_volume']

plt.figure(figsize=(12, 6))
sns.histplot(maintained_relationships_df['volume_percent_change'], bins=30, kde=True, color='#673AB7')
plt.title('Distribution of Volume Changes in Maintained Relationships')
plt.xlabel('Percent Change in Volume')
plt.ylabel('Number of Trade Relationships')
plt.axvline(x=0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('analysis_results/volume_change_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Create relationship network visualization (if networkx is available)
try:
    import networkx as nx
    print("\nCreating trade network visualizations...")
    
    # Create a graph of original trade relationships
    G_original = nx.Graph()
    
    # Add edges for original relationships (using FOB as edge weight)
    for _, row in existing_relationships_df.iterrows():
        G_original.add_edge(
            f"E: {row['exporter']}", 
            f"I: {row['importer']}", 
            weight=row['original_fob']
        )
    
    # Create a graph of optimized trade relationships
    G_optimized = nx.Graph()
    
    # Add edges for optimized relationships
    for _, row in active_relationships_df.iterrows():
        # Color edges differently for new vs maintained relationships
        edge_type = 'new' if row['is_new_relationship'] else 'maintained'
        
        G_optimized.add_edge(
            f"E: {row['exporter']}", 
            f"I: {row['importer']}", 
            weight=row['optimized_fob'],
            edge_type=edge_type
        )
    
    # Calculate network metrics
    original_density = nx.density(G_original)
    optimized_density = nx.density(G_optimized)
    
    print(f"Original network density: {original_density:.6f}")
    print(f"Optimized network density: {optimized_density:.6f}")
    print(f"Change in network density: {100 * (optimized_density - original_density) / original_density:+.2f}%")
    
    print(f"Original network average degree: {sum(dict(G_original.degree()).values())/G_original.number_of_nodes():.2f}")
    print(f"Optimized network average degree: {sum(dict(G_optimized.degree()).values())/G_optimized.number_of_nodes():.2f}")
    
    # Function to draw network with improved visualization
    def draw_network(G, title, filename, top_n=50):
        # Sort edges by weight and take top N
        edges = [(u, v) for u, v, d in sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)]
        
        # Take subgraph of top N edges
        if len(edges) > top_n:
            edges = edges[:top_n]
            G_sub = G.edge_subgraph(edges)
        else:
            G_sub = G
        
        plt.figure(figsize=(15, 15))
        
        # Identify exporter and importer nodes
        exporters = [n for n in G_sub.nodes() if n.startswith("E: ")]
        importers = [n for n in G_sub.nodes() if n.startswith("I: ")]
        
        # Position nodes in a bipartite layout
        pos = {}
        for i, node in enumerate(exporters):
            pos[node] = (-1, i * 15 / len(exporters))
        for i, node in enumerate(importers):
            pos[node] = (1, i * 15 / len(importers))
        
        # Draw nodes
        nx.draw_networkx_nodes(G_sub, pos, 
                               nodelist=exporters,
                               node_color='#2196F3', 
                               node_size=100,
                               alpha=0.8)
        
        nx.draw_networkx_nodes(G_sub, pos, 
                               nodelist=importers,
                               node_color='#4CAF50', 
                               node_size=100,
                               alpha=0.8)
        
        # Draw edges with width proportional to weight
        # For optimized graph, color new edges differently
        if 'edge_type' in next(iter(G_sub.edges(data=True)))[2]:
            # Separate edges by type
            maintained_edges = [(u, v) for u, v, d in G_sub.edges(data=True) if d.get('edge_type') == 'maintained']
            new_edges = [(u, v) for u, v, d in G_sub.edges(data=True) if d.get('edge_type') == 'new']
            
            # Get weights for each edge type
            maintained_weights = [G_sub[u][v]['weight'] / 10000000 for u, v in maintained_edges]
            new_weights = [G_sub[u][v]['weight'] / 10000000 for u, v in new_edges]
            
            # Draw maintained edges
            nx.draw_networkx_edges(G_sub, pos, 
                                   edgelist=maintained_edges,
                                   width=maintained_weights,
                                   alpha=0.6,
                                   edge_color='#607D8B')
            
            # Draw new edges
            nx.draw_networkx_edges(G_sub, pos, 
                                   edgelist=new_edges,
                                   width=new_weights,
                                   alpha=0.6,
                                   edge_color='#FF9800',
                                   style='dashed')
            
            # Add a legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#607D8B', lw=2, label='Maintained Relationships'),
                Line2D([0], [0], color='#FF9800', lw=2, linestyle='dashed', label='New Relationships')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
        else:
            # For original graph, just draw all edges the same color
            edge_weights = [G_sub[u][v]['weight'] / 10000000 for u, v in G_sub.edges()]
            
            nx.draw_networkx_edges(G_sub, pos, 
                                  width=edge_weights,
                                  alpha=0.6,
                                  edge_color='#607D8B')
        
        # Draw labels with smaller font
        nx.draw_networkx_labels(G_sub, pos, 
                               font_size=8,
                               font_family='sans-serif')
        
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Draw networks
    draw_network(G_original, 'Original Coffee Trade Network (Top 50 Relationships)', 'analysis_results/original_trade_network.png')
    draw_network(G_optimized, 'Optimized Coffee Trade Network (Top 50 Relationships)', 'analysis_results/optimized_trade_network.png')
    
except ImportError:
    print("Note: networkx library not available for network visualization. Install with 'pip install networkx'.")

print("\nAnalysis complete. Results saved in 'analysis_results/' directory.") 