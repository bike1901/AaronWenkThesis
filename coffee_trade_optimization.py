import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

# Helper function to calculate Herfindahl-Hirschman Index (HHI)
def calculate_hhi(dataframe, entity_column, value_column):
    # Calculate total value
    total_value = dataframe[value_column].sum()
    
    # Group by entity and calculate market shares
    market_shares = dataframe.groupby(entity_column)[value_column].sum() / total_value
    
    # Calculate HHI (sum of squared market shares)
    hhi = (market_shares ** 2).sum()
    
    return hhi

# Timer start
start_time = time.time()

# Read data
print("Loading dataset...")
df = pd.read_csv('brazil-coffee-v2.5.1-2024-04-26.csv')

# Data preprocessing
print("Preprocessing data...")

# Calculate maximum values in the data for setting BIG_M
max_fob = df['fob'].max()
max_volume = df['volume'].max()
total_fob = df['fob'].sum()
total_volume = df['volume'].sum()

print(f"Data statistics for setting constraints:")
print(f"Maximum FOB value: ${max_fob:,.2f}")
print(f"Maximum volume: {max_volume:,.2f}")
print(f"Total FOB: ${total_fob:,.2f}")
print(f"Total volume: {total_volume:,.2f}")

# Set BIG_M to be safely larger than any possible trade value
# Use 10 times the maximum of: (maximum individual trade, total trade)
BIG_M = 10 * max(max_fob, max_volume, total_fob, total_volume)
print(f"Setting BIG_M to: {BIG_M:,.2f}")

# Filter for a specific year if needed (e.g., most recent or complete year)
latest_year = df['year'].max()
df = df[df['year'] == latest_year]
print(f"Filtered for year: {latest_year}")

# Create unique IDs for each exporter and importer
exporters = df['exporter'].unique()
importers = df['importer'].unique()

# Get existing importer-destination country relationships
importer_countries = df.groupby('importer')['country_of_destination'].unique().to_dict()

# Store the original destination countries for each importer for later constraint checking
importer_original_destinations = {}
for importer, countries in importer_countries.items():
    importer_original_destinations[importer] = set(countries)

print("Stored original destination countries for each importer")

# Create a dictionary to track which exporters already trade with which importers
existing_relationships = df.groupby(['exporter', 'importer']).size().reset_index().rename(columns={0: 'count'})
existing_pairs = set(zip(existing_relationships['exporter'], existing_relationships['importer']))

# Create a dictionary to store the FOB values and volumes for each exporter-importer pair
trade_data = df.groupby(['exporter', 'importer']).agg({
    'fob': 'sum',
    'volume': 'sum'
}).reset_index()

# Convert to a dictionary for easy lookup
fob_values = {(row['exporter'], row['importer']): row['fob'] for _, row in trade_data.iterrows()}
volumes = {(row['exporter'], row['importer']): row['volume'] for _, row in trade_data.iterrows()}

# Create a dictionary to store total export volumes per exporter
exporter_volumes = df.groupby('exporter')['volume'].sum().to_dict()
importer_volumes = df.groupby('importer')['volume'].sum().to_dict()

# Calculate average FOB per unit for each exporter
exporter_avg_fob_per_unit = {}
for exporter in exporters:
    exporter_data = trade_data[trade_data['exporter'] == exporter]
    if not exporter_data.empty and exporter_data['volume'].sum() > 0:
        exporter_avg_fob_per_unit[exporter] = exporter_data['fob'].sum() / exporter_data['volume'].sum()
    else:
        exporter_avg_fob_per_unit[exporter] = 0

# Calculate average FOB per unit for each importer
importer_avg_fob_per_unit = {}
for importer in importers:
    importer_data = trade_data[trade_data['importer'] == importer]
    if not importer_data.empty and importer_data['volume'].sum() > 0:
        importer_avg_fob_per_unit[importer] = importer_data['fob'].sum() / importer_data['volume'].sum()
    else:
        importer_avg_fob_per_unit[importer] = 0

print(f"Number of exporters: {len(exporters)}")
print(f"Number of importers: {len(importers)}")
print(f"Number of existing trade relationships: {len(existing_pairs)}")

# Create optimization model with numeric focus
print("Building optimization model...")
model = gp.Model("CoffeeTradeOptimization")
model.setParam('NumericFocus', 3)  # Highest precision to avoid numerical issues

# Define switching cost parameters
# These can be adjusted based on industry knowledge or sensitivity analysis
COST_PER_NEW_RELATIONSHIP = 50000  # Fixed cost ($) for establishing a new trade relationship
COST_PER_DROPPED_RELATIONSHIP = 75000  # Fixed cost ($) for dropping an existing relationship
print(f"Switching costs: ${COST_PER_NEW_RELATIONSHIP} per new relationship, ${COST_PER_DROPPED_RELATIONSHIP} per dropped relationship")

# Decision variables: x[e,i] = 1 if exporter e trades with importer i, 0 otherwise
x = {}
allowed_pairs = []

# Allow new pairings within certain constraints
for exporter in exporters:
    # Get countries this exporter has exported to
    exporter_countries = set()
    for importer in df[df['exporter'] == exporter]['importer'].unique():
        for country in importer_countries.get(importer, []):
            exporter_countries.add(country)
    
    for importer in importers:
        # Allow trade if:
        # 1) It's an existing relationship OR
        # 2) The importer imports to countries this exporter has already exported to
        importer_dest_countries = set(importer_countries.get(importer, []))
        if (exporter, importer) in existing_pairs or len(exporter_countries.intersection(importer_dest_countries)) > 0:
            x[exporter, importer] = model.addVar(vtype=GRB.BINARY, name=f"x_{exporter}_{importer}")
            allowed_pairs.append((exporter, importer))

print(f"Number of allowed trade relationships in model: {len(allowed_pairs)}")

# Limit the number of allowed pairs if there are too many (for computational feasibility)
if len(allowed_pairs) > 20000:
    print(f"Too many potential relationships ({len(allowed_pairs)}). Limiting to existing plus top potential pairs...")
    
    # Keep all existing pairs
    filtered_pairs = [(e, i) for e, i in allowed_pairs if (e, i) in existing_pairs]
    
    # For each exporter, find top 5 potential new importers
    for exporter in exporters:
        potential_importers = [(e, i) for e, i in allowed_pairs if e == exporter and (e, i) not in existing_pairs]
        # Use importer's volume as a proxy for potential value
        potential_importers.sort(key=lambda pair: importer_volumes.get(pair[1], 0), reverse=True)
        filtered_pairs.extend(potential_importers[:5])  # Add top 5 potential importers
    
    # For each importer, find top 5 potential new exporters
    for importer in importers:
        potential_exporters = [(e, i) for e, i in allowed_pairs if i == importer and (e, i) not in existing_pairs]
        # Use exporter's volume as a proxy for potential value
        potential_exporters.sort(key=lambda pair: exporter_volumes.get(pair[0], 0), reverse=True)
        filtered_pairs.extend(potential_exporters[:5])  # Add top 5 potential exporters
    
    # Remove duplicates and update allowed_pairs
    allowed_pairs = list(set(filtered_pairs))
    print(f"Reduced to {len(allowed_pairs)} relationships")

# Add continuous variables for FOB values
fob_vars = {}
for exporter, importer in allowed_pairs:
    # For existing relationships, use the original FOB as a baseline
    if (exporter, importer) in existing_pairs:
        original_fob = fob_values.get((exporter, importer), 0)
        min_fob = max(0.90 * original_fob, 0)  # Allow larger downward adjustment
        max_fob = 1.10 * original_fob  # Allow larger upward adjustment
    else:
        # For new relationships, estimate a reasonable FOB based on exporter and importer averages
        exporter_avg = exporter_avg_fob_per_unit.get(exporter, 0)
        importer_avg = importer_avg_fob_per_unit.get(importer, 0)
        
        # Use a weighted average of exporter and importer FOB per unit rates
        if exporter_avg > 0 and importer_avg > 0:
            estimated_fob_per_unit = (exporter_avg + importer_avg) / 2
        elif exporter_avg > 0:
            estimated_fob_per_unit = exporter_avg
        elif importer_avg > 0:
            estimated_fob_per_unit = importer_avg
        else:
            # Fallback to overall average
            all_trades_avg = df['fob'].sum() / df['volume'].sum() if df['volume'].sum() > 0 else 1000
            estimated_fob_per_unit = all_trades_avg
        
        # Estimate a reasonable volume for this new pair
        estimated_volume = min(
            exporter_volumes.get(exporter, 0) * 0.1,  # 10% of exporter's total volume
            importer_volumes.get(importer, 0) * 0.1   # 10% of importer's total volume
        )
        
        if estimated_volume == 0:
            # If either exporter or importer has 0 volume, use a small default
            estimated_volume = 1
            
        estimated_fob = estimated_fob_per_unit * estimated_volume
        
        min_fob = max(0.90 * estimated_fob, 0)
        max_fob = 1.10 * estimated_fob
    
    fob_vars[exporter, importer] = model.addVar(
        lb=0,  # Always start at 0
        ub=max_fob, 
        name=f"fob_{exporter}_{importer}"
    )

# Add a variable for volume traded between each exporter-importer pair
volume_vars = {}
for exporter, importer in allowed_pairs:
    if (exporter, importer) in existing_pairs:
        original_volume = volumes.get((exporter, importer), 0)
        max_volume = 1.5 * original_volume  # Allow even more volume increase
    else:
        # For new relationships, estimate a reasonable volume
        estimated_volume = min(
            exporter_volumes.get(exporter, 0) * 0.2,  # 20% of exporter's total volume
            importer_volumes.get(importer, 0) * 0.2   # 20% of importer's total volume
        )
        
        if estimated_volume == 0:
            # If either exporter or importer has 0 volume, use a small default
            estimated_volume = 1
            
        max_volume = estimated_volume * 1.5
    
    volume_vars[exporter, importer] = model.addVar(
        lb=0,  # Always start at 0
        ub=max_volume, 
        name=f"volume_{exporter}_{importer}"
    )
    
    # Relationship between FOB and volume using indicator constraints instead of big-M
    if (exporter, importer) in existing_pairs and volumes.get((exporter, importer), 0) > 0:
        # Get the original FOB per unit
        original_fob_per_volume = fob_values.get((exporter, importer), 0) / volumes.get((exporter, importer), 1)
        
        # Allow FOB per unit volume to vary within ±10% of the original value
        min_fob_per_volume = 0.90 * original_fob_per_volume
        max_fob_per_volume = 1.10 * original_fob_per_volume
    else:
        # For new relationships, use the average FOB per unit for the exporter and importer
        exporter_avg = exporter_avg_fob_per_unit.get(exporter, 0)
        importer_avg = importer_avg_fob_per_unit.get(importer, 0)
        
        if exporter_avg > 0 and importer_avg > 0:
            avg_fob_per_volume = (exporter_avg + importer_avg) / 2
        elif exporter_avg > 0:
            avg_fob_per_volume = exporter_avg
        elif importer_avg > 0:
            avg_fob_per_volume = importer_avg
        else:
            # Fallback to overall average
            all_trades_avg = df['fob'].sum() / df['volume'].sum() if df['volume'].sum() > 0 else 1000
            avg_fob_per_volume = all_trades_avg
        
        # Tighten FOB per unit constraints to be more realistic
        min_fob_per_volume = 0.90 * avg_fob_per_volume  
        max_fob_per_volume = 1.10 * avg_fob_per_volume  
    
    # If the pair is inactive (x=0), then both fob and volume must be 0
    model.addConstr(fob_vars[exporter, importer] <= BIG_M * x[exporter, importer], 
                   f"fob_inactive_{exporter}_{importer}")
    model.addConstr(volume_vars[exporter, importer] <= BIG_M * x[exporter, importer], 
                   f"volume_inactive_{exporter}_{importer}")
    
    # If the pair is active (x=1), then relate FOB to volume
    # Only add FOB per unit constraints for active relationships and if the min_fob_per_volume is significant
    if min_fob_per_volume > 1.0:  # Only enforce for reasonable values
        model.addGenConstrIndicator(
            x[exporter, importer], 
            True, 
            fob_vars[exporter, importer] >= min_fob_per_volume * volume_vars[exporter, importer],
            name=f"min_fob_per_volume_{exporter}_{importer}"
        )
    
    model.addGenConstrIndicator(
        x[exporter, importer], 
        True, 
        fob_vars[exporter, importer] <= max_fob_per_volume * volume_vars[exporter, importer],
        name=f"max_fob_per_volume_{exporter}_{importer}"
    )

# Add variables to track new relationships and dropped relationships
new_relationship_costs = {}
dropped_relationship_costs = {}

# Create variables for new relationship costs
for exporter, importer in allowed_pairs:
    if (exporter, importer) not in existing_pairs:
        # This is a potential new relationship - will incur cost if activated
        new_relationship_costs[exporter, importer] = model.addVar(
            obj=0.0,  # No direct objective coefficient - we'll handle in the objective function
            vtype=GRB.CONTINUOUS,  # Can be binary, but continuous works with our indicator constraint
            name=f"new_rel_cost_{exporter}_{importer}"
        )
        
        # If x[e,i] = 1 (new relationship is active), then the cost is COST_PER_NEW_RELATIONSHIP
        # If x[e,i] = 0 (new relationship is not active), then the cost is 0
        model.addGenConstrIndicator(
            x[exporter, importer],
            True,
            new_relationship_costs[exporter, importer] == COST_PER_NEW_RELATIONSHIP,
            name=f"new_rel_cost_active_{exporter}_{importer}"
        )
        model.addGenConstrIndicator(
            x[exporter, importer],
            False,
            new_relationship_costs[exporter, importer] == 0,
            name=f"new_rel_cost_inactive_{exporter}_{importer}"
        )

# Create variables for dropped relationship costs
for exporter, importer in existing_pairs:
    if (exporter, importer) in allowed_pairs:  # Make sure the pair is in allowed_pairs
        # This is an existing relationship - will incur cost if dropped
        dropped_relationship_costs[exporter, importer] = model.addVar(
            obj=0.0,  # No direct objective coefficient
            vtype=GRB.CONTINUOUS,  # Can be binary, but continuous works with our indicator constraint
            name=f"dropped_rel_cost_{exporter}_{importer}"
        )
        
        # If x[e,i] = 0 (existing relationship is dropped), then the cost is COST_PER_DROPPED_RELATIONSHIP
        # If x[e,i] = 1 (existing relationship is maintained), then the cost is 0
        model.addGenConstrIndicator(
            x[exporter, importer],
            False,
            dropped_relationship_costs[exporter, importer] == COST_PER_DROPPED_RELATIONSHIP,
            name=f"dropped_rel_cost_active_{exporter}_{importer}"
        )
        model.addGenConstrIndicator(
            x[exporter, importer],
            True,
            dropped_relationship_costs[exporter, importer] == 0,
            name=f"dropped_rel_cost_inactive_{exporter}_{importer}"
        )

# Update model to include new variables and constraints
model.update()

# Calculate total switching costs
total_new_rel_cost = gp.quicksum(new_relationship_costs.values())
total_dropped_rel_cost = gp.quicksum(dropped_relationship_costs.values())
total_switching_cost = total_new_rel_cost + total_dropped_rel_cost

print(f"Added {len(new_relationship_costs)} new relationship cost variables")
print(f"Added {len(dropped_relationship_costs)} dropped relationship cost variables")

# Modify the objective function to account for switching costs
# Original objective
total_fob_objective = gp.quicksum(fob_vars[exporter, importer] for exporter, importer in allowed_pairs)

# New objective: Maximize FOB minus switching costs
model.setObjective(
    total_fob_objective - total_switching_cost,
    GRB.MAXIMIZE
)

print("Objective function updated to account for switching costs")

# Constraints

# Calculate total original volume
total_original_volume = sum(volumes.values())
print(f"Total original volume: {total_original_volume:.2f}")

# Calculate total original FOB
total_original_fob = sum(fob_values.values())
print(f"Total original FOB: ${total_original_fob:,.2f}")

# 1. Total volume constraint (within ±5% of original total volume)
model.addConstr(
    gp.quicksum(volume_vars[exporter, importer] for exporter, importer in allowed_pairs) >= 
    0.95 * total_original_volume,  # Tightened from 0.90 to 0.95
    "min_total_volume"
)
model.addConstr(
    gp.quicksum(volume_vars[exporter, importer] for exporter, importer in allowed_pairs) <= 
    1.05 * total_original_volume,  # Tightened from 1.10 to 1.05
    "max_total_volume"
)

# 2. Add constraint to ensure importers only import from countries they were importing from before

# Create variables to track where each importer is importing to after optimization
importer_destination_vars = {}
for importer in importers:
    # Get the original destinations this importer was shipping to
    original_destinations = importer_original_destinations.get(importer, set())
    
    # For each destination country
    for destination in original_destinations:
        # Get all pairs (exporter, this importer) where this importer ships to this destination
        dest_pairs = [(e, i) for e, i in allowed_pairs if i == importer]
        
        # Create a variable that equals 1 if this importer imports to this destination after optimization
        if dest_pairs:
            importer_destination_vars[(importer, destination)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"imports_to_{importer}_{destination}"
            )
            
            # Constraint: This variable = 1 if any exporter ships to this importer for this destination
            model.addConstr(
                importer_destination_vars[(importer, destination)] <= 
                gp.quicksum(x[e, i] for e, i in dest_pairs),
                f"imports_to_{importer}_{destination}_active"
            )
    
    # Constraint: Importer must import from all original destinations
    # (ensures they don't stop importing to any country they were importing to before)
    if importer in importer_original_destinations and len(importer_original_destinations[importer]) > 0:
        importer_dest_vars = [importer_destination_vars.get((importer, dest), 0) 
                             for dest in importer_original_destinations[importer]]
        
        if importer_dest_vars:
            model.addConstr(
                gp.quicksum(importer_dest_vars) == len(importer_original_destinations[importer]),
                f"maintain_destinations_{importer}"
            )

print("Added constraint to ensure importers only import to their original destination countries")

# 3. Add volume constraints for largest exporters and importers to prevent unrealistic shifts
# Identify top exporters by volume
top_exporter_volume_threshold = sorted(exporter_volumes.values(), reverse=True)[min(10, len(exporter_volumes)-1)]
top_exporters = [exporter for exporter, volume in exporter_volumes.items() if volume >= top_exporter_volume_threshold]

# Identify top importers by volume
top_importer_volume_threshold = sorted(importer_volumes.values(), reverse=True)[min(10, len(importer_volumes)-1)]
top_importers = [importer for importer, volume in importer_volumes.items() if volume >= top_importer_volume_threshold]

print(f"Adding volume constraints for {len(top_exporters)} largest exporters")
for exporter in top_exporters:
    exporter_pairs = [(e, i) for e, i in allowed_pairs if e == exporter]
    if exporter_pairs:
        model.addConstr(
            gp.quicksum(volume_vars[exporter, importer] for exporter, importer in exporter_pairs) >= 
            0.80 * exporter_volumes.get(exporter, 0),
            f"min_volume_top_exporter_{exporter}"
        )
        model.addConstr(
            gp.quicksum(volume_vars[exporter, importer] for exporter, importer in exporter_pairs) <= 
            1.20 * exporter_volumes.get(exporter, 0),
            f"max_volume_top_exporter_{exporter}"
        )

print(f"Adding volume constraints for {len(top_importers)} largest importers")
for importer in top_importers:
    importer_pairs = [(e, i) for e, i in allowed_pairs if i == importer]
    if importer_pairs:
        model.addConstr(
            gp.quicksum(volume_vars[exporter, importer] for exporter, importer in importer_pairs) >= 
            0.80 * importer_volumes.get(importer, 0),
            f"min_volume_top_importer_{importer}"
        )
        model.addConstr(
            gp.quicksum(volume_vars[exporter, importer] for exporter, importer in importer_pairs) <= 
            1.20 * importer_volumes.get(importer, 0),
            f"max_volume_top_importer_{importer}"
        )

# 4. Limit the number of new relationships that can be created
max_new_relationships = int(0.20 * len(existing_pairs))  # Max 20% new relationships
print(f"Limiting new relationships to: {max_new_relationships}")

new_relationship_vars = {}
for exporter, importer in allowed_pairs:
    if (exporter, importer) not in existing_pairs:
        # This is a potential new relationship
        new_relationship_vars[exporter, importer] = x[exporter, importer]

if new_relationship_vars:
    model.addConstr(
        gp.quicksum(new_relationship_vars.values()) <= max_new_relationships,
        "max_new_relationships"
    )

# 5. Ensure at least 70% of existing relationships are maintained
min_existing_relationships = int(0.70 * len(existing_pairs))
print(f"Requiring at least {min_existing_relationships} existing relationships to be maintained")

existing_relationship_vars = {}
for exporter, importer in allowed_pairs:
    if (exporter, importer) in existing_pairs:
        # This is an existing relationship
        existing_relationship_vars[exporter, importer] = x[exporter, importer]

if existing_relationship_vars:
    model.addConstr(
        gp.quicksum(existing_relationship_vars.values()) >= min_existing_relationships,
        "min_existing_relationships"
    )

# 6. Limit the number of trade partners per exporter to prevent excessive fragmentation
for exporter in exporters:
    current_partner_count = len([pair for pair in existing_pairs if pair[0] == exporter])
    exporter_pairs = [(e, i) for e, i in allowed_pairs if e == exporter]
    
    if len(exporter_pairs) > 0:  # Only add constraint if this exporter has potential pairs
        # Allow at most 30% more partners than current (reduced from 50%)
        max_partners = int(min(current_partner_count * 1.3, current_partner_count + 3))
        
        # Only add the constraint if it would actually limit something
        if len(exporter_pairs) > max_partners:
            model.addConstr(
                gp.quicksum(x[e, i] for e, i in exporter_pairs) <= max_partners,
                f"max_partners_{exporter}"
            )

# 7. Limit the number of trade partners per importer to prevent excessive fragmentation
for importer in importers:
    current_partner_count = len([pair for pair in existing_pairs if pair[1] == importer])
    importer_pairs = [(e, i) for e, i in allowed_pairs if i == importer]
    
    if len(importer_pairs) > 0:  # Only add constraint if this importer has potential pairs
        # Allow at most 30% more partners than current (reduced from 50%)
        max_partners = int(min(current_partner_count * 1.3, current_partner_count + 3))
        
        # Only add the constraint if it would actually limit something
        if len(importer_pairs) > max_partners:
            model.addConstr(
                gp.quicksum(x[e, i] for e, i in importer_pairs) <= max_partners,
                f"max_partners_{importer}"
            )

# Solve the model
print("Solving optimization model...")
model.params.TimeLimit = 900  # Set time limit to 15 minutes
model.params.FeasibilityTol = 1e-6  # Slightly relax the feasibility tolerance
model.params.MIPGap = 0.01  # Accept solutions within 1% of optimal
model.optimize()

# Process results
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SUBOPTIMAL:
    print(f"Optimization completed with status: {model.status}")
    
    # Create a dictionary to store optimized results
    optimized_data = []
    
    # Calculate actual switching costs from solution
    actual_new_rel_cost = 0
    actual_dropped_rel_cost = 0
    
    for exporter, importer in allowed_pairs:
        # Get the decision variable value (0 or 1)
        relationship_active = x[exporter, importer].X > 0.5
        
        # Get the original and optimized values
        original_fob = fob_values.get((exporter, importer), 0)
        original_volume = volumes.get((exporter, importer), 0)
        is_existing_relationship = (exporter, importer) in existing_pairs
        
        optimized_fob = fob_vars[exporter, importer].X if relationship_active else 0
        optimized_volume = volume_vars[exporter, importer].X if relationship_active else 0
        
        # Calculate switching costs
        switching_cost = 0
        if relationship_active and not is_existing_relationship:
            # New relationship activated
            switching_cost = COST_PER_NEW_RELATIONSHIP
            actual_new_rel_cost += COST_PER_NEW_RELATIONSHIP
        elif not relationship_active and is_existing_relationship:
            # Existing relationship dropped
            switching_cost = COST_PER_DROPPED_RELATIONSHIP
            actual_dropped_rel_cost += COST_PER_DROPPED_RELATIONSHIP
        
        # Store the result data
        optimized_data.append({
            'exporter': exporter,
            'importer': importer,
            'original_fob': original_fob,
            'optimized_fob': optimized_fob,
            'original_volume': original_volume,
            'optimized_volume': optimized_volume,
            'difference': optimized_fob - original_fob,
            'percent_change': 100 * (optimized_fob - original_fob) / original_fob if original_fob > 0 else None,
            'relationship_active': relationship_active,
            'is_existing_relationship': is_existing_relationship,
            'is_new_relationship': relationship_active and not is_existing_relationship,
            'is_dropped_relationship': not relationship_active and is_existing_relationship,
            'switching_cost': switching_cost
        })
    
    # Convert to DataFrame and save results
    results_df = pd.DataFrame(optimized_data)
    results_df.to_csv('optimized_coffee_trade.csv', index=False)
    
    # Summary statistics
    original_active_relationships = len(existing_pairs)
    original_fob = sum(row['original_fob'] for row in optimized_data if row['is_existing_relationship'])
    
    optimized_active_relationships = results_df['relationship_active'].sum()
    optimized_fob = results_df[results_df['relationship_active']]['optimized_fob'].sum()
    
    existing_relationships_maintained = results_df[(results_df['is_existing_relationship']) & 
                                                  (results_df['relationship_active'])].shape[0]
    new_relationships_created = results_df[(results_df['is_new_relationship'])].shape[0]
    existing_relationships_dropped = original_active_relationships - existing_relationships_maintained
    
    total_switching_costs = results_df['switching_cost'].sum()
    
    print(f"Total original FOB: ${original_fob:,.2f}")
    print(f"Total optimized FOB: ${optimized_fob:,.2f}")
    print(f"Total switching costs: ${total_switching_costs:,.2f}")
    print(f"  - New relationship costs: ${actual_new_rel_cost:,.2f} ({new_relationships_created} new relationships)")
    print(f"  - Dropped relationship costs: ${actual_dropped_rel_cost:,.2f} ({existing_relationships_dropped} dropped relationships)")
    print(f"Net benefit: ${optimized_fob - total_switching_costs - original_fob:,.2f}")
    print(f"Improvement: ${optimized_fob - original_fob:,.2f} ({100 * (optimized_fob - original_fob) / original_fob:.2f}%)")
    print(f"Improvement after switching costs: ${optimized_fob - total_switching_costs - original_fob:,.2f} ({100 * (optimized_fob - total_switching_costs - original_fob) / original_fob:.2f}%)")
    
    # Additional analysis
    print("\nTop 10 optimized trade relationships by FOB value:")
    top_trades = results_df[results_df['relationship_active']].sort_values('optimized_fob', ascending=False).head(10)
    for _, row in top_trades.iterrows():
        status = "NEW" if row['is_new_relationship'] else "Existing"
        print(f"{row['exporter']} → {row['importer']} ({status}): ${row['optimized_fob']:,.2f}")
    
    print("\nTop 10 new trade relationships by FOB value:")
    top_new_trades = results_df[results_df['is_new_relationship']].sort_values('optimized_fob', ascending=False).head(10)
    for _, row in top_new_trades.iterrows():
        print(f"{row['exporter']} → {row['importer']}: ${row['optimized_fob']:,.2f}")
    
    # -------- RELATIONSHIP ANALYSIS --------
    print("\n----- RELATIONSHIP ANALYSIS -----")
    
    # Analyze relationship stability
    # 1. Create dictionaries to count relationships per exporter/importer
    exporter_original_partners = {}
    exporter_optimized_partners = {}
    
    importer_original_partners = {}
    importer_optimized_partners = {}
    
    # Count original partners
    for exporter, importer in existing_pairs:
        exporter_original_partners[exporter] = exporter_original_partners.get(exporter, 0) + 1
        importer_original_partners[importer] = importer_original_partners.get(importer, 0) + 1
    
    # Count optimized partners
    for _, row in results_df[results_df['relationship_active']].iterrows():
        exporter = row['exporter']
        importer = row['importer']
        exporter_optimized_partners[exporter] = exporter_optimized_partners.get(exporter, 0) + 1
        importer_optimized_partners[importer] = importer_optimized_partners.get(importer, 0) + 1
    
    # Calculate average change in partners
    exporter_changes = []
    for exporter in exporters:
        original = exporter_original_partners.get(exporter, 0)
        optimized = exporter_optimized_partners.get(exporter, 0)
        if original > 0:
            percent_change = 100 * (optimized - original) / original
            exporter_changes.append(percent_change)
    
    importer_changes = []
    for importer in importers:
        original = importer_original_partners.get(importer, 0)
        optimized = importer_optimized_partners.get(importer, 0)
        if original > 0:
            percent_change = 100 * (optimized - original) / original
            importer_changes.append(percent_change)
    
    if exporter_changes:
        print(f"\nExporter relationships:")
        print(f"  Average change in number of trade partners per exporter: {np.mean(exporter_changes):.2f}%")
        print(f"  Max reduction in partners: {min(exporter_changes):.2f}%")
        print(f"  Max increase in partners: {max(exporter_changes):.2f}%")
    
    if importer_changes:
        print(f"\nImporter relationships:")
        print(f"  Average change in number of trade partners per importer: {np.mean(importer_changes):.2f}%")
        print(f"  Max reduction in partners: {min(importer_changes):.2f}%")
        print(f"  Max increase in partners: {max(importer_changes):.2f}%")
    
    # Top 5 exporters/importers with most significant changes in relationships
    if exporter_changes:
        exporter_change_df = pd.DataFrame({
            'exporter': list(exporter_original_partners.keys()),
            'original_partners': [exporter_original_partners.get(e, 0) for e in exporter_original_partners.keys()],
            'optimized_partners': [exporter_optimized_partners.get(e, 0) for e in exporter_original_partners.keys()]
        })
        exporter_change_df['change'] = exporter_change_df['optimized_partners'] - exporter_change_df['original_partners']
        exporter_change_df['percent_change'] = 100 * exporter_change_df['change'] / exporter_change_df['original_partners']
        
        print("\nTop 5 exporters with largest reduction in trade partners:")
        for _, row in exporter_change_df.sort_values('percent_change').head(5).iterrows():
            print(f"  {row['exporter']}: {row['original_partners']} → {row['optimized_partners']} partners ({row['percent_change']:.2f}%)")
        
        print("\nTop 5 exporters with largest increase in trade partners:")
        for _, row in exporter_change_df.sort_values('percent_change', ascending=False).head(5).iterrows():
            print(f"  {row['exporter']}: {row['original_partners']} → {row['optimized_partners']} partners (+{row['percent_change']:.2f}%)")
    
    if importer_changes:
        importer_change_df = pd.DataFrame({
            'importer': list(importer_original_partners.keys()),
            'original_partners': [importer_original_partners.get(i, 0) for i in importer_original_partners.keys()],
            'optimized_partners': [importer_optimized_partners.get(i, 0) for i in importer_original_partners.keys()]
        })
        importer_change_df['change'] = importer_change_df['optimized_partners'] - importer_change_df['original_partners']
        importer_change_df['percent_change'] = 100 * importer_change_df['change'] / importer_change_df['original_partners']
        
        print("\nTop 5 importers with largest reduction in trade partners:")
        for _, row in importer_change_df.sort_values('percent_change').head(5).iterrows():
            print(f"  {row['importer']}: {row['original_partners']} → {row['optimized_partners']} partners ({row['percent_change']:.2f}%)")
        
        print("\nTop 5 importers with largest increase in trade partners:")
        for _, row in importer_change_df.sort_values('percent_change', ascending=False).head(5).iterrows():
            print(f"  {row['importer']}: {row['original_partners']} → {row['optimized_partners']} partners (+{row['percent_change']:.2f}%)")
    
    # Create a concentration analysis
    print("\nTrade Concentration Analysis:")
    
    # Create a dataframe for original trade with volumes
    original_df = results_df[results_df['is_existing_relationship']].copy()
    
    # Create a dataframe with optimized values
    optimized_df = results_df[results_df['relationship_active']].copy()
    
    # Calculate HHI for exporters before and after optimization
    exporter_original_hhi = calculate_hhi(original_df, 'exporter', 'original_fob')
    exporter_optimized_hhi = calculate_hhi(optimized_df, 'exporter', 'optimized_fob')
    
    print(f"Exporter market concentration (HHI) - Original: {exporter_original_hhi:.4f}, Optimized: {exporter_optimized_hhi:.4f}")
    print(f"Change in exporter concentration: {100 * (exporter_optimized_hhi - exporter_original_hhi) / exporter_original_hhi:+.2f}%")
    
    # Calculate HHI for importers
    importer_original_hhi = calculate_hhi(original_df, 'importer', 'original_fob')
    importer_optimized_hhi = calculate_hhi(optimized_df, 'importer', 'optimized_fob')
    
    print(f"Importer market concentration (HHI) - Original: {importer_original_hhi:.4f}, Optimized: {importer_optimized_hhi:.4f}")
    print(f"Change in importer concentration: {100 * (importer_optimized_hhi - importer_original_hhi) / importer_original_hhi:+.2f}%")
    
else:
    print(f"Optimization failed with status: {model.status}")

# Timer end
end_time = time.time()
print(f"\nOptimization completed in {end_time - start_time:.2f} seconds") 