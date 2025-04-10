import geopandas as gpd
import pandas as pd
import json
import os

# Path to the shapefile
shapefile_path = 'br_municipios/BRMUE250GC_SIR.shp'

# Load the shapefile using geopandas
print(f"Loading shapefile from {shapefile_path}...")
try:
    municipalities_gdf = gpd.read_file(shapefile_path)
    print(f"Successfully loaded {len(municipalities_gdf)} municipalities from shapefile")
    
    # Print first few rows to see the structure
    print("\nFirst few rows of the shapefile data:")
    print(municipalities_gdf.head())
    
    # Print column names to identify the relevant columns
    print("\nColumns in the shapefile:")
    print(municipalities_gdf.columns.tolist())
    
    # Extract centroids of each municipality
    print("\nExtracting centroids...")
    
    # Calculate centroids directly from the GeoDataFrame
    municipalities_gdf['latitude'] = municipalities_gdf.geometry.centroid.y
    municipalities_gdf['longitude'] = municipalities_gdf.geometry.centroid.x
    
    # Find ID and name columns in the data
    id_columns = [col for col in municipalities_gdf.columns if 'CD_' in col or 'ID' in col or 'COD' in col]
    name_columns = [col for col in municipalities_gdf.columns if 'NM_' in col or 'NAME' in col or 'NOME' in col]
    
    print(f"Potential ID columns: {id_columns}")
    print(f"Potential name columns: {name_columns}")
    
    # Create a mapping dictionary suitable for our geocoding function
    print("\nCreating municipality mapping dictionary...")
    municipality_mapping = {}
    
    # Determine which column is the IBGE code from the id_columns
    if id_columns:
        # Use the first ID column or the most appropriate one
        id_column = id_columns[0]  # Adjust if needed
        print(f"Using {id_column} as the ID column for mapping")
        
        for _, row in municipalities_gdf.iterrows():
            ibge_code = str(row[id_column])
            # Add 'BR-' prefix to match the trase_id format
            trase_id = f"BR-{ibge_code}"
            municipality_mapping[trase_id] = {
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }
            
            # Also add entries for the codes without the BR- prefix
            municipality_mapping[ibge_code] = {
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }
            
        # Add name-based entries if available
        if name_columns:
            name_column = name_columns[0]  # Adjust if needed
            print(f"Using {name_column} as the name column for additional mapping")
            
            for _, row in municipalities_gdf.iterrows():
                name = str(row[name_column]).strip().upper()
                if name and name not in municipality_mapping:
                    municipality_mapping[name] = {
                        'latitude': row['latitude'],
                        'longitude': row['longitude']
                    }
    
    # Save the mapping to a JSON file
    with open('brazil_municipalities_mapping.json', 'w') as f:
        json.dump(municipality_mapping, f)
    
    print(f"\nSaved mapping for {len(municipality_mapping)} municipality identifiers")
    print("Mapping saved to brazil_municipalities_mapping.json")
    
    # Also save a simplified CSV for easier viewing
    simplified_df = municipalities_gdf[id_columns + name_columns + ['latitude', 'longitude']].copy()
    simplified_df.to_csv('brazil_municipalities.csv', index=False)
    print("Simplified data saved to brazil_municipalities.csv")
    
except Exception as e:
    print(f"Error processing shapefile: {e}")
    print("Make sure you have the required libraries installed: geopandas, pandas, shapely") 