# Coffee Optimization Project

This repository contains tools for optimizing coffee trade networks and logistics for coffee farming collectives.

## Overview

This project includes several optimization models:

1. **Coffee Trade Optimization (coffee_trade_optimization.py)**: Optimizes trade relationships between exporters and importers with considerations for switching costs.

2. **Coffee Network Analysis (coffee_network_analysis.py)**: Analyzes networks of coffee trade relationships.

3. **VRP Analysis (gurobi_collective_vrp.py, gurobi_vrp_analysis.py)**: Vehicle Routing Problem analysis for coffee farming collectives near Port Santos, comparing different logistics models:
   - Individual pickups from each municipality to Port Santos
   - Hub-based collection where municipalities bring coffee to a central hub
   - Optimal VRP solution finding the most efficient route

## Key Findings

For the analyzed collective (CHAVANTES, MANDURI, BERNARDINO DE CAMPOS, IPAUSSU, SANTA CRUZ DO RIO PARDO, TIMBURI, OLEO), the hub-based model with IPAUSSU as the hub provided 82.77% distance savings compared to individual pickups, while the optimal VRP solution provided 84.34% savings.

## Dependencies

- Python 3.x
- Gurobi Optimizer
- pandas
- networkx (for network analysis)
- matplotlib (for visualization)

## Usage

Datasets can be found here:
https://trase.earth/open-data/datasets/supply-chains-brazil-coffee
https://www.ibge.gov.br/en/geosciences/territorial-organization/territorial-meshes/18890-municipal-mesh.html?edicao=30154&t=downloads

Each Python file can be run independently to perform different types of analyses.
# AaronWenkThesis
