# EM401 - Using Population Dynamics to Model Large Scale Residential Electrical Demand 
# Code Overview 

## Required Modules: 
* pandas
* geopandas
* matplotlib
* scipy.spatial
* numpy

# Workflow

## 2024 Preprocessing 
'''
sample_2024.py -    initial script to gather sample data from SSEN (200 Virtual Feeders)
                    cleans data and filters for feeders with aggregation 40>N<50
                    OUTPUT: "200_2024.parquet"

sample_checker.py - reads parquet file and conducts health check to look for missing readings and gaps (outages)
                    creates summary and plot of active feeders by day 
                    INPUT: "200_2024.parquet"
                    OUTPUT: "sample_summary.txt" , "feeder_plot.png"

## Static - K=2 
static.md -         md file walkthrough of static_k2.py code 

static_k2.py -      applies custom made K Means clustering with canberra distancing 
                    INPUT: "200_2024.parquet"
                    OUTPUT: "clusters_2024.parquet" , "cluster_plot.png" , "centroids.npy"

'''


