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

### sample_2024.py 
* initial script to gather sample data from SSEN (200 Virtual Feeders)
* cleans data and filters for feeders with aggregation 40>N<50
* OUTPUT: "200_2024.parquet"

### sample_checker_2024.py 
* reads parquet file and conducts health check to look for missing readings and gaps (outages)  
* creates summary and plot of active feeders by day 
* INPUT: "200_2024.parquet"
* OUTPUT: "sample_summary.txt" , "feeder_plot.png"


## Static - Find K 

### static.md 
* md file walkthrough of static.py code 

### static.py
* applies custom made K Means clustering with canberra distancing to find optimal number of clusters prioritising shape over magnitude 
* INPUT: "200_2024.parquet"
* OUTPUT: "centroids.npy" , "cluster_plot.png" , "clusters_2024.parquet" , "elbow_plot.png" 


## Static - K=2 

### static_k2.md 
* md file walkthrough of static_k2.py code 

### static_k2.py    
* applies custom made K Means clustering with canberra distancing 
* INPUT: "200_2024.parquet"
* OUTPUT: "clusters_2024_k2.parquet" , "cluster_plot_k2.png" , "centroids_k2.npy"

## 2025 Preprocessing 

### sample_2025.py 
* takes 2025 data for sample feeders 
* cleans data and saves as parquet file 
* OUTPUT: "200_2025.parquet"

### sample_checker_2025.py 
* reads parquet file and conducts health check to find date range, missing data and complete days
* prints summary in terminal 
* INPUT: "200_2025.parquet"


## Dynamic - K=2 

### preprocessing.py 
* uses static foundation and centroids to iterate through 2025 transitions 
* finds ratio of cluster population for each day
* creates CSVs of ratios using 2024 and 2025 dataset

## Dynamic - K=2 > Markov Chain 
## Dynamic - K=2 > PySindy Model






