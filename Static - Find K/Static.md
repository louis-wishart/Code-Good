# Static Code Walkthrough - static.search.py

## Methodology Switch
* unlike the fixed K model, K is variable (searches 2 to 15)
* looking for shape over magnitude for better analysis of behaviour (Normalisation)

## Function 1
* loads parquet file
* filters data for predetermined ‘training period’ (1/3/24-1/3/25)
* converts timestamped data (from source) into matrix, original data is one row per reading but for clustering data must be in one row per day format (48 vector) where row = individual days and column = 48 half hourly readings

## Normalisation 
* divides every row by its total sum
* converts 'absolute energy' into "relative shape' (percentage of daily usage per half hour)

## Downsampling
* checks if dataset has more than 3000 days
* if so will take random sample as whole dataset requires too much processing power and 3000 days is enough to view trends for centroid identification

## Function 2

### This is the custom made K Means loop to find Optimal K:

1. K-Loop: Iterates through cluster counts from 2 up to 15
2. Restarts: Runs the code 10 times for each K to avoid ‘luck based’ clustering
3. Distance Calculation: Nested FOR loops used to calculate the Canberra Distance between every data point and the K centroids
4. Convergence: Inner loop stops early if the centroids move less than 0.001
5. Inertia: Records the average error for every K to build the 'Elbow Curve'

## Geometric Elbow
* instead of manual selection, calculates the maximum curve of the error plot
* selects K where adding more clusters stops giving significant gains

## Full dataset
* once optimal K is detected and centroids selected, the script is ran through the entire dataset instead of 3000 day sample

## Results
* saves parquet file with cluster ID for each feeder for each day
* Elbow Plot: saves graph showing error vs number of clusters
* Cluster Plot: saves plot of each cluster shape found 
