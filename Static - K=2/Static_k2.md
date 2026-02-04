# Static Code Walkthrough - static_k2.py 

## Methodology Switch 
* unlike other static model, K is preset
* to find high and low users must filter by magnitude not shape of demand (skips normalisation)

## Function 1 
* loads parquet file
* filters data for predetermined ‘training period’ (1/3/24-1/3/25) 
* converts timestamped data (from source) into matrix, original data is one row per reading but for clustering data must be in one row per day format (48 vector) where row = individual days and column = 48 half hourly readings

## Downsampling 
* checks if dataset has more than 3000 days 
* if so will take random sample as whole dataset requires too much processing power and 3000 days is enough to view trends for centroid identification 

## Function 2 

### This is the custom made K Means loop to cluster data: 
1.	Restarts: Outer loop runs the entire code 10 times to avoid ‘luck based’ clustering 
2.	Distance Calculation: Nested FOR loops used to calculate the Canberra Distance between every data point and the two centroids
3.	Aveeraging: Points are assigned to the nearest centroid, and the centroid is moved to the average of points
4.	Convergence: Inner loop stops early if the centroids move less than 0.001 
5.	Error: Finds error of each restart by summing distance from point to centroid, after all restarts are complete the lowest error is printed and held

## Clustering Balance 
* energy of each cluster is summed, this ensures cluster 0 always has lowest usage for each run as if Centroid 0 > Centroid 1 they will swap
* centroid position then saved for dynamic modelling 

## Full dataset 
* once optimal starting centroids have been selected the script is ran through the entire dataset instead of 3000 day sample, captures yearly behaviour

## Results 
* saves parquet file with centroid for each feeder for each day for selected time 
* creates plot to show shape of each cluster
