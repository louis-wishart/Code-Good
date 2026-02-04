## Dynamic Strategy 
* blah blah 

## preprocessing.py 
* script to ensure 2024 and 2025 data is ready for Dynamic analysis 
* 2025 data cleaned using same method as 2024 data in static_k2.py 
* PySindy Model requires population ratios (cluster 0 and cluster 1)
* 2024 and 2025 cleaned datasets were used to construct CSV of ratios 
* 3 day rolling average used to increase model stability, will be compared to raw data 


# Markov Chain 

## markov_training.py

### Methodology
* Uses the 2024 'Training' data to learn user behaviour when switching between Cluster 0 and Cluster 1
* Creates two matrices to analyse Weekday vs Weekend Behaviours
* Creates transition matrix for each and finds coefficients 

### Function 1 
* Matrix Generation function 
1. Counts: every transition recorded and 2x2 enforced (missing state is 0)
2. Normalise: sum of row is 1, turns counts to probability
3. Coefficients: extracts Alpha (Stability of State 0) and Beta (Transition from 0 to 1) probabilities 
4. Plot: both matrices are plotted to visualise transitions with numbers extracted to CSV

### Markov Training

* raw data is list of days, must mathematically link Today to Tomorrow
* sort by feeder then date to ensure accurate tracking 
* "groupby('x').shift(-1)" is used to ensure feeder A analysis stops on last day and doesnt continue into Feeder B data 
* day gap must be one otherwise transition ignored, protects against outtages 
* splits weekdays and weekends into separate dataframes 

### Outputs: 
* weekday and weekend csv storing 2x2 matrix
* weekday and weekend png saving visual transition matrix 


## markov_testing.py 
