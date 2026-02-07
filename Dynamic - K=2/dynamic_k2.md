## Dynamic Strategy 
* Dual Track Dynamic strategy has been deployed 
* Track A - First Order Markov Chain (discrete) is used to capture random switching nature 
* Track B - PySindy Model (continuous) is used to derive governing differential equations 
* Both methods will be trained using 2024 dataset before simulating and comapring to 2025 true data and each other 
* Most accurate model will be that which matches true behaviour closest
* Non Homonegeinty concerns are mitigated by running two models for Track A and Track B to split the analysis into Weekdays and Weekends


## preprocessing.py 
* Script to ensure 2024 and 2025 data is ready for Dynamic analysis 
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

* Raw data is list of days, must mathematically link Today to Tomorrow
* Sort by feeder then date to ensure accurate tracking 
* "groupby('x').shift(-1)" is used to ensure feeder A analysis stops on last day and doesnt continue into Feeder B data 
* Day gap must be one otherwise transition ignored, protects against outtages 
* Splits weekdays and weekends into separate dataframes 

### Outputs: 
* Weekday and weekend csv storing 2x2 matrix
* Weekday and weekend png saving visual transition matrix 


## markov_testing.py 

### Methodology
* Aim - to validate the Markov Training Model (2024) by testing it against unseen data (2025).
* Monte Carlo Simulation (random sampling) is used to create a Digital Twin 
* 1000 simulated feeders were iterated through for 2025 using the 'rules' learned in training
* Model is valid if the simulation matches the real data

### Function 1 
* Simulaton Engine function 
1. Initialise - creates empty matrix and sets day 0 as start point
2. Weekdays - calendar made to find weekdays
3. Optimise - dataframes to numpy arrays for efficiency 
4. Main Loop -  * matrix selected depending on day 
                * update each virtual feeder
                * loop through cluster for batch processing (virtual feeders grouped by state)
                * for virtual feeders in each cluster 'probability dice' rolled to find next state 
                * create list of simulated states 

### Markov Testing
* Check date is within bounds
* Set up parameters for Simulation 
* Set Simulation start date to match real data 
* Ensure 0 probability doesn't crash model
* Simulation ran, 2025 and Sim statistics gathered
* Drift results printed 'Actual' vs 'Predicted'
* Relative Change calculated using (differnce/sim) for Predator and Prey
* RMSE calculated and printed 
* Comparison plotted 

### Outputs:
* Graph showing simulated vs real data 


## demand.py
* Script to analyse the change in demand from sample population 
* 'Perfect' days (48 vectors) are taken from 2024 and 2025 dataset and counted 
* Total Demand is taken and averaged to daily Demand
* Daily Demand is compared to Relative Predator Change to accompany Markov testing output

# PySindy Model 

## pysindy_training.py

### Methodology
* To split Weekday and Weekend behaviour 'Multiple Trajectories' strategy was chosen, this splits each week of data into 2 segments, one for Weekdays and one for Weekends, totaling 52 segments each 
* Uses Sparse Identification of Nonlinear Dynamics (SINDy) to find the differential equation (dx/dt)
* Outputs Governing Differential Equations 

### Initialise 
* Read 2024 ratio csv 
* Convert to df and sort by date 
* Split Weeekdays and Weekends by assigning them value (0-6)
* Create empty lists for value and derivatives 

### Function 1 
* Take 'today' and 'tomorrow'
* Ensure gap is 1 day, otherwise skip data
* Ensure there is no Weekday > Weekend or Weekend > Weekday

* Derivates taken from smoothed data to eliminate noise 
* PySindy uses Centered Finite Difference (x = x_t+1 - x_t-1 / 2*Δt) which crashes on when analysing weekend (only 2 days to analyse)
* Therefore derivates were manually found using Forward Diffrence method (x_t+1 - x_t / 1) to allow weekend calculation

* Store derivatives in correct section 
* PySindy expects 2D input and pandas outputs 1D, ensure all data is 2D by stating 1 column [N > N,1]

### Define and Train
* Use polynomial library to find second degree logistic curve (1, x, x^2)
* STLSQ is used to remove noise 
* Set up empty models, train Weekday and Weekend to find curve fit, t=1 indicates 1 day gap between rows 

### Save Equations 
* Resulting equations are saved to text file 
* Also printed in terminal 

### Plot 
* Create Weekday and Weekend plot to show results 
* Derivative value is plotted against fitted curve from model 

### Outputs:
* Equation result text file 
* Weekday and Weekend curve comparison plot 

## pysindy_testing.py

### Methodology
* Uses the differential equations found in 2024 training to predict 2025 behaviour
* Anchors the simulation to the real Day 1 value to ensure a fair race between model and reality
* Tests if the physics of human behaviour remained constant or drifted over the year

### Coefficients 
* Function to extract training coeff from user input 
* Removes all spaces from string 
* Uses Regex to search for desired input order 
* Converts string input to digits 

### User Input 
* Coeff function used to take input and extract coeff values 
* Prints extracted values for Weekday and Weekend 

### Sim Start 
* Alligns simulation start date with real start date from dataset 
* Prints startng date and value of predator ratio on that day 

### Simulation
* Define ODE (c+ax+bx^2)
* Match simulation to correct 'real' day 

* Loop to iterate through all days
* Checks date of previous day to assign Weekday or Weekend coefficients
* Take current ratio value, apply coefficients and integrate for 1 day using odeint
* Extract final time step value (t=1 day)
* Contain value within 0-1
* Make new value current value to continue iterating 

### Results 
* Pulls real and simulated data from df and convert to list of numbers 
* Calculate RMSE error for simulation 
* Calculate drift for each dataset (distance from start to end)
* Compare drift to measure model success 

### Plot 
* Plots actual data vs simulation results 

### Outputs: 
* RMSE error for simulation
* Total drift value 
* Comparison plot 











