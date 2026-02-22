
FILE_OUT = "coefficients.txt"

# PySINDy 
SINDY_WIN_WE = [-0.09451, 0.47853, -0.50894] 
SINDY_WIN_WD = [0.00886, 0.00496, -0.05857]  
SINDY_SUM_WE = [-0.00573, 0.08363, -0.18024] 
SINDY_SUM_WD = [0.00213, 0.04005, -0.22515]  

# Markov 
MARKOV_WIN_WE_ALPHA, MARKOV_WIN_WE_BETA = 0.096, 0.150 
MARKOV_WIN_WD_ALPHA, MARKOV_WIN_WD_BETA = 0.062, 0.180 
MARKOV_SUM_WE_ALPHA, MARKOV_SUM_WE_BETA = 0.052, 0.210
MARKOV_SUM_WD_ALPHA, MARKOV_SUM_WD_BETA = 0.031, 0.250



report = f"""
CONTINUOUS (PySindy)

                   | {'Alpha (Growth Rate)':<20} | {'Beta (Saturation Limit)':<20}
-----------------------------------------------------------------
{'Winter Weekend':<18} | {SINDY_WIN_WE[1]:<20.5f} | {abs(SINDY_WIN_WE[2]):<20.5f}
{'Winter Weekday':<18} | {SINDY_WIN_WD[1]:<20.5f} | {abs(SINDY_WIN_WD[2]):<20.5f}
{'Summer Weekend':<18} | {SINDY_SUM_WE[1]:<20.5f} | {abs(SINDY_SUM_WE[2]):<20.5f}
{'Summer Weekday':<18} | {SINDY_SUM_WD[1]:<20.5f} | {abs(SINDY_SUM_WD[2]):<20.5f}


DISCRETE (Markov)


                   | {'Alpha (Predation Rate)':<20}   | {'Beta (Recovery Rate)':<20}
-----------------------------------------------------------------
{'Winter Weekend':<18} | {MARKOV_WIN_WE_ALPHA:<20.3f}     | {MARKOV_WIN_WE_BETA:<20.3f}
{'Winter Weekday':<18} | {MARKOV_WIN_WD_ALPHA:<20.3f}     | {MARKOV_WIN_WD_BETA:<20.3f}
{'Summer Weekend':<18} | {MARKOV_SUM_WE_ALPHA:<20.3f}     | {MARKOV_SUM_WE_BETA:<20.3f}
{'Summer Weekday':<18} | {MARKOV_SUM_WD_ALPHA:<20.3f}     | {MARKOV_SUM_WD_BETA:<20.3f}


"""



print(report)

with open(FILE_OUT, "w") as f:
    f.write(report)
