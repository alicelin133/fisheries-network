'''
min_model_constants.py
Stores the input constants for min-model.py
'''
# n = number of spatial units (territories), 1 fisher in each
n = 3
# delta = factor of leakage (in [0,1])
delta = 0
# q = fish "catchability" (from STL 2016 supp. material)
q = 1
# R_0 = initial biomass of fish per territory (from STL 2016 supp. material)
R_0 = 50.0
# e_0 = initial effort level from one fisher (from STL 2016 supp. material)
e_0 = 0.005
# maxstep = number of time steps that the simulation runs
maxstep = 10000