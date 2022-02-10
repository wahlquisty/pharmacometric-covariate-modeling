
using Pkg
# Pkg.instantiate()
Pkg.activate("..")

using DelimitedFiles, Plots, Printf

include("eqreader.jl") # Reads symbolic expression from network.
include("fcts_covmod.jl") # Functions for training, pruning, plotting (and more)
include("masked_dense.jl") # masked dense layer to prune network

# Note that the variable "model" is global and used within most functions.

# Read covariate (input) data
cov = readdlm("csv/patientdata.csv", ',', skipstart = 1) # covariate data, age and weight
AGE = cov[:, 1] # Patient age data
WGT = cov[:, 2] # Patient weight data
max_AGE = maximum(AGE)
max_WGT = maximum(WGT)

# Training data
x_train = [AGE ./ max_AGE WGT ./ max_WGT]' # rescale to values between 0->1, training x data

fV1(x_1, x_2) = 9.294748124171159 * (x_2 / (x_2 + 33.5531248778544)) # True expression for V1
max_V1 = maximum(fV1.(AGE, WGT)) # largest value of V1
V1 = fV1.(AGE, WGT) # central volumes
y_train_V1 = (V1 ./ max_V1)' # scaling output data between 0->1
data_train_V1 = [(x_train, y_train_V1)] # Training data

fV2(x_1, x_2) = 0.36430449290982714 * x_2 * exp(-0.015633 * x_1 + 0.5471550) # True expression for V2
max_V2 = maximum(fV2.(AGE, WGT)) # largest value of V2
V2 = fV2.(AGE, WGT) # peripheral volumes
y_train_V2 = (V2 ./ max_V2)'  # scaling output data between 0->1
data_train_V2 = [(x_train, y_train_V2)] # Training data

# Declaration of hyper-parameters for training (these are all global?)
learning_rate = 0.005 # learning rate for optimizer
n_epochs = 20000 # nbr of epochs for training (N_e in paper)
n_success = 20 # number of (successful) initilisations
seed = 1234 # seed for initilisation
n_final = 12 # maximal final amount of parameters in the model (N_f in paper)

# Hyper-parameters for loss and pruning
lambda = 1e-4 # parameter for L1 regularisation
mu = 1e-5 # parameter for logloss (amount)
lim_logloss = 1e-2 # parameter for logloss (limit for loss)
delta = 1e-3 # limit for setting small weights and biases to zero


## Find expression for V1
@time model, c = train_multipleinit(data_train_V1, learning_rate, n_epochs, n_success, seed, delta, n_final)
print("Max 12 parameters, expression for V1: ")
exprV1 = find_expr(c, max_V1) # Find symbolic expression after training

# Plot result
# fit_scatter(data_train_V1, max_V1) # Scatter plot
p_V1 = fit_function(exprV1, 1) # Function plot compared to training data

# Find expression for V2
@time model, c = train_multipleinit(data_train_V2, learning_rate, n_epochs, n_success, seed, delta, n_final)
print("Max 12 parameters, expression for V2: ")
exprV2 = find_expr(c, max_V2) # Find symbolic expression after training

# Plot result
# fit_scatter(data_train_V2, max_V2) # Scatter plot
p_V2 = fit_function(exprV2, 2) # Function plot compared to training data

plot(p_V1, p_V2, layout = (2, 1), size = (600, 600))


## Reduced model size
n_final = 6 # maximal final amount of parameters in the model (N_f in paper)

# V1
@time model, c = train_multipleinit(data_train_V1, learning_rate, n_epochs, n_success, seed, delta, n_final)
print("Max 6 parameters, expression for V1: ")
exprV1 = find_expr(c, max_V1) # Find symbolic expression after training

# Plot result
# fit_scatter(data_train_V1, max_V1) # Scatter plot
# fit_function(fV1, exprV1, 1) # Function plot compared to training data


# V2
@time model, c = train_multipleinit(data_train_V2, learning_rate, n_epochs, n_success, seed, delta, n_final)
print("Max 6 parameters, expression for V2: ")
exprV2 = find_expr(c, max_V2) # Find symbolic expression after training

# Plot result
# fit_scatter(data_train_V2, max_V2) # Scatter plot
# fit_function(fV2, exprV2, 2) # Function plot compared to training data


## No limit on final model size
# V2
n_success = 1
lambda = 0 # parameter for L1 regularisation
mu = 0 # parameter for logloss (amount)
delta = 0 # limit for setting small weights and biases to zero

@time model, c = train_multipleinit(data_train_V2, learning_rate, n_epochs, n_success, seed, delta, 100, false)
print("No restriction on parameters, expression for V2: ")
exprV2 = find_expr(c, max_V2) # Find symbolic expression after training

# Plot result
# fit_scatter(data_train_V2, max_V2) # Scatter plot
# fit_function(fV2, exprV2, 2) # Function plot compared to training data



