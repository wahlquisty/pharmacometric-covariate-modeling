# 220208 Functions for equation learning

using Flux, Random, Zygote

"""
    l1_loss()
l1 regularisation. Calculates the l1 penalty as sum of absolut value of weights of the network.
"""
function l1_loss()
    l = [1 3 5]
    l1 = 0
    for i in l
        l1 += sum(abs, model[i].weight)
    end
    return l1
end

"""
    log_loss(x)
Takes input data x to network. Calculates penalty for a close to zero denominator (< lim_logloss) in division block.
"""
function log_loss(x)
    cdiv = c[3:end]
    sumlogloss = 0
    z1 = model[1](x) # calculate output from first dense layer
    z2 = functionlayer(z1, [c[1], c[3]]) # output after first activation function
    z3 = model[3](z2) # output from second dense layer
    for l = [1 3] # layers with division
        if l == 1 # access the right outputs for division
            z = z1
        elseif l == 3
            z = z3
        end
        den = z[6, :] # denominator in division
        if !checkdivisionzero(l, cdiv) # if there is no division (left), return logloss = 0
            logi = findall(abs.(den) .< lim_logloss)
            if length(logi) != 0
                sumlogloss += sum(-log.(abs.(den[logi])))
            end
        end
    end
    return sumlogloss
end

"""
    loss(x,y)
Loss function, calculates the total loss where x is input data and y is output data to compare with. Calculates the mse error together with l1 regularisation and logarithmic loss for division function. The degree of l1 regularisation and log loss is determined by lambda and mu, respectively.
"""
function loss(x, y)
    mseloss = Flux.mse(model(x), y) # l2 norm
    totalloss = mseloss + lambda * l1_loss() + mu * log_loss(x)
    global epoch_loss += totalloss
    return totalloss
end

"""
    train_nn!(loss, ps, data, opt)
Train neural network with input (loss function, parameters, training data, optimizer) in Flux using Zygote for calculation of gradient. Updates model parameters in model.
"""
function train_nn!(loss, ps, data, opt)
    ps = Flux.Params(ps)
    for d in data
        train_loss, back = Zygote.pullback(() -> loss(d...), ps)
        gs = back(one(train_loss))
        Flux.update!(opt, ps, gs)
    end
end

"""
    findsmallest()
Finds smallest model parameter (weight or bias) and returns it. Returns layer, index in matrix, bias/weight and its value.
"""
function findsmallest()
    minval = typemax(Int32)
    minind = 0
    layer = 0
    bw = :bias # initial value
    for i = [1 3 5]
        maskedw = model[i].weight .* model[i].W_mask # masked weights
        maskedb = model[i].bias .* model[i].b_mask # masked biases
        nzw = findall(abs.(maskedw) .>= eps(Float32))
        if length(nzw) != 0
            valw, indw = findmin(abs.(maskedw)[nzw])
        else
            valw = typemax(Int32)
            indw = 0
        end

        nzb = findall(abs.(maskedb) .>= eps(Float32))
        if length(nzb) != 0
            valb, indb = findmin(abs.(maskedb)[nzb])
        else
            valb = typemax(Int32)
            indb = 0
        end

        if valb < minval # choose the smallest of these two
            minval = valb
            bw = :bias
            layer = i
            minind = nzb[indb]
        end
        if valw < minval
            minval = valw
            bw = :weight
            layer = i
            minind = nzw[indw]
        end
    end
    return layer, Tuple(minind), bw, minval
end

"""
    removesmallest()
Removes smallest model parameter (weight or bias), only if network is non-empty. Returns if the removal was successful, layer, index, bias/weight.
"""
function removesmallest()
    layer, ind, bw, _ = findsmallest()
    succ_rem = removeparam(layer, bw, ind)
    return succ_rem
end

"""
    removeparam(layer, bw, index)
Removes model parameter at layer, bias/weight, index. Returns true if the removal was successful.
"""
function removeparam(layer, bw, index)
    if layer != 0 # only remove smallest if network is non-empty
        maskparam(model[layer], bw, index)
        return true
    end
    return false
end

"""
    setsmalltozero(lim)
Sets all model parameters (weights and biases) to zero if they are smaller than lim.
"""
function setsmalltozero(lim)
    # if weights/bias are small enough, set to zero
    for i = [1 3 5] # layers with w,b
        rw, cw = size(model[i].weight)
        wm = model[i].weight .* model[i].W_mask
        for j = 1:rw
            for l = 1:cw
                if abs(wm[j, l]) < lim
                    removeparam(i, :weight, (j, l))
                end
            end
        end
        lb = length(model[i].bias)
        lm = model[i].bias .* model[i].b_mask
        for l = 1:lb
            if abs(lm[l]) < lim
                removeparam(i, :bias, l)
            end
        end
    end
end

"""
    replace_functions(c)
Check if activation functions can be replaced by unit functions or set to zero. Input c contains booleans whether exp and div are still active in the network, for each layer. Returns the new c (also modifies model).
"""
function replace_functions(c)
    cexp1, cexp2, cdiv1, cdiv2 = tuple(c...)
    for layer = [1 3]
        if layer == 1
            cexp1 = setexp(layer, cexp1) # For exponential function
        elseif layer == 3
            cexp2 = setexp(layer, cexp2) # For exponential function
        end
        setmultiplication(layer) # For multiplication function
        cdiv1, cdiv2 = setdivision(layer, [cdiv1, cdiv2]) # Division
    end
    return [cexp1, cexp2, cdiv1, cdiv2]
end

""" 
    setexp(layer, cexp)
Set exponential function to unit function if weights are zero. Returns c = [cexp1,cexp2].
"""
function setexp(layer, cexp)
    wm = model[layer].W_mask # weight matrix mask
    if sum(wm[2, :]) == 0 && (cexp == false) # if all weights are zero and functions has not yet been changed
        cexp = true
        model[layer].bias[2] = exp(model[layer].bias[2] .* model[layer].b_mask[2])
        print("Set exp to 1 in layer ", layer, "\n")
    end
    return cexp
end

"""
    setmultiplication(layer)
Remove multiplication function from training if one of a,b in a*b is zero at specific layer. Returns true if this was done. FIXME: is zeroed at every epoch.
"""
function setmultiplication(layer)
    wm = model[layer].W_mask # weight matrix mask

    if sum(wm[3, :]) == 0 # if one of the terms are zero, set the other to zero + bias + in wm3
        model[layer].W_mask[4, :] .= 0
        maskparam(model[layer], :bias, 3)
        maskparam(model[layer], :bias, 4)
        model[layer+2].W_mask[:, 3] .= 0
        # print("Set multiplication to 0 in layer $l\n")
        return true
    end
    if sum(wm[4, :]) == 0 # if one of the terms are zero, set the other to zero
        model[layer].W_mask[3, :] .= 0
        maskparam(model[layer], :bias, 3)
        maskparam(model[layer], :bias, 4)
        model[layer+2].W_mask[:, 3] .= 0
        # print("Set multiplication to 0 in layer $l\n")
        return true
    end
    return false
end

""" 
    setdivision(layer, cdiv)
Remove division from training at layer if nominator is zero or if denominator has only zero weight. If denominator only has zero weight, multiply nominator with bias in denominator and replace entire division with unit function.
"""
function setdivision(layer, cdiv)
    wm = model[layer].W_mask # weight matrix with mask
    bm = model[layer].b_mask # weight matrix 1

    cdiv1, cdiv2 = tuple(cdiv...)

    if sum(wm[5, :]) == 0 && bm[5] == 0 # If nominator is zero, remove denominator from training
        maskparam(model[layer], :bias, 6)
        model[layer].W_mask[6, :] .= 0.0
        model[layer].bias[6] = 1
        model[layer].weight[6, :] = zeros(length(model[layer].weight[6, :]), 1)
    end
    if sum(wm[6, :]) == 0 # If denominator is zero, replace node with value in nominator and remove denominator from training
        if (cdiv1 == false && layer == 1) || (cdiv2 == false && layer == 2)
            model[layer].weight[5, :] .= model[layer].weight[5, :] ./ (sum(model[layer].weight[6, :] .* model[layer].W_mask[6, :]) + model[layer].bias[6] .* model[layer].b_mask[6])
            model[layer].bias[5] = model[layer].bias[5] ./ (sum(model[layer].weight[6, :] .* model[layer].W_mask[6, :]) + model[layer].bias[6] .* model[layer].b_mask[6])
            print("Remove denominator in division in layer ", layer, "\n")
            model[layer].W_mask[6, :] .= 0.0
            model[layer].b_mask[6, :] .= 0.0
            model[layer].bias[6] = 1.0
            if cdiv1 == false
                cdiv1 = true
            elseif cdiv2 == false
                cdiv2 = true
            end
        end
    end
    return [cdiv1, cdiv2]
end

"""
    initdivision(model)
Initialize division to avoid zero in denominator.
"""
function initdivision(model)
    model[1].weight[6, :] .= 1.0
    model[3].weight[6, :] .= 1.0
    model[1].bias[6] = 1.0
    model[3].bias[6] = 1.0
    return model
end

"""
    checkdivisionzero(layer, cdiv)
Check if the division function has been removed at a specific layer.
"""
function checkdivisionzero(layer, cdiv)
    cdiv1, cdiv2 = tuple(cdiv...)
    if layer == 1
        return cdiv1
    elseif layer == 3
        return cdiv2
    else
        return print("Error in checkdivisionzero")
    end
end

# """ Function layer for network"""
# function functionlayer1(x, c) # layer with functions
#     cexp1, _, cdiv1, _ = tuple(c...)
#     if cexp1 == false && cdiv1 == false
#         return [x[1, :] exp.(x[2, :]) (x[3, :] .* x[4, :]) (x[5, :] ./ x[6, :])]'
#     elseif cexp1 == true && cdiv1 == false
#         return [x[1, :] x[2, :] (x[3, :] .* x[4, :]) (x[5, :] ./ x[6, :])]'
#     elseif cexp1 == false && cdiv1 == true
#         return [x[1, :] exp.(x[2, :]) (x[3, :] .* x[4, :]) x[5, :]]'
#     else
#         return [x[1, :] x[2, :] (x[3, :] .* x[4, :]) x[5, :]]'
#     end
# end

# function functionlayer2(x, c) # layer with functions
#     _, cexp2, _, cdiv2 = tuple(c...)
#     if cexp2 == false && cdiv2 == false
#         return [x[1, :] exp.(x[2, :]) (x[3, :] .* x[4, :]) (x[5, :] ./ x[6, :])]'
#     elseif cexp2 == true && cdiv2 == false
#         return [x[1, :] x[2, :] (x[3, :] .* x[4, :]) (x[5, :] ./ x[6, :])]'
#     elseif cexp2 == false && cdiv2 == true
#         return [x[1, :] exp.(x[2, :]) (x[3, :] .* x[4, :]) x[5, :]]'
#     else
#         return [x[1, :] x[2, :] (x[3, :] .* x[4, :]) x[5, :]]'
#     end
# end

"""
    functionlayer(x,c)
Activation functions. Calculates output for input x and c contains information of if exp function and div function are activated or not.
"""
function functionlayer(x, c) # activation functions
    cexp, cdiv = tuple(c...)
    if cexp == false && cdiv == false
        return [x[1, :] exp.(x[2, :]) (x[3, :] .* x[4, :]) (x[5, :] ./ x[6, :])]'
    elseif cexp == true && cdiv == false
        return [x[1, :] x[2, :] (x[3, :] .* x[4, :]) (x[5, :] ./ x[6, :])]'
    elseif cexp == false && cdiv == true
        return [x[1, :] exp.(x[2, :]) (x[3, :] .* x[4, :]) x[5, :]]'
    else
        return [x[1, :] x[2, :] (x[3, :] .* x[4, :]) x[5, :]]'
    end
end

"""
    prune(n_final,c)
Prunes the network down to size n_final parameters given the information in c of if exp and division are activated.
"""
function prune(n_final, c)
    # node = 0
    ntrainparams = sum(model[1].W_mask) + sum(model[3].W_mask) + sum(model[1].b_mask) + sum(model[3].b_mask) + sum(model[5].W_mask) + sum(model[5].b_mask) # number of actual training parameters

    while ntrainparams > n_final # remove parameters if the network is too large
        succ_rem = removesmallest() # remove smallest node (if possible)
        if succ_rem == false # if no network left -> break training
            return c
        end
        c = replace_functions(c)

        ntrainparams = sum(model[1].W_mask) + sum(model[3].W_mask) + sum(model[1].b_mask) + sum(model[3].b_mask) + sum(model[5].W_mask) + sum(model[5].b_mask) # number of training parameters left
    end
    return c
end

"""
    init_model(k)
Initialize model at seed k.
"""
function init_model(k)
    # Training parameters
    nd = 6 # nbr of input nodes to activation function
    nout = 1 # 2 # nbr of outputs

    @show k
    Random.seed!(k) # seed
    # Booleans for which functions are active in the activation functions
    global c = [false, false, true, false] # exp layer 1, exp layer 2, div layer 1, div layer 2
    # NB: changed to remove division in layer 1 initially!
    c1 = [c[1] c[3]] # layer 1
    c2 = [c[2] c[3]] # layer 2
    # Create model
    global model = Chain(
        DenseWithMask(2, nd),
        x -> functionlayer(x, c1),
        DenseWithMask(nd - 2, nd), # two reduced nodes (multiplication and division)
        x -> functionlayer(x, c2),
        DenseWithMask(nd - 2, nout)
    )
    # Initialise network (set denominator in division to nonzero)
    model = initdivision(model)
    return c, model
end

"""
    train_model(k, n_epochs, learning_rate, delta)
Train model at seed k for n_epochs with learning rate learning_rate. After second half of training, set all parameters with a value smaller than delta to zero and remove from further training.
"""
function train_model(data_train, k, n_epochs, learning_rate, delta)
    x_train, y_train = tuple(data_train[1]...)

    global c, model = init_model(k)
    global epoch_loss = 0

    training_params = Flux.params(model) # parameters to train
    opt = ADAM(learning_rate) # optimizer with learning rate

    @printf "Loss before training: %.4f \n" loss(x_train, y_train) # loss before training
    loss_train = zeros(n_epochs, 1)
    @time for epoch = 1:n_epochs

        global epoch_loss = 0
        if epoch % 1000 == 0 # print epoch
            print("Epoch: ", epoch, "\n")
        end

        train_nn!(loss, training_params, data_train, opt) # training
        loss_train[epoch] = epoch_loss

        if isnan(loss(x_train, y_train)) || isinf(loss(x_train, y_train)) #check if NaN (no network left -> break training)
            return c, model, loss_train
        end

        if epoch > n_epochs / 2 # Second half of training, start pruning
            setsmalltozero(delta)
            c = replace_functions(c)
            # opt.eta = 0.001 # update learning rate
        end
    end
    return c, model, loss_train
end

"""
    train_multipleinit(data_train, learning_rate, n_epochs, n_success, seed, delta, n_final, pruning=true)
Training for several initilisations. data_train: training data ordered as [x_train y_train].n_success: number of initialisations that results in a connected network. seed: initialisation starting at Random.seed(). n_epochs: number of training epochs for each initialisation. learning_rate: the optimizers learning rate. n_final: the (maximal) final number of parameters left in the network if pruning=true. delta: after 50 % of the training (and for all following epochs), all parameters with a value smaller than delta are set to zero. pruning: if the network should be pruned down to size n_final after n_epochs, nominal value true.
"""
function train_multipleinit(data_train, learning_rate, n_epochs, n_success, seed, delta, n_final=12, pruning = true)
    x_train, y_train = tuple(data_train[1]...)
    max_interval = seed + n_success - 1
    loss_final = typemax(Int32) # large number for initilisation
    model_final = []
    training_loss_final = []
    c_final = [false, false, false, false]
    n_s = 0 # nbr of successful trainings so far

    k = seed
    while k <= max_interval && n_s < n_success
        c, model, loss_train = train_model(data_train, k, n_epochs, learning_rate, delta)

        if pruning == true # prune after training to size n_final
            c = prune(n_final, c)
        end

        n_trainparams = sum(model[1].W_mask) + sum(model[3].W_mask) + sum(model[1].b_mask) + sum(model[3].b_mask) + sum(model[5].W_mask) + sum(model[5].b_mask) # number of training parameters

        print("Number of training parameters left: ", n_trainparams, "\n")
        @printf "Loss after training: %.4f \n" loss(x_train, y_train) # loss after training

        # Final model
        lossk = loss(x_train, y_train)
        if lossk < loss_final
            loss_final = lossk
            model_final = model
            training_loss_final = loss_train
            c_final = c
        end
        if !(isnan(loss(x_train, y_train)) || isinf(loss(x_train, y_train)))
            n_s += 1
        end
        if k == max_interval && n_s < n_success
            max_interval += 1
        end
        k += 1
        print("Nbr of successful trainings: ", n_s, "\n")
    end

    model = model_final
    n_trainparams = sum(model[1].W_mask) + sum(model[3].W_mask) + sum(model[1].b_mask) + sum(model[3].b_mask) + sum(model[5].W_mask) + sum(model[5].b_mask) # number of actual training parameters

    print("Final number of parameters: ", n_trainparams, "\n")
    @printf "Loss after multiple initilisations: %.4f \n" loss(x_train, y_train) # loss after training
    return model, loss_final, training_loss_final, c_final

end


# Symbolic expression after training
function find_expr(c, max_V)
    Y1 = layer2string(1, ["x_1/$max_AGE"; "x_2/$max_WGT"], c)
    Y2 = layer2string(2, Y1, c)
    Y3 = layer2string(3, Y2, c)
    Y4 = layer2string(4, Y3, c)
    Y = layer2string(5, Y4, c)

    expr = Meta.parse("$max_V" * "*(" * Y[1] * ")") # outmut multiplied with max v1

    @printf "Expression: %s" string(expr)
    return expr
end


## Plotting functions. -- Study result after training- scatter plot

function fit_scatter(data_train, max_V)
    x_train, y_train = tuple(data_train[1]...)
    n_data = length(y_train)
    p = scatter(1:n_data, y_train' .* max_V, label = "True volume", xlabel = "Patient #")
    scatter!(p, 1:n_data, model(x_train)' .* max_V, label = "Predicted volume")
    displai(p)
end

function fit_function(f, expr, V) # FIXME: add inputs?
    expr_1 = :(g(x_1, x_2) = $expr)
    eval(expr_1)

    if V == 1 # V1
        p = scatter(sort(WGT), f.(sort(AGE), sort(WGT)), xlabel = "Weight (kg)", ylabel = "V1 (litres)", label = "True volume")
        plot!(p, sort(WGT), g.(sort(AGE), sort(WGT)), label = "Predicted V1")
        return p
    elseif V == 2 # V2
        ran_x = range(0.001, stop = 1, length = 30) * max_AGE
        ran_y = range(0.001, stop = 1, length = 30) * max_WGT
        p = plot(ran_x, ran_y, g, st = :surface, camera = (30, 50), xlabel = "Age (years)", ylabel = "Weight (kg)", zlabel = "V2 (litres)", grid = true)
        return p
    end
end