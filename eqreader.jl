# 21112 Get equations in readable form from network structure.

function layer2string(layer, input, c)
    cexp1, cexp2, cdiv1, cdiv2 = tuple(c...)
    if layer in [1 3 5]
        W = model[layer].weight .* model[layer].W_mask
        # W = round.(W, digits = 2) # round for visibility
        B = model[layer].bias .* model[layer].b_mask
        # B = round.(B, digits = 2)
        n_outputs = size(W, 1)
        n_inputs = size(W, 2)

        l_str = String[]
        for j = 1:n_outputs
            push!(l_str, "")
            for k = 1:n_inputs
                w0 = W[j, k]
                if w0 != 0 # weights
                    if w0 < 0
                        l_str[j] = "$(l_str[j])+($(w0))*($(input[k]))"
                    else
                        l_str[j] = "$(l_str[j])+$(w0)*($(input[k]))"
                    end
                end
            end
            b0 = B[j]
            if b0 != 0 # bias
                if b0 < 0
                    l_str[j] = "$(l_str[j])+($(b0))"
                else
                    l_str[j] = "$(l_str[j])+$(b0)"
                end
            end
        end
    elseif layer == 2 || layer == 4
        l_str = String[]
        for i = 1:length(input)
            if i == 4 || i == 6
                # do nothing - ugly but works (not included in either of the cases)
            elseif !isempty(input[i])
                if i == 1
                    push!(l_str, "$(input[i])")
                elseif i == 2
                    if (layer == 2 && cexp1 == false) || (layer == 4 && cexp2 == false)
                        push!(l_str, "exp($(input[i]))")
                    elseif (layer == 2 && cexp1 == true) || (layer == 4 && cexp2 == true)
                        push!(l_str, "$(input[i])")
                    end
                elseif i == 3
                    if "$input[i]" != "" || "$input[i+1]" != ""
                        push!(l_str, "($(input[i]))*($(input[i+1]))")
                    end
                elseif i == 5
                    if (layer == 2 && cdiv1 == false) || (layer == 4 && cdiv2 == false)
                        push!(l_str, "($(input[i]))/($(input[i+1]))")
                    elseif (layer == 2 && cdiv1 == true) || (layer == 4 && cdiv2 == true)
                        push!(l_str, "$(input[i])")
                    end
                end
            else
                push!(l_str,"0") # fixme for better solution to not have *0 in some cases
            end
        end
    end
    return l_str
end

# Y1 = layer2string(1,["x_1/max_AGE"; "x_2/max_WGT"],c)
# Y2 = layer2string(2,Y1,c)
# Y3 = layer2string(3,Y2,c)
# Y4 = layer2string(4,Y3,c)
# Y = layer2string(5,Y4,c)

