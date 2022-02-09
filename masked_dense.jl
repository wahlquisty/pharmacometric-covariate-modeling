using Flux

struct DenseWithMask{F, M<:AbstractMatrix, B}
	weight::M
	bias::B
	σ::F
	W_mask::M
	b_mask::B

	# Initialize struct, takes the same arguments as a normal dense layer.
	function DenseWithMask(args...;kwargs...)
		# Create a normal layer just to get identical syntax and initialization as a
		# normal Dense layer. A bit overkill but works.
		l = Dense(args...;kwargs...)

		# Create mask for the params
		W_mask = fill!(similar(l.weight), 1)
		b_mask = fill!(similar(l.bias), 1)

		M,B,F = typeof.( (l.weight,l.bias,l.σ) )
		return new{F,M,B}(l.weight, l.bias, l.σ, W_mask, b_mask)
	end
end

# I am not completely sure what these lines do, but the second specify which
# fields in the struct should be consider parameters when calling Flux.params()
Flux.@functor DenseWithMask
Flux.trainable(l::DenseWithMask) = (l.weight,l.bias)

# This is the function that is called when the layer is evaluated
function (l::DenseWithMask)(x::AbstractVecOrMat)
	W = l.weight .* l.W_mask
	return l.σ.(W*x .+ (l.bias .* l.b_mask) )
end

# helper for masking and unmasking
function set_mask_and_val(l::DenseWithMask,param,idx,mask)
	if param == :weight
		l.W_mask[idx...] = mask
	elseif param == :bias
		l.b_mask[idx...] = mask
	else
		error("Invalid parameter name")
	end
end

# These are the functions that mask and unmask parameters
maskparam(l,param,idx) = set_mask_and_val(l,param,idx,0)
unmaskparam(l,param,idx) = set_mask_and_val(l,param,idx,1)

################################################################################
################################################################################
################################################################################