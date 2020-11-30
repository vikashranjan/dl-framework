using Zygote, LinearAlgebra, BenchmarkTools,HDF5,MLDatasets,Distributions,Printf

#trainable type
abstract type Trainable
end

# Conv Layer
mutable struct Conv <: Trainable
    # These values will be implicitly learned
    weights::Array
    bias::Array
    # These values will not be learned
    pad::String
end
Conv(in_channels::Int, out_channels::Int, kernel_size::Int,pad::String ) = Conv(randn(Float32,kernel_size,kernel_size,in_channels,out_channels), zeros( out_channels),pad)

#Convolution operation
function (c::Conv)(x::Array)
    input_r,input_c,input_ch = size(x)
    input = deepcopy(x)
    filter_r, filter_c,filter_ch,number_of_fil = size(c.weights)
   
    if c.pad == "same"
        pad_r = (filter_r - 1) ÷ 2
        pad_c = (filter_c - 1) ÷ 2
        
        input_padded = Zygote.Buffer(x,input_r+(2*pad_r), input_c+(2*pad_c))
        for i in 1:input_r, j in 1:input_c
            input_padded[i+pad_r, j+pad_c] = input[i, j]
        end
        input = copy(input_padded)
        input_r, input_c = size(input)
    elseif c.pad == "valid"
        # We don't need to do anything here
    else 
        throw(DomainError(padding, "Invalid padding value"))
    end
    
    if filter_r != filter_c
        throw(DomainError(filter, "Filter row and column must be equals"))
    end
    if input_ch != filter_ch
        throw(DomainError(filter, "Filter and input channel must be equal"))
    end
    result= Zygote.Buffer(x,input_r-filter_r+1, input_c-filter_c+1,number_of_fil)
    result_r, result_c,filter_ch = size(result)
    for i in 1:result_r
        for j in 1:result_c
            for k in 1:number_of_fil
            result[i,j,k] = dot(input[i:i+filter_r-1,j:j+filter_c-1,:],c.weights[:,:,:,k])+c.bias[k]
            end
        end
    end
    return copy(result)
end

#Dense Layer
mutable struct Dense <: Trainable
    # These values will be implicitly learned
    weights::Array
    bias::Array
end

Dense(in_channels::Int, out_channels::Int ) = Dense(randn(Float32,in_channels, out_channels), zeros(1, out_channels))

function (c::Dense)(x::Array)
    return x*c.weights .+ c.bias
end

# MaxPool
struct MaxPool
    kernel_size::Int
    stride::Int
end

# pooling operation
function (c::MaxPool)(x::Array)
    input_r,input_c,input_ch = size(x)
    input = deepcopy(x)
    output_r = Int((input_r-c.kernel_size)/c.stride+1)
    output_c = Int((input_c-c.kernel_size)/c.stride+1)
    
    if output_r != output_c
        throw(DomainError(filter, "Row and column must be equals"))
    end

    result= Zygote.Buffer(x,output_r, output_c,input_ch)
    result_r, result_c,filter_ch = size(result)
    
    for i in range(1,input_r, step=c.stride)
        for j in range(1,input_c, step=c.stride)
            for k in 1:input_ch
            result[Int(floor(i/c.stride))+1,Int(floor(j/c.stride))+1,k] = maximum(input[i:i+c.kernel_size-1,j:j+c.kernel_size-1,k])
            end
        end
    end
    return copy(result)
end

# Relu
struct Relu
end

function (c::Relu)(Z::Array)
    A = max.(0, Z)
    return A
end

#Leaky Relu

struct LeakyRelu
    a::Float32
end

function (c::LeakyRelu)(Z::Array)
    A = max.(c.a, Z)
    return A
end

#Reshape 

struct Reshape
    out_channel::Int
end
function (c::Reshape)(Z::Array)
    return reshape(Z,:,c.out_channel)
end

# SoftMax
struct SoftMax
    dim::Int
end


#softmax function
function (c::SoftMax)(x::Array)
    max_ = maximum(x, dims=c.dim)
    exp_ = exp.(x .- max_)
    return exp_ ./ sum(exp_, dims=c.dim)

end


# LogSoftMax
struct LogSoftMax
    dim::Int
end
function (c::LogSoftMax)(x::Array)
    max_ = maximum(x, dims=c.dim)
    exp_ = exp.(x .- max_)
    log_ = log.(sum(exp_, dims=c.dim))
    return (x .- max_) .- log_
end

#Chain

struct Chain 
    layers::Tuple
end

Chain(x...)=Chain(x)
#Fix single argument

#Predict Chain
function (c::Chain)(X::Array)
    x=deepcopy(X)
    for layer in c.layers
        x=layer(x)
   end
   return x
end

#Get model params
function get_params(model)
    params = Zygote.Params()
    for layer in model.layers
        if typeof(layer) <: Trainable
           push!(params,layer.weights)
           push!(params,layer.bias)
        end
    end
    return params
end

#unsqueeze
unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))

#predict
function predict(model::Chain, X)
    return model(X)
end

#loss function
function cross_entropy_loss(Y,Ypred,ϵ=2.220446049250313e-16)
    return -sum(Y.*log.(Ypred.+ϵ))
   
end

function cross_entropy_loss_with_predict(Y,model::Chain,X,ϵ=2.220446049250313e-16)
    Ypred=model(X)
    return cross_entropy_loss(Y,Ypred,ϵ)
end

function cross_entropy_loss_with_predict_batch(Y::Array,model::Chain,X::Array,ϵ=2.220446049250313e-16)
    batch_size=size(X)[4]
    loss=0
    for i in 1:batch_size
        Ypred=model(X[:,:,:,i])
        loss+=cross_entropy_loss(Y[i],Ypred,ϵ)
    end
    return loss
end

#loss function
function cross_entropy_loss_log_softmax(Y,Ypred)
    return -sum(Y.*(Ypred))
   
end

function cross_entropy_loss_with_predict_batch_log_softmax(Y::Array,model::Chain,X::Array)
    batch_size=size(X)[4]
    loss=0
    for i in 1:batch_size
        Ypred=model(X[:,:,:,i])
        loss+=cross_entropy_loss_log_softmax(Y[i],Ypred)
    end
    return loss
end

function cross_entropy_loss_with_predict_log_softmax(Y,model::Chain,X)
    Ypred=model(X)
    return cross_entropy_loss_log_softmax(Y,Ypred)
end
#one hot encoding
function onehot(l, labels)
    i = something(findfirst(isequal(l), labels), 0)
    i > 0 || error("Value $l is not in labels")
    one_hot_labels=zeros(Int64,size(labels)[1])
    one_hot_labels[i]=1
    return one_hot_labels
  end

  #update params
function update_params!(model,grads, η = 0.001)
    
    for layer in model.layers
        if typeof(layer) <: Trainable
           grad_w = grads[layer.weights]
           grad_b = grads[layer.bias]
           layer.weights .-= η .* grad_w
           layer.bias -= η * grad_b
        end
    end
end