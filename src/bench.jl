datageneration(config) = error("`datageneration` is not implemented for $(typeof(config))!")

buildmodel(config) = error("`buildmodel` is not implemented for $(typeof(config))!")

"""
    training!(config, loss::Function, model, train, test; verbose=true, ckpt_dir=nothing, ckpt_step=100)

Train the model and return `merge(config, result)`.  If `ckpt_dir` is not
nothing, saves a checkpoint of the model every `ckpt_step` iterations.
"""
function training!(config, loss::Function, model, train, test;
                   verbose=true, ckpt_dir::String="", ckpt_step=100)
    @unpack Opt, lr = config
    opt = eval(Opt)(lr)

    # define mse for monitoring
    mse(x,y) = Flux.mse(model(x),y)

    nriter = 0
    itercb() = nriter += 1
    callbacks = []
    if verbose
        cb() = (@info "Iteration #$nriter" loss(test...) mse(test...))
        push!(callbacks, Flux.throttle(cb, 0.1))
    end
    if length(ckpt_dir) > 0
        ckpt() = struct2ckpt(model)
        ckptcb = @ckptcallback ckpt ckpt_dir nriter
        push!(callbacks, skipcalls(ckptcb, ckpt_step))
    end
    push!(callbacks, itercb)

    Flux.train!(loss, params(model), train, opt, cb=callbacks)
    (X,Y) = first(train)
    trainloss = loss(X,Y)
    testloss  = loss(test...)
    trainmse  = mse(X,Y)
    testmse   = mse(test...)
    result    = @dict model trainloss testloss trainmse testmse
    merge(struct2dict(config), result)
end

function bench(config; kw...)
    (train,test) = datageneration(config)
    model = buildmodel(config)
    result = training!(config, model, train, test; kw...)
end
