function training!(config, loss::Function, model, train, test; verbose=true, ckpt_dir=nothing)
    @unpack Opt, lr = config
    opt = eval(Opt)(lr)

    # define mse for monitoring
    mse(x,y) = Flux.mse(model(x),y)

    nriter = 0
    itercb() = nriter += 1
    callbacks = []
    if verbose
        push!(callbacks, Flux.throttle(()->(@info loss(test...) mse(test...)),0.1))
    end
    # if !isnothing(ckpt_dir)
    #     ckptcb = @ckptcallback ckpt ckpt_dir nriter
    #     push!(callbacks, ckpt)
    # end
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

function bench(config, path, nrruns; digits=10, kw...)
    for nr in 1:nrruns
        config = typeof(config)(nrrun=nr)

        #calls bench(config) internally
        produce_or_load(path, config, bench; digits=digits, kw...)
    end
end

function bench(config)
    (train,test) = datageneration(config)
    model = buildmodel(config)
    result = training!(config, model, train, test)
end
