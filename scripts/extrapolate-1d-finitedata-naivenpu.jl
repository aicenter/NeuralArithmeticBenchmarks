using DrWatson
@quickactivate

using Flux
using NeuralArithmetic
using Parameters

include(srcdir("bench.jl"))
#include(srcdir("struct2ckpt.jl"))

FORCERUN = false
NRRUNS   = 10
SAVEPATH = datadir(splitext(basename(@__FILE__))[1])

@with_kw struct Ext1DFConfig
    T        = :Float32

    dx       = 0.01
    xstart   = -1
    xend     = 1
    xtstart  = -2
    xtend    = 2

    hdim     = 2

    nrepochs = 200
    Opt      = :RMSProp
    lr       = 1e-3

    nrrun    = 1
end

"""
    datageneration(config::Ext1DFConfig)

Takes a config and returns a tuple two datasets: `(train,test)`.  The train
dataset should be an iterable that contains inputs and labels, i.e.:

    train = ((X,Y) for i in 1:nrepochs)

The testing dataset just just a tuple of inputs and labels:

    test  = (Xt,Yt)
"""
function datageneration(config::Ext1DFConfig)
    @unpack nrepochs, dx, xstart, xend, xtstart, xtend, T = config

    f(x::Real) = x^2

    # make sure we are using type `T` and add a batch dimension
    prep(x) = reshape(eval(T).(collect(x)), 1, :)

    x     = xstart:dx:xend
    xt    = vcat(xtstart:dx:xstart, xend:dx:xtend) 
    X     = x |> prep
    Xt    = xt |> prep  # non-overlapping test set
    Y     = f.(X)
    Yt    = f.(Xt)
    train = ((X,Y) for i in 1:nrepochs)
    test  = (Xt,Yt)
    (train, test) 
end

function buildmodel(config::Ext1DFConfig)
    @unpack hdim, T = config
    model = Chain(NaiveNPU(1,hdim), NAU(hdim,1))

    # make sure model is of eltype T
    fmap(x->eval(T).(x), model)
end

function training!(config::Ext1DFConfig, model, train, test; kw...)

    # data error. could also be something else
    mse(x,y)  = Flux.mse(model(x), y)
    # maybe some regularization?
    reg(ps)   = sum(x->sum(abs,x), ps)
    # final loss that is passed to `training!`
    loss(x,y) = mse(x,y)  # + reg(ps)

    training!(config, loss, model, train, test; kw...)
end

config = Ext1DFConfig()
bench(config, SAVEPATH, NRRUNS, force=FORCERUN)
