"""
    struct2ckpt(s; prefix="/Params")

Calls `struct2ckpt!` with an empty `Dict`.
"""
function struct2ckpt(s; prefix="/Params")
    r = Dict{Any,Any}()
    struct2ckpt!(r,s,prefix=prefix)
    return r
end

"""
    struct2ckpt!(d::Dict, s; prefix="/Params")

Convert a struct `s` to a checkpoint dictionary e.g.:

    m = Chain(Dense(1,1),...)

    -> "/Params/Chain-layers/Tuple-1/Dense-b" => Float32[0.0]
       "/Params/Chain-layers/Tuple-1/Dense-W" => Float32[-0.783413]
       ...

Recursion is stopped at `AbstractArray`s.
"""
struct2ckpt!(d::Dict, s; prefix="/Params") = flatten!(d,prefix,s)

function flatten!(out::Dict, key::String, x::AbstractArray)
    if !haskey(out, key)
        out[key] = x
    else
        error("Key '$key' already exists!")
    end
end

function flatten!(out::Dict, key::String, s)
    for k in fieldnames(typeof(s))
        x = getfield(s,k)
        n = s |> typeof |> nameof
        flatten!(out, "$key/$n-$(string(k))", x)
    end
end
