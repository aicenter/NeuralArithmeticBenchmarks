"""
    struct2ckpt(s; prefix="/Params")

Calls `struct2ckpt!` with a new ckpt.
"""
function struct2ckpt(s; prefix="/Params")
    r = Dict{Any,Any}()
    struct2ckpt!(r,s,prefix=prefix)
    return r
end

"""
    struct2ckpt!(d::Dict, s; prefix="/Params")

Convert a struct `s` to a checkpoint dictionary e.g.:

    m = Chain(Dense(2,2),...)

    -> "/prefix/layers/1/W" => W
       "/prefix/layers/1/b" => b
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
