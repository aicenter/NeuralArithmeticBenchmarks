"""
    make_ckptcallback(f::Function, folder::String, args...)

Create a callback that can be used with e.g. `Flux.train!` to save checkpoints.
The returned callback has to be called with an `nriter` which will be used to
save the checkpoints at '/dir/checkpoint-nriter=\$nriter.bson'.

The function `f` has to return a dictionary as constructed by
`struct2ckpt`.
"""
function make_ckptcallback(f::Function, folder::String, args...)
    function cb(nriter::Int)
        d = f(args...)
        d["_ckpt/nriter"] = nriter

        nrpadded = lpad(string(nriter),10,"0")
        fi = joinpath(folder, "checkpoint-nriter=$nrpadded.bson")
        msg = "Saving checkpoint at $fi"
        println(msg)
        tagsave(fi, d)
    end
end


"""
    @ckptcallback

Wraps `make_ckptcallback` to create a callback that saves checkpoints.
* `f`: Function that returns a dictionary as constructed by `struct2ckpt`.
* `dir`: checkpoint save dir
* `nriter`: initial checkpoint number

```julia-repl
julia> @ckptcallback f dir nriter
```
"""
macro ckptcallback(f, folder, nriter)
    cb = esc(:(make_ckptcallback($f,$folder)))
    nr = esc(:($nriter))
    :(()->$cb($nr))
end

function extract(d::Array{<:Dict}, key::String, is::Int...)
    vi = map(d) do x
        v = x[key][is...]
        i = x["_ckpt/nriter"]
        (i,v)
    end
    map(first,vi), map(last,vi)
end
# for scalars:
function extract(d::Array{<:Dict}, key::String)
    vi = map(d) do x
        v = x[key]
        i = x["_ckpt/nriter"]
        (i,v)
    end
    map(first,vi), map(last,vi)
end
