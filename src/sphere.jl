include("util.jl")
using Plots
function sphere(h;a=1)
    x(ϕ,θ) = a .* [cos(θ)*sin(ϕ),sin(θ)*sin(ϕ),cos(ϕ)]
    dϕ = π/round(π*a/h)
    mapreduce(vcat,0.5dϕ:dϕ:π) do ϕ
        dθ = 2π/round(2π*a*sin(ϕ)/h)
        param_props.(x,ϕ,0.5dθ:dθ:2π,dϕ*dθ)
    end
end
h = 0.1; panels = sphere(h)
x,y,z = eachrow(stack(get.(panels,"x",0)))
plot(x,y,z,camera=(-20,20),legend=false)
dA = get.(panels,"dA",0)
sum(dA)/4π-1
plot(X[3,:],dA/h^2,ylim=(0,2),xlabel="z",ylabel="dA/h²",legend=false)

include("solve.jl")
A = ∂ₙG.(panels,permutedims(panels))
b = -Uₙ.(panels)
q = A \ b
A*q ≈ b
ma = added_mass(q,panels)
ma[1]/(2π/3)-1
u,v,w = eachrow(body_velocity(q,panels))
equator = filter(i->abs(z[i])<h,1:length(panels))
quiver(x[equator],y[equator],quiver = (u[equator],v[equator]), aspect_ratio=:equal)