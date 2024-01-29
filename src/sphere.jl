include("util.jl")
using Plots
function sphere(h;a=1)
    x(ϕ,θ) = a .* [cos(θ)*sin(ϕ),sin(θ)*sin(ϕ),cos(ϕ)]
    dϕ = π/round(π*a/h)
    mapreduce(vcat,0.5dϕ:dϕ:π) do ϕ
        dθ = 2π/round(2π*a*sin(ϕ)/h)
        param_props.(x,ϕ,0.5dθ:dθ:2π,dϕ,dθ)
    end |> Table
end
h = 0.3; panels = sphere(h); N=length(panels)
x,y,z = eachrow(stack(panels.x));
plot(x,y,z,camera=(-20,20),legend=false)
sum(panels.dA)/4π-1
plot(z,panels.dA/h^2,ylim=(0,2),xlabel="z",ylabel="dA/h²",legend=false)
equator = filter(i->-h<z[i]<h,1:length(panels));
nx,ny,_ = eachrow(stack(panels.n));
quiver(x[equator],y[equator],quiver = (nx[equator],ny[equator]), aspect_ratio=:equal)

include("solve.jl")
A = ∂ₙϕ.(panels,permutedims(panels));
b = -Uₙ.(panels);
q = A \ b;
A*q ≈ b
ma = added_mass(q,panels)
ma[1]/(2π/3)-1
u,v,w = eachrow(body_velocity(q,panels))
extrema(u)
quiver(x[equator],y[equator],quiver = (u[equator],v[equator]), aspect_ratio=:equal)
plot(0:0.01:π,x->1.5sin(x),label="exact",xlabel="θ",ylabel="|u|")
θ,mag_u = @. abs(atan(y[equator],x[equator])),hypot(u[equator],v[equator]);
scatter!(θ,mag_u,label="numeric")
title!("Sphere surface velocity magnitude with h/a=$h")