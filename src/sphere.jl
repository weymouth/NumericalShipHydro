include("util.jl")
using Plots
"""
    sphere(h;a=1) -> Table(panel_props)

Sample a sphere of radius `a` such that the arc-length ≈ h in each direction.
θ₁=[0,π]: azimuth angle, θ₂=[0,2π]: polar angle.
"""
function sphere(h;a=1)
    S(θ₁,θ₂) = a .* [cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    dθ₁ = π/round(π*a/h)
    mapreduce(vcat,0.5dθ₁:dθ₁:π) do θ₁
        dθ₂ = 2π/round(2π*a*sin(θ₁)/h) # get polar step at this azimuth
        param_props.(S,θ₁,0.5dθ₂:dθ₂:2π,dθ₁,dθ₂)
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
A,b = ∂ₙϕ.(panels,permutedims(panels)),-Uₙ.(panels);
q = A \ b; A*q ≈ b
ma = added_mass(q,panels); ma[1]/(2π/3)-1
u,v,w = eachrow(body_velocity(q,panels)); extrema(u)
quiver(x[equator],y[equator],quiver = (u[equator],v[equator]), aspect_ratio=:equal)
plot(0:0.01:π,x->1.5sin(x),label="exact",xlabel="θ",ylabel="|u|")
θ,mag_u = @. abs(atan(y[equator],x[equator])),hypot(u[equator],v[equator]);
scatter!(θ,mag_u,label="numeric")
title!("Sphere surface velocity magnitude with h/a=$h")