include("util.jl")
using Plots
function sphere(n,m;a=1)
    ϕ = range(0,π,length=n+1)[1:n]; dϕ = ϕ[2]
    θ = range(0,2π,length=m+1)[1:m]; dθ = θ[2]
    x(ϕ,θ) = a .* [cos(θ)*sin(ϕ),sin(θ)*sin(ϕ),cos(ϕ)]
    p = @. param_props(x,ϕ'+0.5dϕ,θ+0.5dθ,dϕ*dθ)
    return reshape(p,:),stack(get.(p,"x",0))
end
panels,X = sphere(10,20,a=1);
scatter(X[1,:,:],X[2,:,:],X[3,:,:],camera=(-20,20),legend=false)
sum(get.(panels,"dA",0))/4π-1

include("solve.jl")
A = uₙ.(panels,permutedims(panels),ϕ)
U = [0,0,1]
b = -Uₙ.(panels;U)
q = A \ b
A*q≈b
plot(-10:0.1:-1,x->u([0,0,x],q,panels,G;U)[3])