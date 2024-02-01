include("solve.jl")
using Plots
wigley(ξ,ζ;c=(0,0,0)) = (1-ζ^2)*(1-ξ^2)*(1+c[1]*ξ^2+c[2]*ξ^4)+c[3]*ζ^2*(1-ζ^8)*(1-ξ^2)^4
function section_lines(;kwargs...) 
    ζ,ξ = -1:0.01:0,-1:0.05:0
    η = wigley.(ξ',ζ;kwargs...)
    plot(η,ζ,ξ,c=:black,legend=false,aspect_ratio=:equal,size=(420,420),xlabel="η",ylabel="ζ")
end
section_lines(c=(0,0,0))
section_lines(c=(0.2,0.,1.))
section_lines(c=(0.6,1.,1.))

function wigley_hull(h;double=false,L=1,B=1,D=1,kwargs...)
    oneside(ξ,ζ;s=1) = [0.5L*ξ,-s*0.5B*wigley(ξ,ζ;kwargs...),D*ζ]
    hull(ξ,ζ) = ξ<1 ? oneside(ξ,ζ) : oneside(2-ξ,ζ,s=-1)
    top = double ? 1 : 0
    n = round(2L/h); dξ = 4/n; ξ = 0.5dξ-1:dξ:3
    m = round((1+top)*D/h); dζ = (1+top)/m; ζ = 0.5dζ-1:dζ:top
    Table(@. param_props(hull,ξ',ζ,dξ,dζ))
end
L,B,D,h = 3,1,1,0.25; panels = wigley_hull(h;L,B,D,double=true); length(panels)
x,y,z = eachrow(stack(panels.x));
plot(x,y,z,camera=(-60,20),legend=false,aspect_ratio=:equal)
sum(panels.dA)/(L*D)
scatter(x/L,panels.dA/h^2,ylim=(0,2),xlabel="x/L",ylabel="dA/h²",legend=false)
equator = filter(i->-h<z[i]<h,1:length(panels));
nx,ny,_ = eachrow(stack(panels.n));
quiver(x[equator],y[equator],quiver = (nx[equator],ny[equator]), aspect_ratio=:equal)

A,b = ∂ₙϕ.(panels,permutedims(panels)),-Uₙ.(panels);
q = A \ b; A*q ≈ b
u,v,w = eachrow(body_velocity(q,panels)); extrema(u)
quiver(x[equator],y[equator],quiver = (u[equator],v[equator]), aspect_ratio=:equal)

wpx,wpy = -L:0.1:L,-1.5B:0.1:1.5B
inside(x,y,z) = abs(y/0.5B)<wigley(x/0.5L,z/D)
cₚ(x,y,z,q,panels;U=[1,0,0],G=source) = 1-sum(abs2,U+∇φ([x,y,z],q,panels;G))/sum(abs2,U)
contour(wpx,wpy,(x,y)->inside(x,y,0.) ? NaN : cₚ(x,y,0.,q,panels),aspect_ratio=:equal)
title!("Wigley double-body x-y plane cₚ with h/D=$(h/D)")

include("kelvin.jl")
L,B,D = 1,1/10,1/16; h=D/4; panels = wigley_hull(h;L,B,D); length(panels)
x,y,z = eachrow(stack(panels.x));
equator = filter(i->-h<z[i]<h,1:length(panels));all(z .< 0)
plot(x,y,z,camera=(-60,20),legend=false,aspect_ratio=:equal)
U,G,Fn = [-1,0,0],kelvin,0.316

plt = plot(xlabel="x/L",ylabel="ζ/Fn²",title="Wigley hull waterline, Fn=$Fn, L/h=$L_h")
for i=4:7
    xgl,wgl = gausslegendre(2^i+1);
    A,b = ∂ₙϕ.(panels,permutedims(panels);G,Fn,xgl,wgl),-Uₙ.(panels;U);
    q = A \ b; A*q ≈ b
    # elevation_plot(q,panels;G,k) very very slow
    L_h =Int(L÷h)
    plot!(plt,-0.5L:h:0.5L,x->ζ(x,0.5B*wigley(2x/L,0),q,panels;G,Fn,xgl,wgl),label=2^i+1)
end
display(plt)