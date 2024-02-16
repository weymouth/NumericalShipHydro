using NumericalShipHydro
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

function wigley_hull(hx,hz;L=1,B=1,D=1,kwargs...)
    oneside(ξ,ζ;s=1) = SA[0.5L*ξ,-s*0.5B*wigley(ξ,ζ;kwargs...),D*ζ]
    hull(ξ,ζ) = ξ<1 ? oneside(ξ,ζ) : oneside(2-ξ,ζ,s=-1)
    dξ = 2/round(L/hx); ξ = 0.5dξ-1:dξ:3
    dζ = 1/round(D/hz); ζ = 0.5dζ-1:dζ:0
    param_props.(hull,ξ',ζ,dξ,dζ) |> Table
end
L,B,D = 1,0.1,0.0625; hx,hz=L/30,D/5; hx/hz
panels = wigley_hull(hx,hz;L,B,D); length(panels)
# x,y,z = eachrow(stack(panels.x));
# scatter(x,y,z,xlabel="x",ylabel="y",zlabel="z",legend=false,camera=(60,20))

# A,b = ∂ₙϕ.(panels,permutedims(panels)),-Uₙ.(panels);
# q = A \ b; A*q ≈ b
# u,v,w = eachrow(body_velocity(q,panels)); extrema(u)
# quiver(x[equator],y[equator],quiver = (u[equator],v[equator]), aspect_ratio=:equal)

# wpx,wpy = -L:0.1:L,-1.5B:0.1:1.5B
# inside(x,y,z) = abs(y/0.5B)<wigley(x/0.5L,z/D)
# cₚ(x,y,z,q,panels;U=[1,0,0],G=source) = 1-sum(abs2,U+∇φ([x,y,z],q,panels;G))/sum(abs2,U)
# contour(wpx,wpy,(x,y)->inside(x,y,0.) ? NaN : cₚ(x,y,0.,q,panels),aspect_ratio=:equal)
# title!("Wigley double-body x-y plane cₚ with h/D=$(h/D)")

include("kelvin.jl")
U,G,Fn = [-1,0,0],kelvin,0.316

A,b = ∂ₙϕ.(panels,permutedims(panels);G,Fn),-Uₙ.(panels;U);
heatmap(A,yflip=true,colorbar_title="A",size=(400,300))
for i in 1:size(A,1)
    A[i,i] = 0
end
heatmap(A,yflip=true,colorbar_title="A",size=(400,300))
heatmap(A[1:50,1:50],yflip=true,colorbar_title="A",size=(400,300))
q = A \ b
