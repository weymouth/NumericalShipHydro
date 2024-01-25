include("util.jl")
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

using LinearAlgebra: ×
function wigley_panel(ξ,ζ,dζ,dξ;L=1,B=1,D=1,kwargs...)
    xyz(ξ,ζ) = [0.5L*ξ,0.5B*wigley(ξ,ζ;kwargs...),D*ζ]
    n = derivative(ζ->xyz(ξ,ζ),ζ)×derivative(ξ->xyz(ξ,ζ),ξ)
    mag = hypot(n...)
    xyz(ξ,ζ),n/mag,mag*dζ*dξ
end
wigley_panel(-0.95,-0.05,0.1,0.1;L=10)

function wigley_hull(n,m;double=false,kwargs...)
    ξ = range(-1,1,length=n+1)[1:n]; dξ = ξ[2]+1
    top = double ? 1 : 0
    ζ = range(-1,top,length=m+1)[1:m]; dζ = ζ[2]+1
    @. wigley_panel(ξ'+0.5dξ,ζ+0.5dζ,dξ,dζ;kwargs...)
end
panels = wigley_hull(20,10;L=3)
X = stack(first.(panels))
surface(X[1,:,:]',X[2,:,:]',X[3,:,:]',camera=(150,-10),legend=false)
sum(last.(panels))
# source(x,y,z,a,b,c) = 1/hypot(x-a,y-b,z-c)
# ϕ(x,y,z,panels) = 