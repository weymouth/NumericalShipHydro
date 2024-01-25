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

function wigley_hull(n,m;double=false,L=1,B=1,D=1,kwargs...)
    top = double ? 1 : 0
    ξ = range(-1,3,length=n+1)[1:n]; dξ = ξ[2]+1
    ζ = range(-1,top,length=m+1)[1:m]; dζ = ζ[2]+1
    x(ξ,ζ;s=1) = [0.5L*ξ,s*0.5B*wigley(ξ,ζ;kwargs...),D*ζ]
    mirror(ξ,ζ) = ξ<1 ? x(ξ,ζ) : x(2-ξ,ζ,s=-1)
    p = @. param_props(mirror,ξ'+0.5dξ,ζ+0.5dζ,dξ*dζ)
    return reshape(p,:),stack(get.(p,"x",0))
end
panels,X = wigley_hull(40,40,L=3,B=1,double=true);
surface(X[1,:,:]',X[2,:,:]',X[3,:,:]',camera=(-20,20),legend=false)
sum(get.(panels,"dA",0))