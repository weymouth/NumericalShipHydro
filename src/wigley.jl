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

function wigley_hull(n,m;double=false,L=1,B=1,D=1,kwargs...)
    oneside(ξ,ζ;s=1) = [0.5L*ξ,s*0.5B*wigley(ξ,ζ;kwargs...),D*ζ]
    hull(ξ,ζ) = ξ<1 ? oneside(ξ,ζ) : oneside(2-ξ,ζ,s=-1)
    top = double ? 1 : 0
    ξ = range(-1,3,length=n+1)[1:n]; dξ = ξ[2]+1
    ζ = range(-1,top,length=m+1)[1:m]; dζ = ζ[2]+1
    Table(@. param_props(hull,ξ'+0.5dξ,ζ+0.5dζ,dξ,dζ))
end
L,B,D = 3,1,1; panels = wigley_hull(40,40;L,B,D,double=true);
x,y,z = eachrow(stack(panels.x));
plot(x,y,z,camera=(-60,20),legend=false,aspect_ratio=:equal)
sum(panels.dA)/(L*D)