include("quad.jl")
len(x) = √sum(abs2,x)
function G(x,y,z,a,b,c;k=1,kwargs...)
    # Wavenumber-scaled vector from source and image source
    r = k .* (x-a,y-b,z-c); r′ = k .* (x-a,y-b,z+c)
    # Rankine and wave potentials
    return 1/len(r)-1/len(r′)+D(r′...;kwargs...)+W(r′...;kwargs...)
end

# Potential and surface elevation
ϕ(x,y,z;kwargs...) = G(x,y,z,0,0,-1;kwargs...) # replace with summation over points
ζ(x,y;h=0.05,kwargs...) = (ϕ(x+h,y,0;kwargs...)-ϕ(x-h,y,0;kwargs...))/2h

using Plots
x,y = -15:0.1:3,-6.05:0.1:6.05
contour(x,y,(x,y)->ϕ(x,y,0.),aspect_ratio=:equal,levels=[-2.5,-2,-1.5,-1,-.5,.5,1,1.5],legend=false)
contour(x,y,ζ,aspect_ratio=:equal,levels=[-2.5,-2,-1.5,-1,-.5,.5,1,1.5],legend=false)