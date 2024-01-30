include("util.jl")
function kelvin(x,a;k=1,kwargs...)
    # Wavenumber-scaled vector from source and image source
    r = k * (x-a); r′ = k * (x-a .* [1,1,-1])
    # Rankine and wave potentials
    return 1/hypot(r...)-1/hypot(r′...)+GD(r′...;kwargs...)+GW(r′...;kwargs...)
end

# Near-field non-oscillatory wave
Di(x,y,z,T) = real(expintx(complex((1+T^2)*z,eps(T)+(x+y*T)*hypot(1,T))))
GD(x,y,z;kwargs...) = 2/π*quadgl_inf(T->Di(x,y,z,T);kwargs...)

# Far-field oscillatory wave
Wi(x,y,z,T) = exp((1+T^2)*z)*sin((x-abs(y)*T)*hypot(1,T))
function GW(x,y,z;kwargs...)
    a,b = x/abs(y),√(-5log(10)/z-1) # integral domain
    (a≥b || x==y==0) ? 0. : 4*quadgl_ab(T->Wi(x,y,z,T),max(a,-b),b;kwargs...)
end

# Surface elevation plots
ζ(x,y,q,panels;kwargs...) = derivative(x->φ([x,y,0],q,panels;kwargs...),x)

using Plots
function source_elevation_plot(a=[0,0,-1],k=1)
    ζ(x,y) = derivative(x->kelvin([x,y,0],a;k),x) # unit-point source
    x,y = -15:0.1:3,-6.05:0.1:6.05
    contour(x,y,ζ,aspect_ratio=:equal,levels=[-2.5,-2,-1.5,-1,-.5,.5,1,1.5],legend=false)
end
function elevation_plot(q,panels;kwargs...)
    x,y = -15:0.1:3,-6.05:0.1:6.05
    contour(x,y,(x,y)->ζ(x,y,q,panels;kwargs...),aspect_ratio=:equal,levels=[-2.5,-2,-1.5,-1,-.5,.5,1,1.5],legend=false)
end