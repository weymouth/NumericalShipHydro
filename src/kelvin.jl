include("util.jl")
function G(x,y,z,a,b,c;k=1,kwargs...)
    # Wavenumber-scaled vector from source and image source
    r = k .* (x-a,y-b,z-c); r′ = k .* (x-a,y-b,z+c)
    # Rankine and wave potentials
    return 1/hypot(r...)-1/hypot(r′...)+D(r′...;kwargs...)+W(r′...;kwargs...)
end

# Near-field non-oscillatory wave
Di(x,y,z,T) = real(expintx(complex((1+T^2)*z,eps(T)+(x+y*T)*hypot(1,T))))
D(x,y,z) = 2/π*quadgl_inf(T->Di(x,y,z,T)) 

# Far-field oscillatory wave
Wi(x,y,z,T) = exp((1+T^2)*z)*sin((x-abs(y)*T)*hypot(1,T))
function W(x,y,z)
    a,b = x/abs(y),√(-5log(10)/z-1) # integral domain
    (a≥b || x==y==0) ? 0. : 4quadgl_ab(T->Wi(x,y,z,T),max(a,-b),b)
end

# Potential and surface elevation
ϕ(x,y,z;kwargs...) = G(x,y,z,0,0,-1;kwargs...) # replace with summation over points
ζ(x,y;kwargs...) = derivative(x->ϕ(x,y,0;kwargs...),x)

using Plots
x,y = -15:0.1:3,-6.05:0.1:6.05
contour(x,y,(x,y)->ϕ(x,y,0.),aspect_ratio=:equal,levels=[-2.5,-2,-1.5,-1,-.5,.5,1,1.5],legend=false)
contour(x,y,ζ,aspect_ratio=:equal,levels=[-2.5,-2,-1.5,-1,-.5,.5,1,1.5],legend=false)