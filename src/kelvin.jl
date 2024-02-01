include("solve.jl")
"""
    kelvin(ξ,a;Fn=1,kwargs...)

Green Function `G(ξ)` for a source at position `α` moving with `Fn≡U/√gL` below 
the free-surface. This function assumes that the free surface location is ζ=0 
and the apparent velocity direction is Û=[-1,0,0]. See Noblesse 1981 for details.
"""
function kelvin(ξ,α;Fn=1,kwargs...)
    α[3] ≥ 0 && throw(DomainError(α[3],"Source must be below the free surface at ζ=0"))
    # Froude number scaled distances from the source's image
    x,y,z = (ξ-α .* [1,1,-1])/Fn^2;
    # Near-field image and non-oscillatory wave
    N = 1/hypot(x,y,z)+2/π*quadgl_inf(T->Ni(x,y,z,T);kwargs...)
    # Far-field oscillatory wave
    b = √max(-5log(10)/z-1,0); a = max(x/abs(y),-b) # integration limits
    W = ifelse(a≥b || x==y==0, 0., 4*quadgl_ab(T->Wi(x,y,z,T),a,b;kwargs...))
    # Total Green function
    return source(ξ,α)+(N+W)/Fn^2
end
Ni(x,y,z,T) = real(expintx(complex((1+T^2)*z,eps(T)+(x+y*T)*hypot(1,T))))
Wi(x,y,z,T) = exp((1+T^2)*z)*sin((x-abs(y)*T)*hypot(1,T))

# Surface elevation ζ/Fn²
ζ(x,y,q,panels;kwargs...) = derivative(x->φ([x,y,0],q,panels;kwargs...),x)
function source_elevation_plot(;Fn=1) # test function 
    ζ(x,y) = derivative(x->kelvin([x,y,0],[0,0,-1];Fn),x) # unit-point source
    x,y = -15:0.1:3,-6:0.1:6; z = ζ.(x',y)
    mn,mx = round.(extrema(z),digits=1); 
    contour(x,y,z/max(-mn,mx),levels=filter(≠(0),-0.9:0.15:0.9),aspect_ratio=:equal,
        title="Point-source wave elevation, Fn=$Fn, ζ/Fn²=[$mn,$mx]",legend=false)
end