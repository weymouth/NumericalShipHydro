using StaticArrays
function G(x,x₀;k=1,kwargs...)
    # Wavenumber-scaled vector from source and image source
    r = k*(x-x₀); r′ = k*(x-x₀.*SA[1,1,-1])
    # Rankine and wave potentials
    return 1/hypot(r...)-1/hypot(r′...)+D(r′...;kwargs...)+W(r′...;kwargs...)
end

using QuadGK,SpecialFunctions
# Local non-oscillatory wave
D(x,y,z;atol=1e-3,Tmax=Inf) = 2/π*quadgk(T->Di(x,y,z,T),-Tmax,Tmax;atol)[1]
Di(x,y,z,T) = real(expintx(χ(x,y,z,T)))
χ(x,y,z,T) = complex((1+T^2)*z,eps(x)+(x+y*T)*hypot(1,T))
# Far-field oscillatory wave
W(x,y,z;atol=1e-3,Tmax=√(log(atol/10)/z-1)) = abs(y)<x*atol ? 0 : 4quadgk(T->Wi(x,y,z,T),x/abs(y),Tmax;atol)[1]
Wi(x,y,z,T) = exp((1+T^2)*z)*sin((x-abs(y)*T)*hypot(1,T))

using FiniteDifferences
# Potential and velocity
ϕ(x;kwargs...) = G(x,SA[0,0,-1];kwargs...) # replace with summation over points
u(x;kwargs...) = grad(central_fdm(2, 1),x->ϕ(x;kwargs...),x)[1]
ζ(x,y;kwargs...) = grad(central_fdm(2, 1),x->ϕ(SA[x,y,0];kwargs...),x)[1]

# Check that errors are quite small 
for x in (SA[0.1,0.1,-0.9],SA[1.,1.,0.],SA[-10.,3.,0.])
    println(√sum(abs2,u(x).-u(x,atol=1e-16,Tmax=Inf))/3)
end

using Plots
contour(-15:0.1:3,-6:0.1:6,ζ,aspect_ratio=:equal,levels=[-2.5,-2,-1.5,-1,-.5,.5,1,1.5],legend=false)