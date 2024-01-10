using StaticArrays
function G(x,x₀;k=1)
    # Wavenumber-scaled distance from source and image source
    r = k*(x-x₀); r′ = k*(x-x₀.*SA[1,1,-1])
    # Rankine and wave potentials
    return 1/hypot(r...)-1/hypot(r′...)+D(r′...)+W(r′...)
end

using QuadGK,SpecialFunctions
# Local non-oscillatory wave
D(x,y,z) = 2/π*quadgk(T->Di(x,y,z,T),-Inf,Inf,rtol=1e-5)[1]
Di(x,y,z,T) = real(expintx(χ(x,y,z,T)))
χ(x,y,z,T) = complex((1+T^2)*z,(x+y*T)*hypot(1,T))
# Far-field oscillatory wave
W(x,y,z) = 4quadgk(T->Wi(x,y,z,T),x/abs(y),√(-5log(10)/z-1),rtol=1e-5)[1]
Wi(x,y,z,T) = exp((1+T^2)*z)*sin((x-abs(y)*T)*hypot(1,T))

G(SA[0.1,0.2,0],SA[0,0,-1])