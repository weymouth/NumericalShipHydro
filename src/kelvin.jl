len(x) = √(x'*x)
G(x,x₀) = R(x,x₀)+D(x,x₀)+W(x,x₀)

function R(x,x₀) # rankine source/image pair
    x₀⁻ = x₀*SA[1,1,-1] # reflect z-value
    return 1/len(x-x₀⁻)-1/len(x-x₀)
end

using QuadGK,Plots
D(x,y,z) = 2/π*quadgk(T->Di(x,y,z,T),-Inf,Inf,rtol=1e-8)
Di(x,y,z,T) = 0
W(x,y,z) = quadgk(T->Wi(x,y,z,T),x/abs(y),Inf,rtol=1e-8)
Wi(x,y,z,T) = exp((1+T^2)*z)*sin((x-abs(y)*T)*√(1+T^2))
W(-0.1,0.2,-0.01)