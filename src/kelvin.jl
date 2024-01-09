function R(x,x₀) # rankine source/image pair
    x₀⁻ = x₀*SA[1,1,-1] # reflect z-value
    return 1/len(x-x₀⁻)-1/len(x-x₀)
end
len(x) = √(x'*x)

using QuadGK,Plots,SpecialFunctions
D(x,y,z) = quadgk_count(T->Di(x,y,z,T),-Inf,Inf,rtol=1e-5)
Di(x,y,z,T) = 2/π*real(SpecialFunctions.expintx(χ(x,y,z,T)))
χ(x,y,z,T) = complex((1+T^2)*z,(x+y*T)*hypot(1,T))
W(x,y,z) = quadgk_count(T->Wi(x,y,z,T),x/abs(y),√(-5log(10)/z-1),rtol=1e-5)
Wi(x,y,z,T) = 4exp((1+T^2)*z)*sin((x-abs(y)*T)*hypot(1,T))

D(-0.1,0.2,-0.01)
W(-0.1,0.2,-0.01)