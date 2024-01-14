using FastGaussQuadrature,SpecialFunctions
# using QuadGK,BenchmarkTools
Tgl65, wgl65 = gausslegendre(65);

# Local non-oscillatory wave
Di(x,y,z,T) = real(expintx(complex((1+T^2)*z,eps(T)+(x+y*T)*hypot(1,T))))
@fastmath function D(x,y,z;w=wgl65,T=Tgl65)
    s = zero(eltype(w))
    @simd for i in eachindex(w,T)
        s += w[i] * Di(x,y,z,T[i]/(1-T[i]^2))*(1+T[i]^2)/(1-T[i]^2)^2
    end; 2s/π
end
# D2(x,y,z;Tmax=Inf) = quadgk_count(T->Di(x,y,z,T),-Tmax,Tmax)
# @btime D2(0.1,0.1,-0.01)
# @btime D2(1.,1.,-1.)
# @btime D2(-10.,3.,-1.)
# @btime D(0.1,0.1,-0.01)
# @btime D(1.,1.,-1.,$w,$T)
# @btime D(-10.,3.,-1.,$w,$T)
# plot(T,x->Di(0.1,0.1,-0.01,x/(1-x^2))*(1+x^2)/(1-x^2)^2)

# Far-field oscillatory wave
Wi(x,y,z,T) = exp((1+T^2)*z)*sin((x-abs(y)*T)*hypot(1,T))
@fastmath function W(x,y,z;w=wgl65,T=Tgl65)
    s = zero(eltype(w))
    a,b = x/abs(y),√(-5log(10)/z-1); a = max(a,-b)
    (a≥b || x==y==0) && return s
    h,j = (b-a)/2,(a+b)/2
    @simd for i in eachindex(w,T)
        s += w[i] * Wi(x,y,z,j+h*T[i])
    end; 4h*s
end

# W2(x,y,z) = quadgk_count(T->Wi(x,y,z,T),x/abs(y),Inf)
# @btime W2(0.1,0.1,-0.1)
# @btime W2(1.,1.,-1.)
# @btime W2(-10.,3.,-1.)
# @btime W(0.1,0.1,-0.1,$w,$T)
# @btime W(1.,1.,-1.,$w,$T)
# @btime W(-10.,3.,-1.,$w,$T)