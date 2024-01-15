using FastGaussQuadrature
Tgl, wgl = gausslegendre(101);

"""
    quadgl_inf(f;w=wgl,T=Tgl)

Approximate the integral ∫f(x)dx from x=[-∞,∞]. Maps the domain 
to t=[-1,1] using the change of variable x=t/(1-t^2) and uses 
the Gauss-Legendre weights and evaluation points `w,T`.
"""
@fastmath function quadgl_inf(f;w=wgl,T=Tgl)
    s = zero(eltype(w))
    @simd for i in eachindex(w,T)
        s += w[i] * f(T[i]/(1-T[i]^2))*(1+T[i]^2)/(1-T[i]^2)^2
    end; s
end
"""
    quadgl_ab(f,a,b;w=wgl,T=Tgl)

Approximate the integral ∫f(x)dx from x=[a,b]. Maps the domain 
to t=[-1,1] using the change of variable x=(a+b+t*(b-a))/2 and 
uses the Gauss-Legendre weights and evaluation points `w,T`.
"""
@fastmath function quadgl_ab(f,a,b;w=wgl,T=Tgl)
    s = zero(eltype(w))
    h,j = (b-a)/2,(a+b)/2
    @simd for i in eachindex(w,T)
        s += w[i] * f(j+h*T[i])
    end; h*s
end

# using QuadGK,BenchmarkTools
# D2(x,y,z;Tmax=Inf) = quadgk_count(T->Di(x,y,z,T),-Tmax,Tmax)
# @btime D2(0.1,0.1,-0.01)
# @btime D2(1.,1.,-1.)
# @btime D2(-10.,3.,-1.)
# @btime D(0.1,0.1,-0.01)
# @btime D(1.,1.,-1.,$w,$T)
# @btime D(-10.,3.,-1.,$w,$T)
# plot(T,x->Di(0.1,0.1,-0.01,x/(1-x^2))*(1+x^2)/(1-x^2)^2)

# W2(x,y,z) = quadgk_count(T->Wi(x,y,z,T),x/abs(y),Inf)
# @btime W2(0.1,0.1,-0.1)
# @btime W2(1.,1.,-1.)
# @btime W2(-10.,3.,-1.)
# @btime W(0.1,0.1,-0.1,$w,$T)
# @btime W(1.,1.,-1.,$w,$T)
# @btime W(-10.,3.,-1.,$w,$T)