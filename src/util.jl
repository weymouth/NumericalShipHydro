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

using SpecialFunctions
using ForwardDiff: derivative, value, partials, Dual
# Fix automatic differentiation of expintx(Complex(Dual))
# https://discourse.julialang.org/t/add-forwarddiff-rule-to-allow-complex-arguments/108821
function SpecialFunctions.expintx(ϕ::Complex{<:Dual{Tag}}) where {Tag}
    x, y = reim(ϕ); px, py = partials(x), partials(y)
    z = complex(value(x), value(y)); Ω = expintx(z)
    u, v = reim(Ω); ∂u, ∂v = reim(Ω - inv(z))
    complex(Dual{Tag}(u, ∂u*px - ∂v*py), Dual{Tag}(v, ∂v*px + ∂u*py))
end