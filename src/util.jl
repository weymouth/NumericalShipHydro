using FastGaussQuadrature
xgl32, wgl32 = gausslegendre(32)
"""
    quadgl(f;wgl=[1,1],xgl=[-1/√3,1/√3])

Approximate ∫f(x)dx from x=[-1,1] using the Gauss-Legendre weights and points `w,x`.
"""
@fastmath quadgl(f;wgl=SA[1,1],xgl=SA[-1/√3,1/√3]) = wgl'*f.(xgl)

"""
    quadgl_inf(f;kwargs...)

Approximate ∫f(x)dx from x=[-∞,∞] using `quadgl` with the change of variable x=t/(1-t^2).
"""
@fastmath quadgl_inf(f;kwargs...) = quadgl(t->f(t/(1-t^2))*(1+t^2)/(1-t^2)^2;kwargs...)

"""
    quadgl_ab(f,a,b;kwargs...)

Approximate ∫f(x)dx from x=[a,b] using `quadgl` with the change of variable x=½(a+b+tb-ta).
"""
@fastmath function quadgl_ab(f,a,b;kwargs...)
    h,j = (b-a)/2,(a+b)/2
    h*quadgl(t->f(j+h*t);kwargs...)
end

using SpecialFunctions
using ForwardDiff: derivative, gradient, value, partials, Dual
# Fix automatic differentiation of expintx(Complex(Dual))
# https://discourse.julialang.org/t/add-forwarddiff-rule-to-allow-complex-arguments/108821
function SpecialFunctions.expintx(ϕ::Complex{<:Dual{Tag}}) where {Tag}
    x, y = reim(ϕ); px, py = partials(x), partials(y)
    z = complex(value(x), value(y)); Ω = expintx(z)
    u, v = reim(Ω); ∂u, ∂v = reim(Ω - inv(z))
    complex(Dual{Tag}(u, ∂u*px - ∂v*py), Dual{Tag}(v, ∂v*px + ∂u*py))
end