using FastGaussQuadrature
xgl, wgl = gausslegendre(101);
"""
    quadgl(f;w=wgl,x=xgl)

Approximate ∫f(x)dx from x=[-1,1] using the Gauss-Legendre weights and points `w,x`.
"""
@fastmath quadgl(f;w=wgl,x=xgl) = sum(wᵢ*f(xᵢ) for (wᵢ,xᵢ) in zip(w,x))
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

using LinearAlgebra: ×,⋅
using TypedTables
"""
    param_props(S,u,v,du,dv) -> (x,n̂,dA,Tᵤ,Tᵥ)

Given a parametric surface function `x=S(u,v)`, return `x`, the unit 
normal `n̂=n/|n|`, the surface area `dA≈|n|`, and the tangent vectors 
`Tᵤ=du*∂x/∂u`,`Tᵥ=dv*∂x/∂v`, where `n≡Tᵤ×Tᵥ`.
"""
function param_props(S,u,v,du,dv)
    Tu,Tv = du*derivative(u->S(u,v),u),dv*derivative(v->S(u,v),v) 
    n = Tu×Tv; mag = hypot(n...)
    (x=S(u,v), n=n/mag, dA=mag, Tu=Tu, Tv=Tv)
end