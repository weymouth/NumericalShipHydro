using LinearAlgebra: ×,⋅
using TypedTables
include("util.jl")
source(x) = -inv(hypot(x...))
"""
    param_props(S,ξ₁,ξ₂,dξ₁,dξ₂) -> (x,n̂,dA,T₁,T₂)

Given a parametric surface function `x=S(ξ₁,ξ₂)`, return `x`, the unit 
normal `n̂=n/|n|`, the surface area `dA≈|n|`, and the tangent vectors 
`T₁=dξ₁*∂x/∂ξ₁`,`T₂=dξ₂*∂x/∂ξ₂`, where `n≡T₁×T₂`.
"""
function param_props(S,ξ₁,ξ₂,dξ₁,dξ₂)
    T₁,T₂ = dξ₁*derivative(ξ₁->S(ξ₁,ξ₂),ξ₁),dξ₂*derivative(ξ₂->S(ξ₁,ξ₂),ξ₂) 
    n = T₁×T₂; mag = hypot(n...)
    (x=S(ξ₁,ξ₂), n=n/mag, dA=mag, T₁=T₁, T₂=T₂)
end

ξgl,ωgl = gausslegendre(2)./2 # use an even power to avoid ξ=0
quadξ(f;x=ξgl,w=ωgl) = quadgl(f;x,w) # integrate over ξ=[-0.5,0.5]
"""
    ϕ(x,p;G=source)
    ∇ϕ(x,p;G=source)

Approximate influence `ϕ(x) = ∫ₚ G(x-x')ds'` over panel `p`'s and it's gradient
`∇ϕ` using automatic-differentiation, except at `x==p.x`, where `∇ϕ=2πn̂`.
"""
@fastmath function ϕ(x,p;G=source)
    sum(abs2,x-p.x)>9p.dA && return p.dA*G(x-p.x) # single-point quadrature
    p.dA*quadξ(ξ₁->quadξ(ξ₂->G(x-p.x-ξ₁*p.T₁-ξ₂*p.T₂)))
end
∇ϕ(x,p;G=source) = x==p.x ? 2π*p.n : gradient(x->ϕ(x,p;G),x)
""" 
    ∂ₙϕ(pᵢ,pⱼ;G=source) = A

Normal derivative of the influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;G=source) = ∇ϕ(pᵢ.x,pⱼ;G) ⋅ pᵢ.n
Uₙ(pᵢ;U=[1,0,0]) = U ⋅ pᵢ.n
"""
    φ(x,q,panels;G=source)
    ∇φ(x,q,panels;G=source)

The velocity potential `φ(x) = ∫_S q(x')G(x-x')ds' = ∑ᵢqᵢϕ(x,pᵢ)` induced by an
array of `panels` with strength `q`, and its gradient `∇φ(x) = ∑ᵢqᵢ∇ϕ(x,pᵢ)`.
"""
φ(x,q,panels;G=source) = sum(qᵢ*ϕ(x,pᵢ;G) for (qᵢ,pᵢ) in zip(q,panels)) 
∇φ(x,q,panels;G=source) = sum(qᵢ*∇ϕ(x,pᵢ;G) for (qᵢ,pᵢ) in zip(q,panels))
body_velocity(q,panels;U=[1,0,0],G=source) = map(x->U+∇φ(x,q,panels;G),panels.x) |> stack
added_mass(q,panels;G=source) = sum(p->φ(p.x,q,panels;G)*p.n*p.dA,panels)