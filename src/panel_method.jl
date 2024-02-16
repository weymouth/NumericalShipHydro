"""
    param_props(S,ξ₁,ξ₂,dξ₁,dξ₂) -> (x,n̂,dA,T₁,T₂)

Properties of a parametric surface function `x=S(ξ₁,ξ₂)`. Returns `x`, 
the unit normal `n̂=n/|n|`, the surface area `dA≈|n|`, and the tangent 
vectors `T₁=dξ₁*∂x/∂ξ₁` and `T₂=dξ₂*∂x/∂ξ₂`, where `n≡T₁×T₂`.
"""
function param_props(S,ξ₁,ξ₂,dξ₁,dξ₂)
    T₁,T₂ = dξ₁*derivative(ξ₁->S(ξ₁,ξ₂),ξ₁),dξ₂*derivative(ξ₂->S(ξ₁,ξ₂),ξ₂) 
    n = T₁×T₂; mag = hypot(n...)
    (x=S(ξ₁,ξ₂), n=n/mag, dA=mag, T₁=T₁, T₂=T₂)
end
"""
    ϕ(x,p;G=source,kwargs...)

Approximate potential influence `ϕ(x) ≈ ∫ₚ G(x,x')ds'` of panel `p`. 
The quadrature is improved when `x∼p.x`. The gradient is overloaded with 
the exact value `∇ϕ=2πn̂` when `x=p.x`.
"""
ϕ(x,p;kwargs...) = _ϕ(x,p;kwargs...) # wrapper
@fastmath function _ϕ(x,p;G=source,kwargs...)
    sum(abs2,x-p.x)>9p.dA && return p.dA*G(x,p.x;kwargs...) # single-point quadrature
    p.dA*quadξ(ξ₁->quadξ(ξ₂->G(x,p.x+ξ₁*p.T₁+ξ₂*p.T₂;kwargs...))) # multipoint
end
quadξ(f) = 0.5quadgl(x->f(0.5x)) # integrate over ξ=[-0.5,0.5]

function ϕ(d::AbstractVector{<:Dual{Tag}},p;kwargs...) where Tag
    value(d) ≠ p.x && return _ϕ(d,p;kwargs...) # use ∇ϕ=∇(_ϕ)
    x,Δx = value.(d),stack(partials.(d))
    Dual{Tag}(ϕ(x,p;kwargs...),2π*Δx*p.n...)   # enforce ∇ϕ(x,x)=2πn̂
end
""" 
    ∂ₙϕ(pᵢ,pⱼ;kwargs...) = A

Normal velocity influence of panel `pⱼ` on `pᵢ`.
"""
∂ₙϕ(pᵢ,pⱼ;kwargs...) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ;kwargs...),0.)
Uₙ(pᵢ;U=[1,0,0]) = U ⋅ pᵢ.n
"""
    φ(x,q,panels;kwargs...)

Potential `φ(x) = ∫ₛ q(x')G(x-x')ds' = ∑ᵢqᵢϕ(x,pᵢ)` of `panels` with strengths `q`.
"""
φ(x,q,panels;kwargs...) = sum(qᵢ*ϕ(x,pᵢ;kwargs...) for (qᵢ,pᵢ) in zip(q,panels))
∇φ(x,q,panels;kwargs...) = gradient(x->φ(x,q,panels;kwargs...),x)
body_velocity(q,panels;U=[1,0,0],kwargs...) = map(x->U+∇φ(x,q,panels;kwargs...),panels.x) |> stack
added_mass(q,panels;kwargs...) = sum(p->φ(p.x,q,panels;kwargs...)*p.n*p.dA,panels)
ζ(x,y,q,panels;kwargs...) = derivative(x->φ([x,y,0],q,panels;kwargs...),x)