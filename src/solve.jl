source(x) = -inv(hypot(x...))
ξgl,ωgl = gausslegendre(2)./2 # use an even power to avoid ξ₁=ξ₂=0
"""
    ϕ(x,p;G=source)

Approximate influence `ϕ(x) = ∫ G(x-x')dx'` over panel `p` using Gauss-Legendre quadrature.
"""
@fastmath function ϕ(x,p;G=source,ξgl=ξgl,ωgl=ωgl)
    sum(abs2,x-p.x)>9p.dA && return p.dA*G(x-p.x) # single-point quadrature
    p.dA*quadgl(ξ₁->quadgl(ξ₂->G(x-p.x-ξ₁*p.T₁-ξ₂*p.T₂),x=ξgl,w=ωgl);x=ξgl,w=ωgl)
end
"""
    ∇ϕ(x,p;G=source)

Evaluate the gradient of the influence, using ∇ϕ=2πn̂ for the self-influence.
"""
∇ϕ(x,p;G=source) = x==p.x ? 2π*p.n : gradient(x->ϕ(x,p;G),x)
""" 
    ∂ₙϕ(pᵢ,pⱼ;G=source) 

Normal derivative of the influence of panel `pⱼ` on `pᵢ`
"""
∂ₙϕ(pᵢ,pⱼ;G=source) = ∇ϕ(pᵢ.x,pⱼ;G) ⋅ pᵢ.n
Uₙ(pᵢ;U=[1,0,0]) = U ⋅ pᵢ.n
φ(x,q,panels;G=source) = sum(qⱼ*ϕ(x,pⱼ;G) for (qⱼ,pⱼ) in zip(q,panels)) 
∇φ(x,q,panels;G=source) = sum(qᵢ*∇ϕ(x,pᵢ;G) for (qᵢ,pᵢ) in zip(q,panels))
body_velocity(q,panels;U=[1,0,0],G=source) = map(x->U+∇φ(x,q,panels;G),panels.x) |> stack
added_mass(q,panels;G=source) = sum(p->φ(p.x,q,panels;G)*p.n*p.dA,panels)