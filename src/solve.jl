source(x) = inv(hypot(x...))
xglϕ, _ = gausslegendre(2); xglϕ ./=2
ϕ(x,p;G=source) = 0.25*p.dA*sum(G(x-p.x-u*p.Tu-v*p.Tv) for u in xglϕ for v in xglϕ)
∇ϕ(x,p;G=source) = x==p.x ? 2π*p.n : -gradient(x->ϕ(x,p;G),x)
∂ₙϕ(pᵢ,pⱼ;G=source) = ∇ϕ(pᵢ.x,pⱼ;G)⋅pᵢ.n
Uₙ(pᵢ;U=[1,0,0]) = U⋅pᵢ.n
function body_velocity(q,panels;U=[1,0,0],G=source)
    u(p) = U+sum(qᵢ*∇ϕ(p.x,pᵢ;G) for (qᵢ,pᵢ) in zip(q,panels))
    stack(u.(panels))
end
added_mass(q,panels;G=source) = -sum(panels) do pᵢ
    ϕᵢ = sum(qⱼ*ϕ(pᵢ.x,pⱼ;G) for (qⱼ,pⱼ) in zip(q,panels)) 
    ϕᵢ*pᵢ.n*pᵢ.dA
end