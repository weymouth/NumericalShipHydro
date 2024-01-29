source(x) = inv(hypot(x...))
xglϕ, _ = gausslegendre(2); xglϕ ./=2
ϕ(x,p;G=source) = 0.25*p.dA*sum(G(x-p.x-u*p.Tu-v*p.Tv) for u in xglϕ for v in xglϕ)
∇ϕ(x,p;G=source) = x==p.x ? 2π*p.n : -gradient(x->ϕ(x,p;G),x)
∂ₙϕ(pᵢ,pⱼ;G=source) = ∇ϕ(pᵢ.x,pⱼ;G) ⋅ pᵢ.n
Uₙ(pᵢ;U=[1,0,0]) = U ⋅ pᵢ.n
body_velocity(q,panels;U=[1,0,0],G=source) = map(panels.x) do x
    U+sum(qᵢ*∇ϕ(x,pᵢ;G) for (qᵢ,pᵢ) in zip(q,panels))
end |> stack
added_mass(q,panels;G=source) = -sum(panels) do pᵢ
    ϕᵢ = sum(qⱼ*ϕ(pᵢ.x,pⱼ;G) for (qⱼ,pⱼ) in zip(q,panels)) 
    ϕᵢ*pᵢ.n*pᵢ.dA
end