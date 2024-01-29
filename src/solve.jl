source(x) = inv(hypot(x...))
xglϕ, wglϕ = gausslegendre(2); quadϕ = zip(xglϕ./2,wglϕ)
ϕ(x,p;G=source) = 0.25*p["dA"]*sum(wu*wv*G(x-p["x"]-u*p["Tu"]-v*p["Tv"]) for (u,wu) in quadϕ for (v,wv) in quadϕ)
∇ϕ(x,p;G=source) = gradient(x->ϕ(x,p;G),x)
∂ₙϕ(pᵢ,pⱼ;G=source) = pᵢ==pⱼ ? 2π : ∇ϕ(pᵢ["x"],pⱼ;G)⋅pᵢ["n"]
Uₙ(pᵢ;U=[1,0,0]) = U⋅pᵢ["n"]
function body_velocity(q,panels;U=[1,0,0],G=source)
    u(p) = U-Uₙ(p;U)*p["n"]+sum(qᵢ*∇ϕ(p["x"],pᵢ;G) for (qᵢ,pᵢ) in zip(q,panels) if p≠pᵢ)
    stack(u.(panels))
end
added_mass(q,panels;G=source) = -sum(panels) do pᵢ
    sum(qⱼ*ϕ(pᵢ["x"],pⱼ;G) for (qⱼ,pⱼ) in zip(q,panels) if pᵢ≠pⱼ)*pᵢ["n"]*pᵢ["dA"]
end