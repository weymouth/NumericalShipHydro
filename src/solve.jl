source(x,a) = inv(hypot(x-a...))
∇G(x,a;G=source) = gradient(x->G(x,a),x)
∂ₙG(pᵢ,pⱼ;G=source) = pᵢ==pⱼ ? 2π : ∇G(pᵢ["x"],pⱼ["x"];G)⋅pᵢ["n"]*pⱼ["dA"]
Uₙ(pᵢ;U=[1,0,0]) = U⋅pᵢ["n"]
function body_velocity(q,panels;U=[1,0,0],G=source)
    u(p) = U-Uₙ(p;U)*p["n"]+sum(qᵢ*∇G(p["x"],pᵢ["x"];G)*pᵢ["dA"] for (qᵢ,pᵢ) in zip(q,panels) if p≠pᵢ)
    stack(u.(panels))
end
function added_mass(q,panels;G=source)
    ϕ(x) = sum(qᵢ*G(x,pᵢ["x"])*pᵢ["dA"] for (qᵢ,pᵢ) in zip(q,panels) if x≠pᵢ["x"])
    -sum(ϕ(pᵢ["x"])*pᵢ["n"]*pᵢ["dA"] for pᵢ in panels)
end