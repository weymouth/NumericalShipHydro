G(x,a) = inv(hypot(x-a...))
∇G(x,a,G) = gradient(x->G(x,a),x)
uₙ(pᵢ,pⱼ,G) = pᵢ["x"]==pⱼ["x"] ? 0.5π : ∇G(pᵢ["x"],pⱼ["x"],G)⋅pᵢ["n"]*pⱼ["dA"]
Uₙ(pᵢ;U=[1,0,0]) = U⋅pᵢ["n"]
u(x,q,panels,G;U=[1,0,0]) = U+sum(qᵢ*∇G(x,pᵢ["x"],G)*pᵢ["dA"] for (qᵢ,pᵢ) in zip(q,panels))