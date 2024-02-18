### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 83af53e0-cd99-11ee-2742-1d664ed611f4
begin ## Get the NumericalShipHydro package before it's been registered!
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
	Pkg.add("PlutoUI")
	Pkg.add("QuadGK")
	using PlutoUI,QuadGK
    using NumericalShipHydro
end

# ╔═╡ 6ece81d2-6950-41a9-a7bd-f632dfe7f079
""" source(x,a) 

Green function `G(x)` for a source at position `a`.
"""
source(x,a) = -1/hypot(x-a...)

# ╔═╡ 439c9fa7-70e6-4ff6-b460-129c6d4826ae
md"""x $(@bind x Slider(-5:0.5:2,default=-1,show_value=true))"""

# ╔═╡ b85c817a-e4e9-4d36-9a95-96092e7b7585
md"""y $(@bind y Slider([1/8,1/4,1/2,1,2],default=1/2,show_value=true))"""

# ╔═╡ 1a07c4f9-bd96-4bbd-be33-0a74aadea4e1
md"""z $(@bind z Slider([-10,-1,-0.1,-0.01,-0.001],default=-0.1,show_value=true))"""

# ╔═╡ 219b0b75-2f1e-4c12-af88-b1f6b28e002d
md"""n $(@bind n Slider([8,16,32,64,128],default=32,show_value=true))"""

# ╔═╡ 58f96f50-045c-4549-90c5-5d594d7218f4
begin
	using FastGaussQuadrature
	using SpecialFunctions
	xgl, wgl = gausslegendre(n)
	Ni(x,y,z,T) = real(expintx(complex((1+T^2)*z,eps(T)+(x+y*T)*hypot(1,T))))
	Wi(x,y,z,T) = exp((1+T^2)*z)*sin((x-abs(y)*T)*hypot(1,T))
	dψ(x,y,T) = (x*T-abs(y)*(2T^2+1))^4/(1+T^2)^2
	dψπ(x,y) = dψ(x,y,2π)
	Wi2(x,y,z,T) = exp((1+T^2)*z-dψ(x,y,T)/dψπ(x,y))*sin((x-abs(y)*T)*hypot(1,T))
end

# ╔═╡ 3f411d7d-f936-4885-978c-4f4a0027374d
"""
    kelvin(ξ,α;Fn=1,ltol=-3log(10),xgl=xgl32,wgl=wgl32)

Green Function `G(ξ)` for a source at position `α` moving with `Fn≡U/√gL` below 
the free-surface. The free surface is ζ=0, the coordinates are scaled by L and
the apparent velocity direction is Û=[-1,0,0]. See Noblesse 1981 for details.
Smaller log-tolerance `ltol` will only reduce errors when using a large number of
Gauss-Legendre points.
"""
function kelvin(ξ,α;Fn=1,ltol=-3log(10),xgl=xgl32,wgl=wgl32)
    α[3] ≥ 0 && throw(DomainError(α[3],"Source must be below the free surface at ζ=0"))

    # Froude number scaled distances from the source's image
    x,y,z = (ξ-α .* SA[1,1,-1])/Fn^2

    # Wave-like far-field disturbance
    b = min(-2ltol,√max(ltol/z-1,0)); a = max(x/abs(y),-b) # integration limits
    W = ifelse(a≥b || x==y==0, 0., 4quadgl_ab(T->Wi(x,y,z,T),a,b;xgl,wgl))

    # Near-field disturbance
    T₀ = ifelse(y==0,0,clamp(x/y,-b,b)); S = max(abs(T₀),π) # center & scale
    N = 1/hypot(x,y,z)+2S/π*quadgl_inf(T->Ni(x,y,z,S*T-T₀);xgl,wgl)

    # Total Green function
    return source(ξ,α)+(N+W)/Fn^2
end

# ╔═╡ 6a32d7b0-b49f-4157-92f7-3d487196cf46
begin
	ltol = -3log(10)
	b = √max(ltol/z-1,0)
	a = max(x/abs(y),-b)
	T₀ = clamp(x/y,-b,b)
	S = max(abs(T₀),1)
	using Plots 
	plot(range(a,b,1000),T->Wi(x,y,z,T),label="Wᵢ",ylims=(-1.5,1.5))
	plot!(range(-S-4,S+4,1000),T->Ni(x,y,z,T),label="Nᵢ")
	scatter!([a],[Wi(x,y,z,a)],c=:lightblue,label=nothing)
	plot!(range(a,b,1000),T->Wi2(x,y,z,T),label="Wᵢ2")
end

# ╔═╡ a5c6d1cd-371c-4d07-ad52-c8b54ec6531b
begin
	f(T)=(1+T^2)*z-dψ(x,y,T)/dψπ(x,y)-ltol/2
	root(f,b) = (while abs(f(b))>1e-2
		b -= f(b)/derivative(f,b)
	end; b)
	b1 = root(f,-ltol/2)
	a1 = a
	h,j = (b1-a1)/2,(a1+b1)/2
	plt1 = plot(range(-1,1,1000),T->h*Wi(x,y,z,h*T+j),ylabel="Wᵢ",label=nothing 	 ,title="Rescaled with Gauss points")
	scatter!(plt1,xgl,T->h*Wi(x,y,z,h*T+j),c=:lightblue,label=nothing)
	inf(f,t) = f(t/(1-t^2))*(1+t^2)/(1-t^2)^2
	plt2 = plot(range(-1,1,1000),T->S*inf(T->Ni(x,y,z,S*T-T₀),T),ylabel="Nᵢ",label=nothing,c=:orange)
	scatter!(plt2,xgl,T->S*inf(T->Ni(x,y,z,S*T-T₀),T),c=:orange,label=nothing)
	plot!(plt1,range(-1,1,1000),T->h*Wi2(x,y,z,h*T+j),c=:orchid,label=nothing)
	scatter!(plt1,xgl,T->h*Wi2(x,y,z,h*T+j),c=:orchid,label=nothing)
	plot(plt1,plt2,layout=(2,1))
end

# ╔═╡ b7fc4254-8a61-44f2-99f2-cb184bd54020
Wqgk,_,_ = quadgk_count(T->Wi(x,y,z,T),x/abs(y),Inf,atol=1e-3)

# ╔═╡ 859f9dc9-58d1-4fea-a641-cf5a9dfe70e3
W2qgk,_,_ = quadgk_count(T->Wi2(x,y,z,T),x/abs(y),Inf,atol=1e-3)

# ╔═╡ d0e7703b-a0ce-4b72-a3e1-5e3dc03f0ac4
Nqgk,_,_ = quadgk_count(T->Ni(x,y,z,T),-Inf,x/y,Inf,atol=1e-3)

# ╔═╡ 08e89998-ff24-4fc2-bb3a-c40f86a46a70
begin
	np = 2 .^(1:9)
	Wp = map(np) do n
		xgl, wgl = gausslegendre(n)
		NumericalShipHydro.quadgl_ab(T->Wi(x,y,z,T),a,b;xgl,wgl)
	end
	Wp2 = map(np) do n
		a1 ≥ b1 && return 0
		xgl, wgl = gausslegendre(n)
		NumericalShipHydro.quadgl_ab(T->Wi2(x,y,z,T),a1,b1;xgl,wgl)
	end
	Np = map(np) do n
		xgl, wgl = gausslegendre(n)
		S*NumericalShipHydro.quadgl_inf(T->Ni(x,y,z,S*T-T₀);xgl,wgl)
	end
	plot(np,Wp,label="W",xscale=:log2)
	plot!(np,Np,label="N",xlabel="# Gauss points")
	plot!(np,Wp2,label="W2")
	hline!([Wqgk],ls=:dash,c=:lightblue,label=nothing)
	scatter!([n],[Wp[Int(log2(n))]],c=:lightblue,label=nothing)
	hline!([Nqgk],ls=:dash,c=:orange,label=nothing)
	scatter!([n],[Np[Int(log2(n))]],c=:orange,label=nothing)
end

# ╔═╡ Cell order:
# ╠═83af53e0-cd99-11ee-2742-1d664ed611f4
# ╠═6ece81d2-6950-41a9-a7bd-f632dfe7f079
# ╠═3f411d7d-f936-4885-978c-4f4a0027374d
# ╠═58f96f50-045c-4549-90c5-5d594d7218f4
# ╟─6a32d7b0-b49f-4157-92f7-3d487196cf46
# ╟─439c9fa7-70e6-4ff6-b460-129c6d4826ae
# ╟─b85c817a-e4e9-4d36-9a95-96092e7b7585
# ╟─1a07c4f9-bd96-4bbd-be33-0a74aadea4e1
# ╟─219b0b75-2f1e-4c12-af88-b1f6b28e002d
# ╟─a5c6d1cd-371c-4d07-ad52-c8b54ec6531b
# ╟─08e89998-ff24-4fc2-bb3a-c40f86a46a70
# ╠═b7fc4254-8a61-44f2-99f2-cb184bd54020
# ╠═859f9dc9-58d1-4fea-a641-cf5a9dfe70e3
# ╠═d0e7703b-a0ce-4b72-a3e1-5e3dc03f0ac4
