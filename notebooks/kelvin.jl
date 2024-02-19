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
	Pkg.add("ShortCodes")
	using PlutoUI,QuadGK,Printf
    using NumericalShipHydro
end

# ╔═╡ 4367124b-ac09-4c79-92cc-4db328d0af76
begin
	using ShortCodes
	YouTube("-UgQEHHXTRM")
end

# ╔═╡ ebe67ec1-6b3c-4656-9614-840d17d457f8
md"""
# Kelvin source integrals 

The near-field and wave-like integrals are

$N = \frac 1{|\vec x|}+\frac 2\pi\int_{-\infty}^\infty N_i(\vec x,T) dT,\quad W = 4 \int_{x/|y|}^\infty W_i(\vec x,T) dT$

where the integrands are

$N_i = \Re[\text{expintx}((1+T^2)z+i (x+yT)\sqrt{1+T^2})]$

and

$W_i = \text{exp}((1+T^2)z)\sin((x-|y|T)\sqrt{1+T^2})$

## Advanced numerical integration

The integrals above are complicated by a number of factors:
 - Infinite integration limits
 - Singulary points in the integrand: $\text{expintx}(0)=\infty$
 - Rapidly oscillating integrands: $\sin(\psi(T))$

where $\psi(T)=(x-|y|T)\sqrt{1+T^2}$ is the wave phase.

When dealing with more advanced integrals such as this your choices range from very general to very customized
 1. Use a black-box quadrature routine
 1. Transform your integrals so you can keep using `quadgl`
 1. Find and implement a technique specifically for your integrals

#### Activity
 - What pros and cons do you anticipate on each level?
"""

# ╔═╡ f4588bb2-8605-42b7-b149-7f9f33044f6b
md"""
In the block above I've defined the integrands and used a Gaussian quadrature package [quadgk.jl](https://juliamath.github.io/QuadGK.jl/stable/) to perform the integrations. 
 - The package is pefectly happy doing all the work for us and even provides an error estimate. 
 - The number of evaluations is quite high, and can get much much worse depending on the values of `x,y,z`. 

Take a look at the plot below and play with the `x,y,z` sliders to see why.
"""

# ╔═╡ b4cde787-2d4c-4152-9972-38ce9945367b
begin
	magic = @bind damp CheckBox(default=false)
	md"""magical fix? $magic"""
end

# ╔═╡ a27b8f0a-dc4a-452e-88ed-8fcc0b8a1a9a
begin
	xs = @bind x Slider(-5:0.5:2,default=-1,show_value=true)
	ys = @bind y Slider([1/8,1/4,1/2,1,2],default=1/2,show_value=true)
	zs = @bind z Slider([-1,-0.1,-0.01,-0.001],default=-0.1,show_value=true)
	md"""x $xs ,   y $ys,  z $zs"""
end

# ╔═╡ 58f96f50-045c-4549-90c5-5d594d7218f4
begin
	using SpecialFunctions
	Ni(x,y,z,T) = real(expintx(complex((1+T^2)*z,eps(T)+(x+y*T)*hypot(1,T))))
	Wi(x,y,z,T,damp) = exp(decay(x,y,z,T,damp))*sin(ψ(x,y,T))
	decay(x,y,z,T,damp) = (1+T^2)*z-ifelse(damp,dψ⁴(x,y,T)/dψ⁴(x,y,π),0.)
	ψ(x,y,T) = (x-abs(y)*T)*hypot(1,T)
	dψ⁴(x,y,T) = (x*T-abs(y)*(2T^2+1))^4/(1+T^2)^2

	using Printf: @sprintf
	Wgk = quadgk_count(T->Wi(x,y,z,T,damp),x/abs(y),Inf)
	Ngk = quadgk_count(T->Ni(x,y,z,T),-Inf,x/y,Inf)
	sWgk=@sprintf "∫Wᵢ=%.3f, error≈%.1e, fevals=%i" Wgk...
	sNgk=@sprintf "∫Nᵢ=%.3f, error≈%.1e, fevals=%i" Ngk...
end;

# ╔═╡ 1b84b2ef-468c-4ddc-9a70-653294f74caf
sWgk

# ╔═╡ 1ded3510-5c3c-4d1e-92f4-3a7811c90308
sNgk

# ╔═╡ 6a32d7b0-b49f-4157-92f7-3d487196cf46
begin
	# tolerance
	ltol = -3log(10)

	#integration limits
	a = x/abs(y)
	b = max(√max(ltol/z-1,0),a)

	using Plots
	plot(range(min(-b,a),b,1000),T->Ni(x,y,z,T),label=sNgk,
		color=:orange,xlabel="T",ylims=(-1.5,1.5))
	damp &&	plot!(range(a,b,1000),T->Wi(x,y,z,T,false),
			label=nothing,color=:blue,alpha=0.2)
	plot!(range(a,b,1000),T->Wi(x,y,z,T,damp),label=sWgk,color=:blue)
	scatter!([a],[Wi(x,y,z,a,damp)],c=:blue,label=nothing)
end

# ╔═╡ b5f9c1fc-ddc4-47d8-99dd-6f707bdd8fe7
md"""
The wave-like integrand becomes highly oscillatory when $z\rightarrow 0$. Note that I have cut the plot off for very large values of $T$, but this keeps going! That's why the adaptive routine can require more than 150k(!) function evaluations to sample integrand completely.

This issue led to 100s of papers on special evaluation techniques for this integral, but I've already coded a "magical fix". When you add the magic, you will see:
 - The function decays **much** faster when $z\rightarrow 0$
 - The number of required evaluations can be 100x smaller
 - The integral remains within 1% of the original function

What magic is this?
"""

# ╔═╡ 7898c3e1-6224-4200-9bae-e1d68627c6ab
md"""
## Stationary Phase

The exact integral decays like $\exp((1+T^2)z)$ which eventually goes to zero as $T\rightarrow \pm\infty$ since $z$ is negative. However, this drop off happens slower and slower as $z\rightarrow 0$, meaning the wavesteepness becomes huge and we need tons of samples. 

Despite this, the indefinite *integral* remains bounded as $z\rightarrow 0$. Why? It is because when the phase $\psi$ is rapidly changing, the waves are thin and constant amplitude, meaning their integral is almost perfectly zero. Only the regions where $\psi'\approx 0$ (where the phase is "stationary") make a big contribution to the integral. 

This comes up in wave analysis all the time, and I've linked to a videos below where it is used to derive the group velocity of a wave. This also leads to the [Kelvin wake angle](https://en.wikipedia.org/wiki/Kelvin_wake_pattern) of $\approx 19.47^o$.

My "magic fix" increases the integral's decay proportional to $\psi'^4$. This kills the amplitude when $|\psi'|$ is large, but leaves the regions of stationary phase untouched, mimicing the analytic approach.
"""

# ╔═╡ f4cdf4bc-2bce-4cf8-a8d2-08d4a0659317
md"""
## Gauss quadratures

We don't _really_ need the fancy general quadrature package now that we've fixed the problem with $N_i$. The only thing left is to deal with the indefinite integral ranges, which is helpfully discussed in the packages's [documentation](https://juliamath.github.io/QuadGK.jl/stable/quadgk-examples/). The change of variables

$\int_{-\infty}^\infty f(x) dx = \int_{-1}^1 f\left(\frac t{1-t^2}\right)\frac{1+t^2}{(1-t^2)^2} dt$

is all we need.
"""

# ╔═╡ 219b0b75-2f1e-4c12-af88-b1f6b28e002d
md"""magic fix $magic,    
Gauss points $(@bind n Slider([8,16,32,64,128],default=32,show_value=true))


x $xs ,   y $ys,  z $zs"""

# ╔═╡ a5c6d1cd-371c-4d07-ad52-c8b54ec6531b
begin
	using FastGaussQuadrature
	xgl, wgl = gausslegendre(n)

	T₀ = clamp(x/y,-b,b)
	S = max(abs(T₀),1)

	inf(f,t) = f(t/(1-t^2))*(1+t^2)/(1-t^2)^2
	plt1 = plot(range(-1,1,1000),T->S*inf(T->Ni(x,y,z,S*T-T₀),T),
		ylabel="Nᵢ",label=nothing,c=:orange,title="Rescaled integrands")
	scatter!(plt1,xgl,T->S*inf(T->Ni(x,y,z,S*T-T₀),T),c=:orange,label=nothing)

	function root(f,x₀,ftol=0.1) 
		while f(x₀)<-ftol 
			x₀ -= f(x₀)/derivative(f,x₀)
		end; x₀
	end

	# update limits
	a1 = a>0 ? a : root(T->decay(x,y,z,T,damp)-ltol,a)
	b1 = root(T->decay(x,y,z,T,damp)-ltol, damp ? -1/(z-0.1) : b)
	b1 = max(a1,b1)
	
	h,j = (b1-a1)/2,(a1+b1)/2
	plt2 = plot(range(-1,1,1000),T->h*Wi(x,y,z,h*T+j,damp),ylabel="Wᵢ",
		c=:blue,label=nothing)
	scatter!(plt2,xgl,T->h*Wi(x,y,z,h*T+j,damp),c=:blue,label=nothing)
	plot(plt1,plt2,layout=(2,1))
end

# ╔═╡ 08e89998-ff24-4fc2-bb3a-c40f86a46a70
begin
	np = 2 .^(1:9)
	Wp = map(np) do n
		xgl, wgl = gausslegendre(n)
		NumericalShipHydro.quadgl_ab(T->Wi(x,y,z,T,damp),a1,b1;xgl,wgl)
	end
	Np = map(np) do n
		xgl, wgl = gausslegendre(n)
		S*NumericalShipHydro.quadgl_inf(T->Ni(x,y,z,S*T-T₀);xgl,wgl)
	end
	plot(np,Wp,label="W",xscale=:log2,size=(600,300))
	plot!(np,Np,label="N",xlabel="# Gauss points",ylabel="integral")
	hline!([Wgk[1]],ls=:dash,c=:lightblue,label=nothing)
	scatter!([n],[Wp[Int(log2(n))]],c=:lightblue,label=nothing)
	hline!([Ngk[1]],ls=:dash,c=:orange,label=nothing)
	scatter!([n],[Np[Int(log2(n))]],c=:orange,label=nothing)
end

# ╔═╡ Cell order:
# ╟─83af53e0-cd99-11ee-2742-1d664ed611f4
# ╟─ebe67ec1-6b3c-4656-9614-840d17d457f8
# ╠═58f96f50-045c-4549-90c5-5d594d7218f4
# ╟─1b84b2ef-468c-4ddc-9a70-653294f74caf
# ╟─1ded3510-5c3c-4d1e-92f4-3a7811c90308
# ╟─f4588bb2-8605-42b7-b149-7f9f33044f6b
# ╟─6a32d7b0-b49f-4157-92f7-3d487196cf46
# ╟─b4cde787-2d4c-4152-9972-38ce9945367b
# ╟─a27b8f0a-dc4a-452e-88ed-8fcc0b8a1a9a
# ╟─b5f9c1fc-ddc4-47d8-99dd-6f707bdd8fe7
# ╟─7898c3e1-6224-4200-9bae-e1d68627c6ab
# ╟─4367124b-ac09-4c79-92cc-4db328d0af76
# ╟─f4cdf4bc-2bce-4cf8-a8d2-08d4a0659317
# ╟─a5c6d1cd-371c-4d07-ad52-c8b54ec6531b
# ╟─219b0b75-2f1e-4c12-af88-b1f6b28e002d
# ╟─08e89998-ff24-4fc2-bb3a-c40f86a46a70
