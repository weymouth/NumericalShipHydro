### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ b4934dba-da8a-4d46-b933-5e99c94e06dc
begin ## Get the NumericalShipHydro package before it's been registered!
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using NumericalShipHydro
end

# ╔═╡ 89ec7c70-c0cf-4bef-8e32-9dc4b2df91f4
md"""
# Numerical Panel Methods 🚢 

In the first notebook we studied

 1. _How can we calculate the influence of each panel?_ **Gaussian quadratures.**
 2. _How should we compute the derivatives of our functions?_ **Automatic differentiation.**

The 3d versions of the source and Gauss quadrature have been defined in the `NumericalShipHydro` package, imported below.

In this notebook we will actually start solving p-flow problems with panels methods addressing two more questions: 

 3. _How can we determine the correct strength for each panel?_ **Set-up and solve a linear system.**
 4. _How should we determine if a method is working?_ **Convergence and validation tests.** 
"""

# ╔═╡ a899175c-85fd-46ed-bd3a-b30c1e97c9d6
md"""
## Sphere geometry

Our example problem will be the potential flow around a sphere. The `sphere` function below creates a vector of panels using the [parametric equation](https://en.wikipedia.org/wiki/Sphere#Parametric) 

$[S_x,S_y,S_z] = R[\cos(θ₂)\sin(θ₁),\sin(θ₂)\sin(θ₁),\cos(θ₁)]$

where `θ₁,θ₂` are the azimuth and polar angles.
"""

# ╔═╡ 0576ce9d-dd33-4b20-915a-24421037d7c0
"""
    sphere(h;R=1) -> Table(panel_props)

Sample a sphere of radius `R` such that the arc-length ≈ h in each direction.
θ₁=[0,π]: azimuth angle, θ₂=[0,2π]: polar angle.
"""
function sphere(h;R=1)
    S(θ₁,θ₂) = R .* SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    dθ₁ = π/round(π*R/h) # azimuth step size
    mapreduce(vcat,0.5dθ₁:dθ₁:π) do θ₁
        dθ₂ = 2π/round(2π*R*sin(θ₁)/h) # polar step size at this azimuth
        param_props.(S,θ₁,0.5dθ₂:dθ₂:2π,dθ₁,dθ₂)
    end |> Table
end

# ╔═╡ 71df011e-bf21-47c3-85f0-18a9aca54fee
md"""
The function sets the azimuth and polar spacings so each panel's arc length is approximately $h$. Then, it evaluates the function `S(θ₁,θ₂)` at those spacings, filling a vector with all the panel information using the `param_props` function defined in `NumericalShipHydro`. 

Let's set the panel size to `h=0.5R` and create an array of panels.
"""

# ╔═╡ 9bffe4db-31e1-4984-a20b-6ff67f3db7ea
h = 0.5; panels = sphere(h); display(panels)

# ╔═╡ 7d7397ad-8f64-41c8-9742-62921d21be9c
begin 
	using Plots
	# Split the centroids (`panels.x`) into  `x,y,z` vectors for plotting
	x,y,z = eachrow(stack(panels.x));
	N=length(panels)
	plot(x,y,z,legend=false,title="Sphere represented with $N panels")
end

# ╔═╡ 6a2f8a49-6a92-45ac-b538-74f478594033
md"""
Each panel has a few properties incluing the centroid `x`, the normal `n`, and the area `dA`. 

Let's start by plotting the centroids.
"""

# ╔═╡ 44da77a5-8fbc-40f8-a3c5-1006295dd59e
md"""
Next let's verify the panels have the correct total area $4\pi R^2$ and all the panels have an area colse to $h^2$.
"""

# ╔═╡ 75db3a69-1e06-4482-9df0-a77451ceb005
begin
	A_percent = round(100sum(panels.dA)/4π-100,digits=1)
	plot(z,panels.dA/h^2,ylim=(0,2),xlabel="z",ylabel="dA/h²",
	    legend=false, title="Total surface area error: $A_percent%")
end

# ╔═╡ b40064aa-c1b6-46c6-86fc-87661b3d4d27
md"""
#### Activity
 - Discuss why a uniform panel size might be a good idea for this problem.
 - What happens if we use a constant dθ₂ value? Try it, but fix if needed.

Note that the number of panels is approximately $N\approx 4\pi R^2/h^2$. This shows that $N \sim 1/h^2$, which we might worry us as we start thinking about how to solve for the flow on these panels.
"""

# ╔═╡ 5e3f59e0-92a6-486e-8e62-67d282d5821e
md"""
## Superposition 💧+💧+💧=💦

The potential of the full sphere is simply the superposition of each panel's contribution,

$\Phi(x) = \sum_{i=1}^N q_i \varphi_i(x)$

where $N$ is the number of panels, $q$ is the vector of **unknown** panel strengths and $\varphi_i$ is influence of panel $i$. (Annoyingly, the characters ϕ and φ are rendered interchangably. They mean the same thing.)

Our panel potential $\varphi_i$ uses a source as it's Greens function $G=-1/r$. Since $\nabla^2 G=0$, _any_ vector $q$ will satisfy the laplace equation $\nabla^2\Phi=0$ and be a valid potential flow. This is nice since we can't mess that up, but it means we need an additional equation to determine the _correct_ $q$ for a given geometry and flow condition.

> 3. _How can we determine the correct $q$ for each panel?_ 
"""

# ╔═╡ 1719d47e-5861-481a-8398-84c92e58fbb1
md"""
## Apply boundary conditions

The additional equations from our problem description are the boundary conditions. Defining $\vec U$ as the free stream velocity and $\hat n$ as the surface normal, the conditions in an infinite fluid (no free surface) are
 - Flow tangency on the solid body's surface: $U_n+u_n = U_n+\frac{\partial\Phi}{\partial n}=0$
 - No disturbance far from the body: $u(\infty)\rightarrow 0$ 

The second condition is achieved automatically since $u(r) \sim \frac{\partial G}{\partial r} = 1/r^2$. Therefore, the first condition must be used to set $q$.

Substituting the equation for $\phi$ into the body BC, we have

$\vec U \cdot \hat n(\vec S) + \sum_{j=1}^N \frac{\partial\varphi_j}{\partial n}(\vec S) q_j = 0$

This boundary condition is linear in $q$ and applies to every point on the body surface. Applying the boundary condition at $N$ specific locations will create $N$ linear equations of the $N$ unknown components of $q$. We will choose the centroid $\vec c$ of each panel to apply this condition. Defining 

$a_{i,j} = \frac{\partial\varphi_j}{\partial n}(\vec c_i), \quad b_i = -\vec U \cdot \hat n(\vec c_i)$

as the components of the influence matrix $A$ and the excitation vector $b$, we have

$\sum_{j=1}^N a_{i,j} q_j = b_i \quad i=1\ldots N$

or simply $Aq = b$. Meaning once we construct `A,b` we simple use `q = A\b`.
"""

# ╔═╡ e5ce94d5-5fdc-44d3-9009-6455624e9244
begin
	# dot product of U and `pᵢ.n`
	U = [1,0,0]
	Uₙ(pᵢ) = U ⋅ pᵢ.n
	
	# derivative of ϕⱼ in direction pᵢ.n
	∂ₙϕ(pᵢ,pⱼ) = derivative(t->ϕ(pᵢ.x+t*pᵢ.n,pⱼ),0.)
	
	# Construct A,b
	A,b = ∂ₙϕ.(panels,panels'),-Uₙ.(panels)
	
	# solve & check that it worked
	q = A \ b 
	A*q ≈ b && "Solved!"
end

# ╔═╡ 6d5d2289-15df-47ba-82ff-2d5bf18627de
md"""
The code above uses a `ϕ` function defined in `NumericalShipHydro` which integrates the source Greens function using Gauss points, but is otherwise complete.

#### Activity
 - How big is `A`? How does this scale with `h`? Is that a problem?
 - What is that `derivative(t->ϕ(x+t*n))` stuff? Why not use `gradient(ϕ)⋅n`?

## Verification

We got the solution to the linear system we created, but was that the right system? And how accurate is the solution?

> 4. _How should we determine if a method is working?_ 

First, lets **verify** that the influence matrix and excitation vector values are reasonable.
"""

# ╔═╡ e4876085-f4ac-4e33-8230-773e7206b822
begin
	plt1=heatmap(A,yflip=true,colorbar_title="A")
	plt2=heatmap(inv.(A),yflip=true,colorbar_title="inv.(A)")
	plot(plt1,plt2,layout=(1,2),size=(700,270))
end

# ╔═╡ 06a9e3db-ef2f-4236-8ab0-8adbaca743bc
plot(x,b,xlabel="x/R",ylabel="b/U",label=nothing,size=(400,270))

# ╔═╡ f9755b5a-cca1-4f7d-ae53-ffee8caaa9ff
md"""
The matrix is diagonally dominated with a value around 6, and the off diagonals get as small as 1/15. Does that make sense?

 - The diagonal values is the self-induced velocity of the panel in the normal direction. We found earlier that this is exactly $u_n/q = 2\pi$ so the diagonal magnitude looks right.

 - The off diagonals are the influence of one panel on another. The smallest values will be antipodes, at a distance of $2R$. Then $u_n/q \approx (h/2R)^2 = 1/16$, so this matches as well.

The $b$ vector also looks correct, since $U_n = n_x = x/R$ for the sphere as we set $U=[1,0,0]$. 
"""

# ╔═╡ 80b175b6-f2ec-49c7-84f1-9addd5ca7d18
md"""
## Validation and Convergence

Next, lets **validate** the method, comparing the numerical solution to a known exact solution.

The analytic potential flow surface velocity on the sphere is

$u_\alpha = \frac 32 U \sin(\alpha)$

where $\alpha$ is the angle of the surface point with respect to the flow direction.
"""

# ╔═╡ 708e7a92-95d0-43af-9805-047399dc38a3
begin
	# Functions to compute disturbance potential and velocity
	Φ(x,q,panels) = q'*ϕ.(Ref(x),panels)
	∇Φ(x,q,panels) = gradient(x->Φ(x,q,panels),x)

	# Velocity magnitude |u| on panel centroids
	abs_u(q,panels) = map(x->hypot(U+∇Φ(x,q,panels)...),panels.x)
	
	# Get the angle and velocity
	function alpha_velocity(panels)
		A,b = ∂ₙϕ.(panels,panels'),-Uₙ.(panels)
		q = A \ b
		# cos(α) = x/R = -b
		return acos.(b),abs_u(q,panels)
	end
	
	plot(0:0.01:π,x->1.5sin(x),label="exact",xlabel="α",ylabel="uₐ")
	α,uₐ = alpha_velocity(panels)
	scatter!(α,uₐ,label="$N panels")
end

# ╔═╡ 3fcfafb1-e71e-4965-85bc-e2fdaac46778
md"""
Not bad, but there is some error.

#### Activity: 
 - Discuss: what type of error is this?
 1. System description
 2. Modelling
 3. Truncation
 3. Finite precision
 3. Human
 - Confirm/refute your theory by adding to the code above.
"""

# ╔═╡ a4da9745-2e0b-4933-8b47-a6bb4cdb0268
md"""
## Added mass

We can further quantify the panel method's error by computing a relevant **integrated quantity**. The added mass matrix is a good choice for potential flows, which is defined as 

$m_{i,j} = -\rho\oint_S \tilde\Phi_i n_j da$

where $\rho$ is the fluid desity and $\tilde\Phi_i$ is the scaled potential resulting from unit velocity in direction $i$. The forces due to an acceleration vector $a$ are then $f = Ma$.

The analytic solution for a sphere is $\frac 23 \rho \pi R^3$ on the diagonal and zero for the off diagonals. How does our numerical method perform?
"""

# ╔═╡ 00965cd1-b233-42f7-8a6d-1688175cac1e
begin
	function added_mass(panels;ρ=1)
	    A = ∂ₙϕ.(panels,panels')
	    B = stack(panels.n)' # all three excitations
	    Q = A \ B            # solve for all three q's
	    -ρ*sum(p->Φ(p.x,Q,panels)*p.n'*p.dA,panels)
	end
	added_mass(panels)/(2π/3) # scale by exact solution
end

# ╔═╡ 1495dcb8-104f-41ff-a988-5abdca975248
md"""
Notice the off diagonals are near machine precision zero. Again, the values are close, but still have error.

#### Actitivity:
 - Write a function `sphere_ma_error(h)` which computes the percent error of `tr(M)`.
 - Make a plot of $h/R$ vs the `tr(M)` error. Is the code validated?
 - What valueof $h$ is pragmatic, and how might you generalize this to other geometries?
"""

# ╔═╡ 1aa3a5c4-a4d7-45fe-b514-12bfb60d893b
md"""

## Summary

This notebook develops a 3D potential flow panel method and tests it's predictions. The proceedure had three parts:
 - We used the parametric equation for a sphere to generate panels data.
 - We used the normal velocity boundary condition to develop a set of linear equations for the strength of each panel.
 - We used the solution to measure the velocity on the panels, and the added mass matrix.

We were careful to visually and quantitatively check our intermediate results **at every step along the way**
 - We plotted the geometry and verified that every panel $dA\sim h^2$ and the total $A\approx 4\pi$.
 - We plotted the influence matrix and source vector and checked their values matched expectations for our test case.
 - We validated $|u|$ and $M$ by comparing to the analytic solution for the sphere, and demonstrated the numerical convergence with $h$.

The same three-part proceedure and careful verification and validation will be present for every example we use in this class (and hopefully in your work in the future as well).
"""

# ╔═╡ Cell order:
# ╟─89ec7c70-c0cf-4bef-8e32-9dc4b2df91f4
# ╠═b4934dba-da8a-4d46-b933-5e99c94e06dc
# ╟─a899175c-85fd-46ed-bd3a-b30c1e97c9d6
# ╠═0576ce9d-dd33-4b20-915a-24421037d7c0
# ╟─71df011e-bf21-47c3-85f0-18a9aca54fee
# ╠═9bffe4db-31e1-4984-a20b-6ff67f3db7ea
# ╟─6a2f8a49-6a92-45ac-b538-74f478594033
# ╠═7d7397ad-8f64-41c8-9742-62921d21be9c
# ╟─44da77a5-8fbc-40f8-a3c5-1006295dd59e
# ╠═75db3a69-1e06-4482-9df0-a77451ceb005
# ╟─b40064aa-c1b6-46c6-86fc-87661b3d4d27
# ╟─5e3f59e0-92a6-486e-8e62-67d282d5821e
# ╟─1719d47e-5861-481a-8398-84c92e58fbb1
# ╠═e5ce94d5-5fdc-44d3-9009-6455624e9244
# ╟─6d5d2289-15df-47ba-82ff-2d5bf18627de
# ╟─e4876085-f4ac-4e33-8230-773e7206b822
# ╟─06a9e3db-ef2f-4236-8ab0-8adbaca743bc
# ╟─f9755b5a-cca1-4f7d-ae53-ffee8caaa9ff
# ╟─80b175b6-f2ec-49c7-84f1-9addd5ca7d18
# ╠═708e7a92-95d0-43af-9805-047399dc38a3
# ╟─3fcfafb1-e71e-4965-85bc-e2fdaac46778
# ╟─a4da9745-2e0b-4933-8b47-a6bb4cdb0268
# ╠═00965cd1-b233-42f7-8a6d-1688175cac1e
# ╟─1495dcb8-104f-41ff-a988-5abdca975248
# ╟─1aa3a5c4-a4d7-45fe-b514-12bfb60d893b
