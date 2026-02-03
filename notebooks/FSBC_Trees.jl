### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 258e231b-49d9-419e-9ad1-31d7317a8be1
using NeumannKelvin, BenchmarkTools

# ╔═╡ a785a460-00da-11f1-3bbd-79048a8f01e3
md"""
# Free surface boundary condition and approximate potentials

Up to this point, we have solved *unbounded* potential flows:

> - Each constant-strength source panel on the body $\cal B$ is defined as
>$\phi(\vec x) = q \int G\left(\vec x,\vec \xi\right) \text{ d}\vec \xi\quad\text{where}\quad G\left(\vec x,\vec a\right) = \frac 1 {|\vec x-\vec a|}$
>so that the total potential is given by
>
>$\Phi(\vec x) = \sum_{i=1}^N q_i \int G\left(\vec x,\vec \xi\right) \text{ d}\vec \xi_i$

> - We apply the Neumann boundary condition
>$\frac{\partial\Phi}{\partial n}(\vec x) = \vec U\cdot \hat n \quad\text{on}\quad \vec x \in \cal B$
>to each panel centroid to form the full $N\times N$ influence matrix and solve for $q$ directly.

When we consider steady flows with a free surface $\cal S$, we also must satisfy the free surface boundary condition (FSBC). The *linear* version of the FSBC is:

$\frac{\partial\Phi}{\partial z}+\ell\frac{\partial^2\Phi}{\partial x^2} = 0 \quad\text{on}\quad z=0$

where $\ell\equiv\frac{U^2}g$ is the Froude length, the length scale of the free surface disturbance.

### Activity
 - How is $\ell$ related to the classic Froude number?
 - For what value of $\ell$ does the FSBC become a Neumann condition? How is this related to double-body flow?

> The FSBC condition is derived and further discussed in the appendix below.

"""

# ╔═╡ 6da5eb37-3f23-4c2d-ad7c-b0ee83d57b85
md"""
## Two paths

We have two choices when it comes to solving for the potential which satisfies the FSBC.

1. Free surface panels
    - Place source panels on ${\cal S}_0 = 0$ (typically $z = 0$) in addition to the panels on $\cal B$.
   - The potential becomes the superposition of both sets of panel influences.
    - Solve for $q$ which simultaneously satisfies the linear FSBC on ${\cal S}_0$ and the Neumann condition on $\cal B$.
    - Optionally, update the surface shape ${\cal S}_i \rightarrow {\cal S}_{i+1}$ using the nonlinear kinematic condition (see the appendix), and solve for $q$ again. Repeat until convergence.

2. Neumann–Kelvin panels
    - Formulate a new Kelvin Green's function $K$ which satisfies the linear FSBC by construction.
    - Place these Kelvin panels on $\cal B$, and apply the Neumann condition to determine $q$.

### Activity
 - Discuss the problems and pitfalls you see with the Free Surface Panel method.
 - Discuss the problems and pitfalls you see with the Neumann-Kelvin Panel method.
 - Which would you try first?

"""

# ╔═╡ 19adafcd-6c7f-49f9-979f-fc33485850b1
md"""
## A free surface panel array experiment

Free surface panel methods seem conceptually easier, but there is an issue: The panel size must be less than $h < \ell/3$ to resolve the waves, and $\ell \ll L$ for surface ships.

Let's get a feel for the size of this problem with a fake free surface simulation. We will create a free surface mesh and add it to a `BodyPanelSystem`.

> This is just an illustration. We haven't implemented the FSBC yet, so this is actually just an inefficient way to compute the double-body potential.
"""

# ╔═╡ f3e1e5cf-f0a5-4e9d-a952-8aa176507492
function fake_FSPanelSystem(ℓ;verbose=false,N_max=3500,kwargs...)
	# submerged sphere with diameter=1
	S(θ₁,θ₂) = 0.5SA[cos(θ₁),-sin(θ₂)*sin(θ₁),cos(θ₂)*sin(θ₁)-2]
	body = panelize(S,0,π,0,2π,hᵤ=ℓ;N_max) # these panels can be a bit bigger

	# free surface for x∈[-2,1], y∈[-1,1]
	ζ₀(x,y) = SA[x,y,0]
	freesurf = panelize(ζ₀,-2,1,-1,1,hᵤ=ℓ/3,flip=true;N_max)

	# concatenate the two arrays and make a BodyPanelSystem
	verbose && @show ℓ,length(body),length(freesurf)
	return BodyPanelSystem([body;freesurf];kwargs...)
end

# ╔═╡ 8c218be2-eaae-4135-9de7-98734314e26b
for ℓ in logrange(1,1/8,4)
	sys = fake_FSPanelSystem(ℓ,verbose=true)
	# solve, and measure cₚ everywhere
	@time cₚ(directsolve!(sys,verbose=false))
end

# ╔═╡ 95f68555-57c6-4152-8722-f2dd057e47a2
md"""
Not good!
 - `ℓ = 1/8` is already using N = 3,620 panels, and requires > 200 MiB.
 - The scaling is terrible. Going from $\ell = 1/4 \rightarrow \ell = 1/8$ requires $4\times N$, $10\times$ the memory, and $22\times$ the time!

And $\ell = 1/8$ isn't even small. It corresponds to $\mathrm{Fn} = $(round(1/√8, digits = 2)). Getting to $\mathrm{Fn} = 0.25$ takes 30 times longer, and beyond that gives an out-of-memory error!

## Barnes-Hut

We will steal an idea from cosmology to speed up the evaluation of $\Phi(x)$ when $N>O(100)$ - a [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) approximation.

![](https://upload.wikimedia.org/wikipedia/commons/7/7a/Spatial_quadtree_structure_of_the_Barnes-Hut_algorithm_in_the_N-body_problem.gif)

Galaxy simulations like the one above need to compute the influence of all $N$ stars on each other — $O(N^2)$ operations — every time step. This is the same scaling we have to evaluate the potential $\Phi$ or its derivatives on every panel.

The main idea of the Barnes-Hut approximation is to lump the influence of stars together when they are further away. The details of every star's position and mass in another galaxy aren't critical, just their net effect. The stars are organized into nested boxes; with nearby stars lumped together into a box, then nearby boxes lumped together, and so on. Using this [tree](https://en.wikipedia.org/wiki/Binary_tree) data structure, the cost of updating all the stars is reduced from $O(N^2)$ to $O(N\log(N))$.

> The Fast Multipole Method, which lumps together the target stars as well as the source stars, achieves $O(N)$ updates. However, this requires more than just a tree data structure.

## PanelTree

I've applied a similar idea to our panel method using a `PanelTree`. Let's try it out on our fake free surface panel system.
"""

# ╔═╡ 80888034-2f98-4b99-bd16-f27d06fef1ca
fake_FSPanelSystem(1/4,wrap=PanelTree)

# ╔═╡ dd3fd66f-1e26-44b9-9e28-928828b65dd4
md"""
Our `body` now says it's in a `PanelTree` with 11 nested grid levels. The $\theta^2$ parameter is the Barnes-Hut cut-off (squared) distance.  If `|x-box|^2 > θ²*box.radius^2`, then the `box` is treated like a single lumped panel. Otherwise, we descend a level down and check the two boxes inside.

Let's compare the timings with and without `PanelTree`:
"""

# ╔═╡ 23ceb366-9c46-4d16-bd88-243ae94095b4
function timeit(ℓ,solve! = directsolve!)
	println("Panel Table")
	sys = fake_FSPanelSystem(ℓ); x = sys.body.x[1] # first panel center
	@time directsolve!(sys,verbose=false)
	@btime Φ($x,$sys) seconds=0.05
	println("  Φ=",round(Φ(x,sys),digits=4))

	for θ² in [9,1]
		println("PanelTree θ²=$θ²")
		sys = fake_FSPanelSystem(ℓ,wrap=PanelTree;θ²)
		@time solve!(sys,verbose=false)
		@btime Φ($x,$sys) seconds=0.05
		println("  Φ=",round(Φ(x,sys),digits=4))
	end
end; timeit(1/4)

# ╔═╡ 28477585-40e7-4496-be0a-4a4661512288
md"""

### Activity
 - How does the speedup and the accuracy of `Φ` vary with `θ²`?
 - If you have a quick computer, check how this changes with `ℓ=1/8`.
 - The directsolve! function doesn't speed up. Why?

## Iterative solvers

To take full advantage of the PanelTree we need to avoid forming the NxN matrix. This is easily done using a matrix-free iterative solver, like [GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method). This type of solver only needs to be able to evaluate the linear system error for a guess `q`. It uses the error to update the guess (very cleverly) until convergence.

For our panel system, this means we only need to write a function to evaluate the Neumann BC on the body, which is simply `derivative(t -> Φ(p.x + t*p.n, sys), 0)`. Done.

The `gmressolve!` function uses this approach. Let's try it out:
"""

# ╔═╡ dcaec56e-406d-4845-bcb3-ddb23af5a820
fake_FSPanelSystem(1/4,wrap=PanelTree) |> gmressolve!

# ╔═╡ 2d7bd51f-b8f2-4a5d-82a8-3459907108e2
md"The `gmressolve!` information shows the number of iterations `niter=4`, the `timer: 156ms` and reports the solution meets the given tolerance, which defaults at 0.1%.

Let's repeat our timing test from above using `ℓ=1/8` and `gmressolve!`:
"

# ╔═╡ 67c6e46d-708a-4ff9-82ca-2d6c0a844a47
timeit(1/8,gmressolve!)

# ╔═╡ 82a31d4e-a423-4a7c-828a-e013c0cd3ee3
md"We see a nice speedup with no further loss of accuracy using the GMRES solver. But the huge advantage is the memory — from 206 MiB down to < 1 MiB. This should let us tackle bigger systems without crashing."

# ╔═╡ 15a414fa-b464-4310-bf1e-4caf11156d4a
for ℓ in logrange(1/8,1/16,3)
	sys = fake_FSPanelSystem(ℓ,verbose=true,wrap=PanelTree,N_max=14000)
	@time cₚ(gmressolve!(sys,verbose=false))
end

# ╔═╡ 1cc0a7fa-d09f-4180-92fd-738d3ef8ba17
md"""

## Free surface panel method

We didn't get around to solving the free surface panel method in this notebook, but we developed all the required tools:
1. The linear FSBC to apply on free surface panels.
2. The `PanelTree` wrapper to accelerate `Φ` evaluation.
3. The `gmressolve!` function, which only needs to evaluate the BC error to determine `q`.

The only missing piece is writing a FSBC function, which we do in the appendix below."""

# ╔═╡ 9079b996-f1b5-4506-a57a-fc4ef0708342
md"""## Appendix A: Free surface boundary condition derivation

Potential flows with a free surface $\cal S$ have an additional constraint applied to that surface, made up of two parts.

1. The kinematic condition: the fluid moves with $\cal S$.

The governing equation is therefore $\frac{\text{d}}{\text{d} t}\cal{S} = 0$. Assuming the surface is steady and doesn't overturn, we have $S(\vec x,t) = \zeta(x,y) - z$. Also assuming $\vec U = [-U,0,0]$, we can write this kinematic equation as

$\vec u\cdot\vec\nabla(\zeta-z) = \left[-U+\frac{\partial\Phi}{\partial x}\right]\frac{\partial\zeta}{\partial x}+\frac{\partial\Phi}{\partial y}\frac{\partial\zeta}{\partial y}-\frac{\partial\Phi}{\partial z} = 0 \quad\text{on}\quad z=\zeta$

Unfortunately, this equation is **nonlinear** (derivatives of the two unknowns $\Phi$ and $\zeta$ are multiplied) and applied on the unknown wave elevation $\zeta$. To simplify matters, we **linearize** this equation by further assuming the disturbance amplitude is small: $|\nabla\Phi|\ll U$ and $|\zeta|\ll\ell$, where $\ell$ is the length scale of the disturbance waves. In this case we have:

$\frac{\partial\Phi}{\partial z} = -U\frac{\partial\zeta}{\partial x} \quad\text{on}\quad z=0$

2. The dynamic condition: $p(\cal{S}) = p_{\text{atm}}$, the pressure on $\cal S$ is constant.

Using Bernoulli's equation and the same assumptions as above gives another **nonlinear** equation

$\frac 12\left[\left(-U+\frac{\partial\Phi}{\partial x}\right)^2+\left(\frac{\partial\Phi}{\partial y}\right)^2+\left(\frac{\partial\Phi}{\partial z}\right)^2\right]+g\zeta = \frac 12 U^2 \quad\text{on}\quad z=\zeta$

which we can again **linearize** by assuming the disturbance is small, to give

$g\zeta = U\frac{\partial\Phi}{\partial x} \quad\text{on}\quad z=0$

Finally, we can substitute the dynamic equation into the kinematic to remove the unknown $\zeta$, giving the linear free surface boundary condition (FSBC)

$\frac{\partial\Phi}{\partial z}+\ell\frac{\partial^2\Phi}{\partial x^2} = 0 \quad\text{on}\quad z=0$

where $\ell\equiv\frac{U^2}g$ is the Froude length, the length scale of the free surface disturbance.

- If the forward speed is zero, then $\ell = 0$ and the FSBC becomes a Neumann condition on $z = 0$. This means that the double-body flow is the low-speed limit.
- For $\ell > 0$ this equation is a steady [wave equation](https://en.wikipedia.org/wiki/Wave_equation) for $\Phi$ on the free surface.

> Critically, **any** superposition of sine waves of any amplitude, phase, and direction satisfies this equation. The only restriction is that the projection of the wavelength in $x$ is $\lambda_x = 2\pi\ell$.

"""

# ╔═╡ 9259011f-eb22-4f6d-92c6-4eb17b2ead33
md"""
## Appendix B: Linear FSBC function

The fact that so many different waves satisfy the linear FSBC means we have to treat the $x$-derivative very carefully to get the solution we want.
 - We don't want any waves to radiate upstream from the disturbance. This would violate basic causality but the FSBC can't tell the difference.
 - Similarly, we don't want any waves to reflect off the back edge of the free surface mesh. We must stop the domain at some point (the PanelTree only helps so much) but this potentially induces a large modelling error.

Unfortunately, these issues always led to solver divergence when I tried to use AutoDiff for the FSBC.

The classic approach to "fix" these problems is to use an upwinded finite-difference method to estimate the second derivative in $x$. Such a scheme introduces a strong downwind information bias and a significant truncation error that damps out small waves, accelerating convergence.

In NeumannKelvin, I use the one-sided second-order estimate for the second derivative. [The resulting function](https://github.com/weymouth/NeumannKelvin.jl/blob/69d08ee6ab1865f9453687ff6ca621d653b5f268/src/FSPanelSystem.jl#L72) is frustratingly complicated, but the pseudocode for the FSBC at panel $i$ is essentially

```b[i] = -derivative(z -> Φ(x[i], y, z), z) - ℓ*(2Φ[i] - 5Φ[i-1] + 4Φ[i-2] - Φ[i-3]) / h^2```

which GMRES drives to 0 iteratively.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
NeumannKelvin = "7f078b06-e5c4-4cf8-bb56-b92882a0ad03"

[compat]
BenchmarkTools = "~1.6.3"
NeumannKelvin = "~0.9.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "a2610e10dc5bd82358949f48b1ff790efd0c4bf2"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AcceleratedKernels]]
deps = ["ArgCheck", "GPUArraysCore", "KernelAbstractions", "Markdown", "UnsafeAtomics"]
git-tree-sha1 = "0de01460ed11e90b42ce666c8ed0265bad59aa6a"
uuid = "6a4ca0a5-0e36-4168-a932-d9be78d558f1"
version = "0.4.3"

    [deps.AcceleratedKernels.extensions]
    AcceleratedKernelsoneAPIExt = "oneAPI"

    [deps.AcceleratedKernels.weakdeps]
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "856ecd7cebb68e5fc87abecd2326ad59f0f911f3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.43"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Combinatorics]]
git-tree-sha1 = "c761b00e7755700f9cdf5b02039939d1359330e1"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.1.0"

[[deps.CommonSolve]]
git-tree-sha1 = "78ea4ddbcf9c241827e7035c3a03e2e456711470"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.6"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataInterpolations]]
deps = ["EnumX", "FindFirstFunctions", "ForwardDiff", "LinearAlgebra", "PrettyTables", "RecipesBase", "Reexport"]
git-tree-sha1 = "db37d8739c369b9e7212f8e61e37611bda6fa2e1"
uuid = "82cc6244-b520-54b8-b5a6-8a565e85f1d0"
version = "8.9.0"

    [deps.DataInterpolations.extensions]
    DataInterpolationsChainRulesCoreExt = "ChainRulesCore"
    DataInterpolationsMakieExt = "Makie"
    DataInterpolationsOptimExt = "Optim"
    DataInterpolationsRegularizationToolsExt = "RegularizationTools"
    DataInterpolationsSparseConnectivityTracerExt = ["SparseConnectivityTracer", "FillArrays"]
    DataInterpolationsSymbolicsExt = "Symbolics"

    [deps.DataInterpolations.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"
    RegularizationTools = "29dad682-9a27-4bc3-9c72-016788665182"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "a55766a9c8f66cf19ffcdbdb1444e249bb4ace33"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.6"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "7bebc8aad6ee6217c78c5ddcf7ed289d65d0263e"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.6"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FastChebInterp]]
deps = ["ChainRulesCore", "FFTW", "StaticArrays"]
git-tree-sha1 = "5b59bdc6f9517bf659ac173dac84c577fe48b0c1"
uuid = "cf66c380-9a80-432c-aff8-4f9c79c0bdde"
version = "1.2.0"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "0044e9f5e49a57e88205e8f30ab73928b05fe5b6"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.1.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FindFirstFunctions]]
deps = ["PrecompileTools"]
git-tree-sha1 = "27b495de668ccea58de6b06d6d13181396598ea0"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "1.8.0"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "afb7c51ac63e40708a3071f80f5e84a752299d4f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.39"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "8ee627fb73ecba0b5254158b04d4745611b404a1"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.8.0"

[[deps.ImplicitBVH]]
deps = ["AcceleratedKernels", "Adapt", "ArgCheck", "Atomix", "DocStringExtensions", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra"]
git-tree-sha1 = "23ab86c4458e35f0e38e5af0095f84990c25b1d8"
uuid = "932a18dc-bb55-4cd5-bdd6-1368ec9cea29"
version = "0.7.0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

    [deps.InverseFunctions.weakdeps]
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b5a371fcd1d989d844a4354127365611ae1e305f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.39"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "125d65fe5042faf078383312dd060adf11d90802"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.10.5"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LinearOperators]]
deps = ["FastClosures", "LinearAlgebra", "Printf", "Requires", "SparseArrays", "TimerOutputs"]
git-tree-sha1 = "db137007d2c4ed948aa5f2518a2b451851ea8bda"
uuid = "5c8ed15e-5a4c-59e4-a42b-c7e8811fb125"
version = "2.11.0"

    [deps.LinearOperators.extensions]
    LinearOperatorsAMDGPUExt = "AMDGPU"
    LinearOperatorsCUDAExt = "CUDA"
    LinearOperatorsChainRulesCoreExt = "ChainRulesCore"
    LinearOperatorsJLArraysExt = "JLArrays"
    LinearOperatorsLDLFactorizationsExt = "LDLFactorizations"
    LinearOperatorsMetalExt = "Metal"

    [deps.LinearOperators.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    LDLFactorizations = "40e66cde-538c-5869-a4ad-c39174c6795b"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NeumannKelvin]]
deps = ["AcceleratedKernels", "DataInterpolations", "FastChebInterp", "FastGaussQuadrature", "ForwardDiff", "HCubature", "ImplicitBVH", "Krylov", "LinearAlgebra", "LinearOperators", "QuadGK", "Reexport", "Roots", "SpecialFunctions", "StaticArrays", "TupleTools", "TypedTables"]
git-tree-sha1 = "72f873533e5575a637422eec5a255d60d2786e0e"
uuid = "7f078b06-e5c4-4cf8-bb56-b92882a0ad03"
version = "0.9.0"

    [deps.NeumannKelvin.extensions]
    NeumannKelvinGeometryBasicsExt = "GeometryBasics"
    NeumannKelvinMakieExt = "Makie"
    NeumannKelvinNURBSExt = "NURBS"

    [deps.NeumannKelvin.weakdeps]
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    NURBS = "dde13934-061e-461b-aa91-2c0fad390a0d"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "c5a07210bd060d6a8491b0ccdee2fa0235fc00bf"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "8a433b1ede5e9be9a7ba5b1cc6698daa8d718f1d"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.10"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "c06d695d51cfb2187e6848e98d6252df9101c588"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eee1b9ad8b29ef0d936e3ec9838c7ec089620308"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.16"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a3c1536470bf8c5e02096ad4853606d7c8f62721"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.2"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "9297459be9e338e546f5c4bedb59b3b5674da7f1"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3748bd928e68c7c346b52125cf41fff0de6937d0"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.29"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.TypedTables]]
deps = ["Adapt", "Dictionaries", "Indexing", "SplitApplyCombine", "Tables", "Unicode"]
git-tree-sha1 = "84fd7dadde577e01eb4323b7e7b9cb51c62c60d4"
uuid = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"
version = "1.4.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─a785a460-00da-11f1-3bbd-79048a8f01e3
# ╟─6da5eb37-3f23-4c2d-ad7c-b0ee83d57b85
# ╟─19adafcd-6c7f-49f9-979f-fc33485850b1
# ╠═258e231b-49d9-419e-9ad1-31d7317a8be1
# ╠═f3e1e5cf-f0a5-4e9d-a952-8aa176507492
# ╠═8c218be2-eaae-4135-9de7-98734314e26b
# ╟─95f68555-57c6-4152-8722-f2dd057e47a2
# ╠═80888034-2f98-4b99-bd16-f27d06fef1ca
# ╟─dd3fd66f-1e26-44b9-9e28-928828b65dd4
# ╠═23ceb366-9c46-4d16-bd88-243ae94095b4
# ╟─28477585-40e7-4496-be0a-4a4661512288
# ╠═dcaec56e-406d-4845-bcb3-ddb23af5a820
# ╟─2d7bd51f-b8f2-4a5d-82a8-3459907108e2
# ╠═67c6e46d-708a-4ff9-82ca-2d6c0a844a47
# ╟─82a31d4e-a423-4a7c-828a-e013c0cd3ee3
# ╠═15a414fa-b464-4310-bf1e-4caf11156d4a
# ╟─1cc0a7fa-d09f-4180-92fd-738d3ef8ba17
# ╟─9079b996-f1b5-4506-a57a-fc4ef0708342
# ╟─9259011f-eb22-4f6d-92c6-4eb17b2ead33
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
