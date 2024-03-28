using Plots
using NeumannKelvin
function spheroid(h;Z=-0.5,L=1,r=0.25,AR=2)
    S(θ₁,θ₂) = SA[0.5L*cos(θ₁),r*cos(θ₂)*sin(θ₁),r*sin(θ₂)*sin(θ₁)+Z]
    dθ₁ = π/round(π*0.5L/√AR/h) # cosine sampling increases density at ends 
    mapreduce(vcat,0.5dθ₁:dθ₁:π) do θ₁
        dx = dθ₁*hypot(r*cos(θ₁),0.5L*sin(θ₁))
        dθ₂ = π/round(π*r*sin(θ₁)*AR/dx) # polar step size at this azimuth
        param_props.(S,θ₁,0.5dθ₂:dθ₂:2π,dθ₁,dθ₂)
    end |> Table
end
panels = spheroid(0.06;Z=-1/8,r=1/12)
CwFn = map(0.15:0.05:1) do Fn
	(Fn=Fn,Cw=solve_drag(panels;G=kelvin,Fn,d²=0))
end |> Table;
scatter(CwFn.Fn,CwFn.Cw,ylims=(0,1e-2),title="submerged spheroid wavemaking drag",
    label=nothing,xlabel="U/√gL",ylabel="Fₓ/½ρU²L²")
savefig("submerged.png")

function prism(h;q=0.2,Z=1,r=1.2)
    S(θ,z) = 0.5SA[cos(θ),q*sin(θ),z]
    dθ = π/round(π*0.5/h) # cosine sampling
    mapreduce(vcat,0.5dθ:dθ:2π) do θ
        dx = dθ*hypot(q*cos(θ),sin(θ))
        i = round(log(1+2Z/dx*(r-1))/log(r)) # geometric growth
        mapreduce(vcat,1:i) do j
            z,dz = -dx*(1-r^j)/(1-r),dx*r^j
            param_props.(S,θ,z+0.5dz,dθ,dz)
        end
    end |> Table
end
panels = prism(0.1)
x,y,z = eachrow(stack(panels.x))
scatter(x,z,size=(600,600),aspect_ratio=:equal)
CwFn2 = map(0:0.025:1) do i
    Fn = 0.2+0.04*(11^i-1)
	(Fn=Fn,Cw=solve_drag(panels;G=kelvin,Fn,d²=0))
end |> Table;
scatter(CwFn2.Fn,CwFn2.Cw,ylims=(0,0.11),title="elliptical prism wavemaking drag",
    label=nothing,xlabel="U/√gL",ylabel="Fₓ/½ρU²L²")
savefig("elliptical.png")