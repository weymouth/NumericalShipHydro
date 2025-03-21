using NURBS,FileIO
using NeumannKelvin

function sample(surf::NURBSsurface,du=0.1,dv=0.1;verbose=false,tangentplane=false) 
    S(u,v) = surf(u,v)|>first
    panels = param_props.(S,0.5du:du:1,(0.5dv:dv:1)',du,dv;tangentplane) |> Table
    !verbose && return panels
    println("area = $(sum(panels.dA))")
    println("extrema(dA) = $(extrema(panels.dA))")
    println("Δu = $(S(1.,0.5)-S(0.,0.5))")
    println("Δv = $(S(0.5,1.)-S(0.5,0.))")
    panels
end
vmap(op,m) = reshape(mapslices(op,m,dims=2),:)
function Base.extrema(surf::NURBSsurface)
    S(u,v) = surf(u,v)|>first
    corners = stack(reshape(S.(0.:1,(0.:1)'),:))
    vmap(minimum,corners),vmap(maximum,corners)
end
function extremas(shapes)
    pnts = mapreduce(x->stack(extrema(x)),hcat,shapes)
    vmap(minimum,pnts),vmap(maximum,pnts)
end
extents(shapes) = (pnts = extremas(shapes); pnts[2]-pnts[1])

# Load and put in place
Patches = load("HMS_QueenMary_bare_hull.stp")[[1,4,13,15]];
scale!(Patches,1/extents(Patches)[1])
ex = extremas(Patches)
translate!(Patches,SA[0.5-ex[2][1],-ex[1][2],-ex[2][3]])

# Set sampling resolutions and get panels
du = [1/32,1/4,1/30,1/2]
dv = [1/4,1,1/2,1]
sample.(Patches,du,dv,verbose=true);
panels = vcat(sample.(Patches,du,dv)...)
added_mass(panels)
