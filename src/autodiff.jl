using SpecialFunctions
using ForwardDiff: derivative, value, partials, Dual
# Fix automatic differentiation of Complex(Dual) 
# https://discourse.julialang.org/t/add-forwarddiff-rule-to-allow-complex-arguments/108821
function SpecialFunctions.expintx(ϕ::Complex{<:Dual{Tag}}) where {Tag}
    # Split input into real and imaginary parts
    x, y = reim(ϕ)
    # This gives the 'finite' complex part of the input
    z = complex(value(x), value(y))

    # Calculate the finite part of the output
    Ω = expintx(z)
    # split into real and imaginary parts
    u, v = reim(Ω)

    # Ω - inv(z) is the value of expintx'(z), so we'll split that into real and imaginary parts too
    ∂u, ∂v = reim(Ω - inv(z))

    # Now lets deal with the infinitesimals from the real and imaginary parts of ϕ
    px, py = partials(x), partials(y)
    # We have something of the form (∂u + i ∂v) (px + i py)
    # Split again into real and imaginary parts
    du = Dual{Tag}(u, ∂u*px - ∂v*py)
    dv = Dual{Tag}(v, ∂v*px + ∂u*py)

    # And combine
    complex(du, dv)
end

# using Zygote
# f(x) = real(expintx(complex(-1,x)))
# df_fwd(x) = derivative(f,x)
# df_zygote(x) = gradient(f,x)[1]

# x = -10:0.01:1
# @assert df_fwd.(x) ≈ df_zygote.(x)
