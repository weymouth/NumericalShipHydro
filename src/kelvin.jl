len(x) = √(x'*x)
G(x,x₀) = R(x,x₀)+D(x,x₀)+W(x,x₀)

function R(x,x₀) # rankine source/image pair
    x₀⁻ = x₀*SA[1,1,-1] # reflect z-value
    return 1/len(x-x₀⁻)-1/len(x-x₀)
end

function D(x,x₀) # Near-field standing wave 
    #4/π ∫ dθ ⨘ dk exp(kz) cos(kx cosθ) cos(ky sinθ) / (k cos²θ-1) 
    return 0
end

function W(x,x₀) # Far-field traveling wave 
    #4 ∫ dθ exp(z sec²θ) sin(x secθ) cos(y sec²θ sinθ) sec²θ
    return 0
end
