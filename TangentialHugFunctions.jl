module TangentialHugFunctions

using LinearAlgebra: norm
using Distributions: logpdf

export Hug
export HugTangential

"""
    Tangential Hug Function.
"""
function HugTangential(x0, T, B, N, α, q, logπ, ∇logπ)
    samples = zeros(N, length(x0))
    acceptances = zeros(N)
    for i in 1:N
        v0s = rand(q)          # Draw velocity spherically
        g   = ∇logπ(x0)        # Compute gradient at x0
        g = g / norm(g)        # Normalize gradient at x0
        v0 = v0s - α*g*(g'v0s) # Tilt velocity
        v, x = v0, x0
        logu = log.(rand())    # Log uniform for acceptance ratio
        δ = T / B              # Stepsize

        for _ in 1:B
            x = x + δ*v/2        # Move
            g = ∇logπ(x)
            ĝ = g / norm(g)
            v = v - 2*(v'ĝ)*ĝ    # Reflect velocity at midpoint
            x = x + δ*v/2        # Move
        end

        # Unsqueeze velocity
        g = ∇logπ(x)
        g = g / norm(g)
        v = v + (α / (1 - α)) * g * (g'v)

        # Acceptance ratio (must use spherical velocity)
        if logu ≦ logπ(x) + logpdf(q, v) - logπ(x0) - logpdf(q, v0s)
            samples[i, :] = x
            acceptances[i] = 1   # Accepted
            x0 = x
        else
            samples[i, :] = x0
            acceptances[i] = 0     # Rejected
        end
    end
    return samples, acceptances
end


"""
    Standard Hug Function.
"""
function Hug(x0, T, B, N, q, logπ, ∇logπ)
    samples = zeros(N, length(x0))
    acceptances = zeros(N)
    for i in 1:N
        # Sample spherical velocity & housekeeping
        v0 = rand(q)
        v, x = v0, x0
        logu = log(rand())
        δ = T / B
        # B iterations of Hug dynamic
        for _ in 1:B
            x = x + δ*v/2
            g = ∇logπ(x)
            ĝ = g / norm(g)
            v = v - 2*(v'ĝ)*ĝ
            x = x + δ*v/2
        end
        # Accept-Reject step
        if logu ≦ logπ(x) + logpdf(q, v) - logπ(x0) - logpdf(q, v0)
            samples[i, :]  = x
            acceptances[i] = 1    # Accepted
        else
            samples[i, :] = x0
            acceptances[i] = 0    # Rejected
        end
    end
    return samples, acceptances
end








end
