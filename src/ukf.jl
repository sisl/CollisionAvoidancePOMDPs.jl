@doc raw"""
# Unscented Kalman filter 🧼
Derivative free! How clean!

$$\begin{gather}
\mathbf{f}_T \texttt{ (transition)} \tag{transition dynamics function}\\
\mathbf{f}_O \texttt{ (observation)} \tag{observation dynamics function}
\end{gather}$$
"""
@with_kw mutable struct UKFUpdater{P<:POMDP} <: Updater
    pomdp::P
    Σₛ # state covariance matrix
    Σₒ # obs. covariance matrix
    λ  # sigma point spread parameter
end

function UKFUpdater(pomdp::POMDP; λ=1.0)
    s = rand(initialstate(pomdp))
    o = rand(observation(pomdp, s))
    nₛ = length(s)
    nₒ = length(o)
    Σₛ = diagm(0=>ones(nₛ))
    Σₒ = diagm(0=>ones(nₒ))
    return UKFUpdater(; pomdp, Σₛ, Σₒ, λ)
end

@with_kw mutable struct UKFBelief
    μ = missing # mean vector
    Σ = missing # covariance matrix
    ϵ = 1e-6 # added to covariance for numerical stability in sampling
end

@doc raw"""
##### UKF prediction:

Predict where the agent is going based on the nonlinear transition function $\mathbf{f}_T$.

##### UKF update:

1. Update observation model using predicted mean and covariance.
2. Calculate the _cross covariance matrix_ (measures the variance between two multi-dimensional variables; here it's the transition prediction $𝛍_p$ and observation model update $𝛍_o$).
3. Update mean and covariance of our belief.
"""
function POMDPs.update(up::UKFUpdater, b::UKFBelief, a, o)
    μ, Σ, λ = b.μ, b.Σ, up.λ
    w = weights(μ, λ)

    # Predict
    fₜ = s -> rand(transition(up.pomdp, s, a))
    μₚ, Σₚ, _, _ = unscented_transform(μ, Σ, fₜ, λ, w)
    Σₚ += up.Σₛ

    # Update
    fₒ = sp -> rand(observation(up.pomdp, sp))
    (μₒ, Σₒ, Sₒ, Sₒ′) = unscented_transform(μₚ, Σₚ, fₒ, λ, w)
    Σₒ += up.Σₒ

    # Calculate the cross covariance matrix
    Σₚₒ = cross_cov(μₚ, μₒ, w, Sₒ, Sₒ′)

    # Apply Kalman gain belief
    K = Σₚₒ / Σₒ         # Kalman gain
    μ′ = μₚ + K*(o - μₒ) # updated mean
    Σ′ = Σₚ - K*Σₒ*K'    # updated covariance
    return UKFBelief(μ′, Σ′, b.ϵ)
end

@doc raw"""
##### Sigma point samples:

Create a set of sigma point samples as an approximation for $𝛍′$ and $𝚺′$ that will be updated by the UKF (instead of updating the non-linear, multi-variate Gaussian directly). Common sigma points include the mean $𝛍 \in \mathbb{R}^n$ and $2n$ points formed from perturbations of $𝛍$ in directions determined by the covariance matrix $𝚺$:

$$\begin{align}
𝐬_1 &= 𝛍\\
𝐬_{2i} &= 𝛍 + \left(\sqrt{(n+\lambda)𝚺}\right)_i \quad \text{for } i \text{ in } 1\text{:}n\\
𝐬_{2i+1} &= 𝛍 - \left(\sqrt{(n+\lambda)𝚺}\right)_i \quad \text{for } i \text{ in } 1\text{:}n
\end{align}$$
"""
function sigma_points(μ, Σ, λ)
    n = length(μ)
    Δ = sqrt((n + λ) * Σ)
    S = [μ]
    for i in 1:n
        δ = n == 1 ? Δ[i] : Δ[:,i]
        push!(S, μ + δ)
        push!(S, μ - δ)
    end
    return S
end

@doc raw"""
##### Sigma point weights:

The sigma points are associated with the weights:

$$\begin{align}
\lambda &= \text{spread parameter}\\
w_i &= \begin{cases}
\frac{\lambda}{n+\lambda} & \text{for } i=1\\
\frac{1}{2(n+\lambda)} & \text{otherwise}
\end{cases}
\end{align}$$
"""
weights(μ, λ; n=length(μ)) = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]

@doc raw"""
##### Unscented transform:
Reconstruct updated mean and covariance based on a nonlinear transform $\mathbf{f}$ of the sigma points $\mathbf{s}_i$.

##### Reconstruct original mean and covariance:
If we wanted to reconstruct our provided mean and covariance using the generated sigma points $\mathbf{s}_i$, then we can use these equations (note, they don't pass the sigma points through the nonlinear function $\mathbf{f}$ like we do in the unscented transform).

$$\begin{align}
𝛍 &= \sum_i w_i 𝐬_i\\
𝚺 &= \sum_i w_i (𝐬_i - 𝛍)(𝐬_i - 𝛍)^\top
\end{align}$$
"""
function unscented_transform(μ, Σ, f, λ, w)
    S = sigma_points(μ, Σ, λ)
    S′ = f.(S)
    μ′ = sum(w*s for (w,s) in zip(w, S′))
    Σ′ = sum(w*(s - μ′)*(s - μ′)' for (w,s) in zip(w, S′))
    return (μ′, Σ′, S, S′)
end

cross_cov(μₚ, μₒ, w, S, S′) = sum(w*(s - μₚ)*(s′ - μₒ)' for (w,s,s′) in zip(w,S,S′))

function POMDPs.initialize_belief(up::UKFUpdater, ds::Distributions.ProductDistribution)
    μ = [mean.(ds.dists)...]
    n = length(μ)
    Σ = diagm(0=>[std.(ds.dists)...])
    return UKFBelief(; μ, Σ)
end

Base.rand(rng::AbstractRNG, b::UKFBelief) = rand(rng, MvNormal(b.μ, b.Σ + b.ϵ*I))
