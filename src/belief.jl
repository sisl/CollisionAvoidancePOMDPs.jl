@with_kw mutable struct CASBeliefUpdater <: Updater
    pomdp::POMDP
    up_ukf::UKFUpdater
end

function CASBeliefUpdater(pomdp::POMDP; λ=1)
    up = UKFUpdater(pomdp; λ)
    up.Σₛ = diagm(0=>pomdp.σobs)
    up.Σₒ = diagm(0=>pomdp.σobs)
    return CASBeliefUpdater(pomdp, up)
end

@with_kw mutable struct CASBelief
    ukf::UKFBelief
end

function POMDPs.update(up::CASBeliefUpdater, b::CASBelief, a, o)
    b_ukf′ = update(up.up_ukf, b.ukf, a, o)
    μ′ = b_ukf′.μ
    Σ′ = b_ukf′.Σ

    # discrete and deterministic `a_prev` and `τ`
    for i in [3, 4]
        μ′[i] = trunc(Int, μ′[i])
        Σ′[:,i] .= 0
        Σ′[i,:] .= 0
    end

    return CASBelief(UKFBelief(μ′, Σ′, b.ukf.ϵ))
end

POMDPs.initialize_belief(up::UKFUpdater, ds::CASBelief) = CASBelief(UKFBelief(; μ=mean(ds), Σ=cov(ds)))
function POMDPs.initialize_belief(up::CASBeliefUpdater, ds)
    return CASBelief(initialize_belief(up.up_ukf, ds))
end

Statistics.mean(b::CASBelief) = b.ukf.μ
Statistics.cov(b::CASBelief) = b.ukf.Σ

function Base.rand(rng::AbstractRNG, b::CASBelief; indiv=true)
    μ = b.ukf.μ
    Σ = b.ukf.Σ
    if indiv
        # treat as individual Gaussians
        return [rand(rng, Normal(μ[i], Σ[i,i])) for i in eachindex(μ)]
    else
        # sample directly from the MvNormal
        b′ = rand(rng, b.ukf)
        return b′
    end
end
