@with_kw struct CollisionAvoidancePOMDP <: POMDP{Vector{Float64}, Float64, Vector{Float64}}
    h_rel_range::Vector{Float64} = [-100, 100] # relative altitudes [m]
    dh_rel_range::Vector{Float64} = [-10, 10]  # relative vertical rates [m²]
    ddh_max::Float64 = 1.0                     # vertical acceleration limit [m/s²]
    τ_max::Float64 = 40.0                      # max time to closest approach [s]
    collision_threshold::Float64 = 50.0        # collision threshold [m]
    reward_collision::Float64 = -100.0         # reward obtained if collision occurs
    reward_change::Float64 = -1                # reward obtained if action changes
    px = DiscreteNonParametric([2.0, 0.0, -2.0], [0.25, 0.5, 0.25]) # transition noise on relative vertical rate [m/s²]
    σobs = [15, 1, eps(), eps()]               # observation noise [h_rel, dh_rel, a_prev, τ]
    γ = 0.99                                   # discount factor
end

@doc raw"""
## `CollisionAvoidancePOMDP` state space:

- Relative altitude $h_\text{rel}$
- Relative vertical rate $dh_\text{rel}$
- Previous action $a_\text{prev}$
- Time to closest approach $\tau$
"""
function POMDPs.initialstate(pomdp::CollisionAvoidancePOMDP)
    h_rel = Distributions.Uniform(pomdp.h_rel_range...)
    dh_rel = Distributions.Uniform(pomdp.dh_rel_range...)
    a_prev = DiscreteNonParametric([0], [1.0])
    τ = DiscreteNonParametric([pomdp.τ_max], [1.0])
    return product_distribution(h_rel, dh_rel, a_prev, τ)
end

POMDPs.actions(pomdp::CollisionAvoidancePOMDP) = [-5.0, 0.0, 5.0]
POMDPs.actionindex(m::CollisionAvoidancePOMDP, a) = findfirst(actions(m) .== a)
POMDPs.discount(pomdp::CollisionAvoidancePOMDP) = pomdp.γ

function POMDPs.transition(pomdp::CollisionAvoidancePOMDP, s, a)
    h_rel, dh_rel, a_prev, τ = s

    # Update the dynamics
    h_rel = h_rel + dh_rel
    if a != 0.0
        if abs(a - dh_rel) < pomdp.ddh_max
            dh_rel += a - dh_rel
        else
            dh_rel += sign(a - dh_rel)*pomdp.ddh_max
        end
    end
    a_prev = a
    τ = max(τ - 1.0, -1.0)

    T = SparseCat([Float32[h_rel, dh_rel+x, a_prev, τ] for x in pomdp.px.support], pomdp.px.p)
    return T
end

function POMDPs.reward(mdp::CollisionAvoidancePOMDP, s, a)
    h, dh, a_prev, τ = s
    r = 0.0
    if isfailure(mdp, s)
        # We collided
        r += mdp.reward_collision
    end
    if a != a_prev
        # We changed our action
        r += mdp.reward_change
    end
    return r
end

function POMDPs.observation(pomdp::CollisionAvoidancePOMDP, sp)
    h_rel_obs = Normal(sp[1], pomdp.σobs[1])
    dh_rel_obs = Normal(sp[2], pomdp.σobs[2])
    a_prev_obs = DiscreteNonParametric([sp[3]], [1.0])
    τ_obs = DiscreteNonParametric([sp[4]], [1.0])
    return product_distribution(h_rel_obs, dh_rel_obs, a_prev_obs, τ_obs)
end

function POMDPs.gen(pomdp::CollisionAvoidancePOMDP, s, a,
                    rng::AbstractRNG = Random.GLOBAL_RNG)
    sp = rand(rng, transition(pomdp, s, a))
    r = reward(pomdp, s, a)
    o = rand(rng, observation(pomdp, sp))
    return (sp=sp, r=r, o=o)
end

function POMDPs.isterminal(pomdp::CollisionAvoidancePOMDP, s)
    h, dh, a_prev, τ = s
    return τ < 0.0
end

function isfailure(pomdp::CollisionAvoidancePOMDP, s)
    h, dh, a_prev, τ = s
    return abs(h) < pomdp.collision_threshold && abs(τ) < eps()
end

POMDPs.convert_s(::Type{Vector{Float32}}, s::Vector{Float64}, ::CollisionAvoidancePOMDP) = Float32.(s)
POMDPs.convert_s(::Type{Vector{Float64}}, s::Vector{Float32}, ::CollisionAvoidancePOMDP) = Float64.(s)
