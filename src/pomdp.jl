@with_kw struct CollisionAvoidancePOMDP <: POMDP{Vector{Float64}, Float64, Vector{Float64}}
    h_rel_range::Vector{Real} = [-10, 10] # initial relative altitudes [m]
    dh_rel_range::Vector{Real} = [-1, 1]  # initial relative vertical rates [m²]
    ddh_max::Real = 1.0                   # vertical acceleration limit [m/s²]
    τ_max::Real = 40                      # max time to closest approach [s]
    actions::Vector{Real} = [-5, 0.0, 5]  # relative vertical rate actions [m/s²]
    collision_threshold::Real = 50        # collision threshold [m]
    reward_collision::Real = -100         # reward obtained if collision occurs
    reward_reversal::Real = -2            # reward obtained if action reverses direction (e.g., from +5 to -5)
    reward_alert::Real = -4               # reward obtained if alerted (i.e., non-zero vertical rates)
    px = DiscreteNonParametric([0.1, 0.0, -0.1], [0.25, 0.5, 0.25]) # transition noise on relative vertical rate [m/s²]
    σobs::Vector{Real} = [15, 1, eps(), eps()] # observation noise [h_rel, dh_rel, a_prev, τ]
    γ::Real = 0.99                        # discount factor
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

POMDPs.actions(pomdp::CollisionAvoidancePOMDP) = pomdp.actions
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

function POMDPs.reward(pomdp::CollisionAvoidancePOMDP, s, a)
    h_rel, dh_rel, a_prev, τ = s
    r = 0.0
    if isfailure(pomdp, s)
        # Collided
        r += pomdp.reward_collision
    end
    if a_prev == 0 && a != 0
        # Alerted
        r += pomdp.reward_alert
    end
    if a_prev != 0 && a != 0 && a != a_prev
        # Reversed the action
        r += pomdp.reward_reversal
    end
    # if abs(τ) < eps()
        # r += -abs(h_rel) # minimize separation
    # end
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
    h_rel, dh_rel, a_prev, τ = s
    return τ < 0.0
end

function isfailure(pomdp::CollisionAvoidancePOMDP, s)
    h_rel, dh_rel, a_prev, τ = s
    return abs(h_rel) < pomdp.collision_threshold && τ < eps()
end

POMDPs.convert_s(::Type{Vector{Float32}}, s::Vector{Float64}, ::CollisionAvoidancePOMDP) = Float32.(s)
POMDPs.convert_s(::Type{Vector{Float64}}, s::Vector{Float32}, ::CollisionAvoidancePOMDP) = Float64.(s)
