module CollisionAvoidancePOMDPs

using LinearAlgebra
using Images
using Parameters
using Reexport
@reexport using Distributions
@reexport using Plots
@reexport using POMDPs
@reexport using POMDPTools
@reexport using Random
@reexport using Statistics

default(fontfamily="Computer Modern", framestyle=:box)

export
    CollisionAvoidancePOMDP,
    UKFUpdater,
    UKFBelief,
    CASBeliefUpdater,
    CASBelief,
    isfailure,
    plot_history,
    get_actions,
    get_h_rel,
    get_dh_rel,
    get_a_prev,
    get_taus,
    get_obs_h_rel,
    get_belief_mean_h_rel,
    get_belief_std_h_rel,
    get_rewards

include("pomdp.jl")
include("ukf.jl")
include("belief.jl")
include("plotting.jl")

end # module CollisionAvoidancePOMDPs
