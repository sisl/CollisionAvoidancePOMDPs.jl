# CollisionAvoidancePOMDPs.jl

[![Build Status](https://github.com/sisl/CollisionAvoidancePOMDPs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/CollisionAvoidancePOMDPs.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/sisl/CollisionAvoidancePOMDPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/CollisionAvoidancePOMDPs.jl)


A simple aircraft collision avoidance POMDP in Julia (part of [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl)).

Included is an implementation of the unscented Kalman filter for belief updating.

<p align="center">
    <img src="./img/cas.svg">
</p>

```julia
@with_kw struct CollisionAvoidancePOMDP <: POMDP{Vector{Float64}, Float64, Vector{Float64}}
    h_rel_range::Vector{Float64} = [-100, 100] # relative altitudes [m]
    dh_rel_range::Vector{Float64} = [-10, 10]  # relative vertical rates [m²]
    ddh_max::Float64 = 1.0                     # vertical acceleration limit [m/s²]
    τ_max::Float64 = 40.0                      # max time to closest approach [s]
    collision_threshold::Float64 = 50.0        # collision threshold [m]
    reward_collision::Float64 = -100.0         # reward obtained if collision occurs
    reward_change::Float64 = -1                # reward obtained if action changes
    px = DiscreteNonParametric([2.0, 0.0, -2.0], [0.25, 0.5, 0.25]) # transition noise on relative vertical rate [m/s²]
    σobs = [15, 1, eps(), 5]                   # observation noise [h_rel, dh_rel, a_prev, τ]
    γ = 0.99                                   # discount factor
end
```

## Installation
```julia
] add https://github.com/sisl/CollisionAvoidancePOMDP.jl
```

## Usage
```julia
using CollisionAvoidancePOMDPs

pomdp = CollisionAvoidancePOMDP()
up = CASBeliefUpdater(pomdp)
policy = RandomPolicy(pomdp)

h = simulate(HistoryRecorder(), pomdp, policy, up)
```
