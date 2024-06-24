using Test
using CollisionAvoidancePOMDPs

@testset "README usage" begin
    using CollisionAvoidancePOMDPs

    pomdp = CollisionAvoidancePOMDP()
    up = CASBeliefUpdater(pomdp)
    policy = RandomPolicy(pomdp)

    h = simulate(HistoryRecorder(), pomdp, policy, up)

    @test true
end

@testset "README UFK usage" begin
    using CollisionAvoidancePOMDPs

    pomdp = CollisionAvoidancePOMDP()
    up = UKFUpdater(pomdp; λ=1.0)

    ds = initialstate(pomdp)
    b::UKFBelief = initialize_belief(up, ds)
    s = rand(b)
    a = rand(actions(pomdp))
    o = rand(observation(pomdp, a, s))
    b′ = update(up, b, a, o)

    @test true
end

@testset "POMDP" begin
    pomdp = CollisionAvoidancePOMDP()
    up = CASBeliefUpdater(pomdp)
    ds0 = initialstate(pomdp)
    b0 = initialize_belief(up, ds0)
    s0 = rand(b0)
    o0 = rand(observation(pomdp, s0))
    a0 = rand(actions(pomdp))
    b1 = update(up, b0, a0, o0)
    b2 = update(up, b1, a0, o0)
    mean(b0)
    cov(b0)
    @test true
end

@testset "POMDP (utils)" begin
    pomdp = CollisionAvoidancePOMDP()
    idx = 1
    a = actions(pomdp)[idx]
    @test actionindex(pomdp, a) == idx
    s = rand(initialstate(pomdp))
    s32 = convert_s(Vector{Float32}, s, pomdp)
    @test s32 == Float32.(s)
    s64 = convert_s(Vector{Float64}, s32, pomdp)
    @test s64 == Float64.(s32)
end

@testset "Simulation" begin
    pomdp = CollisionAvoidancePOMDP()
    up = CASBeliefUpdater(pomdp)
    policy = RandomPolicy(pomdp)
    ds0 = initialstate(pomdp)
    b0 = initialize_belief(up, ds0)
    s0 = rand(b0)
    h = simulate(HistoryRecorder(), pomdp, policy, up, b0, s0)
    @test true
end


@testset "Simulation (defaults)" begin
    pomdp = CollisionAvoidancePOMDP()
    up = CASBeliefUpdater(pomdp)
    policy = RandomPolicy(pomdp)
    h = simulate(HistoryRecorder(), pomdp, policy, up)
    @test true
end

@testset "NMAC" begin
    pomdp = CollisionAvoidancePOMDP(px=DiscreteNonParametric([0.0], [1.0]))
    up = CASBeliefUpdater(pomdp)
    policy = FunctionPolicy(b->0)
    ds0 = initialstate(pomdp)
    b0 = initialize_belief(up, ds0)
    s0 = [0.0, 0.0, 0, pomdp.τ_max]
    h = simulate(HistoryRecorder(), pomdp, policy, up, b0, s0)
    is_nmac, is_alert, is_reversal = isfailure(pomdp, h[end].s)
    @test is_nmac && !is_alert && !is_reversal
end

@testset "UKF" begin
    pomdp = CollisionAvoidancePOMDP()
    up = CASBeliefUpdater(pomdp)
    ds0 = initialstate(pomdp)
    b0 = initialize_belief(up, ds0)
    rng = Random.default_rng()
    rand(rng, b0; indiv=false)
    @test true
end

@testset "Plotting" begin
    pomdp = CollisionAvoidancePOMDP()
    up = CASBeliefUpdater(pomdp)
    policy = RandomPolicy(pomdp)
    h = simulate(HistoryRecorder(), pomdp, policy, up)
    plot_history(pomdp, h)
    plot_history(pomdp, h; show_obs=true)
    plot_history(pomdp, h; ymin=-350, ymax=350)
    plot_history(pomdp, h; show_actions=false)
    plot_history(pomdp, h; show_actions=true, show_zero_actions=false)
    H = generate_histories(pomdp, policy, up, 2)
    _ = generate_histories(pomdp, policy, up, 2; parallel=true)
    plot_histories(pomdp, H)
    get_actions(h)
    get_h_rel(h)
    get_dh_rel(h)
    get_a_prev(h)
    get_taus(h)
    get_obs_h_rel(h)
    get_belief_mean_h_rel(h)
    get_belief_std_h_rel(h)
    get_rewards(h)
    @test true
end
