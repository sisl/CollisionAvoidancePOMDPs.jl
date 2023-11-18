using Revise
using CollisionAvoidancePOMDPs
using LocalApproximationValueIteration
using LocalFunctionApproximation
using GridInterpolations

POMDPs.isterminal(bmdp::GenerativeBeliefMDP, b::CASBelief) = isterminal(bmdp.pomdp, mean(b))
POMDPs.actionindex(bmdp::GenerativeBeliefMDP{CollisionAvoidancePOMDP, CASBeliefUpdater}, a::Real) = actionindex(bmdp.pomdp, a)
POMDPs.convert_s(::Type{Vector{Float64}}, b::CASBelief, bmdp::GenerativeBeliefMDP{CollisionAvoidancePOMDP,CASBeliefUpdater}) = Float64.(mean(b)[[1,4]])
function POMDPs.convert_s(::Type{CASBelief}, v::Any, bmdp::GenerativeBeliefMDP{CollisionAvoidancePOMDP,CASBeliefUpdater})
    ukf = initialize_belief(bmdp.updater, initialstate(bmdp.pomdp)).ukf
    ukf.μ[1] = v[1]
    ukf.μ[4] = v[2]
    return CASBelief(ukf)
end

pomdp = CollisionAvoidancePOMDP()
up = CASBeliefUpdater(pomdp)

min_h = -350
max_h = 350
min_τ = 0
max_τ = pomdp.τ_max

grid = RectangleGrid(range(min_h, stop=max_h, length=100),
                     range(min_τ, stop=max_τ, length=41));

interpolation = LocalGIFunctionApproximator(grid)

lavi_solver = LocalApproximationValueIterationSolver(interpolation,
                                                     max_iterations=3,
                                                        is_mdp_generative=true,
                                                     verbose=true,
                                                     n_generative_samples=10)

bmdp = GenerativeBeliefMDP(pomdp, up)

lavi_policy = solve(lavi_solver, bmdp)


using Plots

b0 = initialize_belief(up, initialstate(pomdp))

begin
    default(fontfamily="Computer Modern", framestyle=:box)

    h_rel_range = -100:2:100
    dh_rel = 0
    a_prev = 0
    τ_range = 0:40

    policy_map = Matrix(undef, length(h_rel_range), length(τ_range))
    for (i,x) in enumerate(τ_range)
        for (j,y) in enumerate(h_rel_range)
            b = deepcopy(b0)
            b.ukf.μ[1] = y
            b.ukf.μ[2] = dh_rel # randn()
            b.ukf.μ[3] = a_prev # rand(actions(pomdp))
            b.ukf.μ[4] = x
            # ai = action(lavi_policy, b)
            ai = value(lavi_policy, b)
            policy_map[j,i] = ai
        end
    end

    Plots.heatmap(τ_range, h_rel_range, policy_map, xflip=true, label=false, c=cgrad(:jet, rev=true))
    Plots.xlabel!(raw"time to closest approach ($\tau$)")
    Plots.ylabel!(raw"relative altitude ($h_\mathrm{rel}$)")
end
