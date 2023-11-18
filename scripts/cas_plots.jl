using ProgressMeter
using ImageFiltering
default(fontfamily="Computer Modern", framestyle=:box)

blur(img, σ) = imfilter(Float64.(img), ImageFiltering.Kernel.gaussian(σ))

coarse = true
h_rel_range = coarse ? (-100:10:100) : (-100:0.5:100)
dh_rel = 0
a_prev = 0
τ_range = coarse ? (0:40) : (0:0.25:40)

ds0 = initialstate(pomdp)
b0 = initialize_belief(up, ds0)
A = actions(pomdp)

@enum PlotTyle ValuePlot PolicyPlot PfailPlot

# plot_type = ValuePlot
plot_type = PolicyPlot
# plot_type = PfailPlot

replan = true
verbose = false
n_runs = 3
use_mean = true
discrete_action_colors = false

policy = online_mode!(solver, policy)

s = rand(ds0)
policy_map = Matrix(undef, length(h_rel_range), length(τ_range))

@showprogress for (i,x) in enumerate(τ_range)
    for (j,y) in enumerate(h_rel_range)
        # s[1] = y
        # s[2] = dh_rel
        # s[3] = a_prev
        # s[4] = x
        # o = rand(observation(pomdp, s))
        b = deepcopy(b0)
        b.ukf.μ[1] = y
        b.ukf.μ[2] = dh_rel # randn()
        b.ukf.μ[3] = a_prev # rand(actions(pomdp))
        b.ukf.μ[4] = x
        # b = update(up, b, a_prev, o)
        if plot_type == ValuePlot
            z = value_lookup(policy.surrogate, b)
        elseif plot_type == PolicyPlot
            if replan
                as = []
                for n in 1:n_runs
                    a = action(policy, b)
                    push!(as, a)
                end
                if use_mean
                    z = mean(as)
                else
                    a = as[argmax(map(a->sum(a .== as), as))]
                    z = A[actionindex(pomdp, a)]
                end
            else
                z = A[argmax(policy_lookup(policy.surrogate, b))]
            end
        elseif plot_type == PfailPlot
            z = pfail_lookup(policy.surrogate, b)
        end
        verbose && @info b.ukf.μ z
        policy_map[j,i] = z
    end
end

if plot_type == ValuePlot
    c = cgrad(:viridis, rev=true)
    kwargs = (c=c,)
    title_str = "value estimate"
elseif plot_type == PolicyPlot
    # c = palette(:pigeon, 3)
    # action_colors = ["#8c1515", :white, "#007662"]
    action_colors = ["#008000", :white, "#0000FF"]
    c = cgrad(action_colors)
    kwargs = (c=c,)
    if discrete_action_colors
        c = palette(c, length(A))
        kwargs = (c=c, level=length(A))
    end
    title_str = "online policy"
elseif plot_type == PfailPlot
    c = cgrad(["#007662", :white, "#8c1515"])
    kwargs = (c=c, ) # clims=(0,1))
    title_str = "failure probability estimate"
end

if plot_type == PolicyPlot
    Plots.heatmap(τ_range, h_rel_range, blur(policy_map, 1.5); xflip=true, label=false, size=(400,250), title=title_str, clims=(minimum(A), maximum(5)), kwargs...)
else
    Plots.heatmap(τ_range, h_rel_range, policy_map; xflip=true, label=false, size=(400,250), title=title_str, kwargs...)
end
Plots.xlabel!(raw"time to closest approach ($\tau$)")
Plots.ylabel!(raw"relative altitude ($h_\mathrm{rel}$)")
