get_actions(h) = [step.a for step in h]
get_h_rel(h) = [step.s[1] for step in h]
get_dh_rel(h) = [step.s[2] for step in h]
get_a_prev(h) = [step.s[3] for step in h]
get_taus(h) = [step.s[4] for step in h]
get_obs_h_rel(h) = [step.o[1] for step in h]
get_belief_mean_h_rel(h) = [mean(step.b)[1] for step in h]
get_belief_std_h_rel(h) = [cov(step.b)[1,1] for step in h]
get_rewards(h) = [step.r for step in h]

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

function plot_history(pomdp::CollisionAvoidancePOMDP, h::SimHistory, t=length(h);
                      show_actions=true, show_belief=true, show_obs=false,
                      show_zero_actions=true, show_collision_area=true, show_aircraft=true,
                      hold=false, ymin=missing, ymax=missing,
                      climb_color="#007662", descend_color="#8c1515",
                      alpha=1, fillalpha=0.5, belief_lw=2, action_ms=4, fontfamily="Computer Modern")
    X = get_taus(h)[1:t]

    plotf = hold ? plot! : plot

    plotf(size=(450, 300), xlims=(0, pomdp.τ_max),
          xlabel=raw"time to closest approach ($\tau$)", ylabel=raw"relative altitude ($h_\mathrm{rel}$)",
          fontfamily=fontfamily, framestyle=:box)

    !hold && hline!([0], label=false, c=:black, lw=0.5, ls=:dot)

    if show_belief
        plot!(X, get_belief_mean_h_rel(h)[1:t], c=:gray, lw=belief_lw, ls=:dash,
            ribbon=get_belief_std_h_rel(h)[1:t], label=false, fillalpha=fillalpha)
    end

    # true state
    plot!(X, get_h_rel(h)[1:t], label=false, xflip=true, c=:black, lw=1, alpha=alpha)

    if show_obs
        mark = :circle
        color = :white
        msc = :black
        ms = 2
        obs_x = X
        obs_y = get_obs_h_rel(h)[1:t]
        scatter!(obs_x, obs_y; ms, label=false, mark, color, msc)
    end

    if show_actions
        act_x = copy(X)
        act_y = get_h_rel(h)[1:t]
        A = get_actions(h)[1:t]
        AI = map(a->actionindex(pomdp, a), A)
        markers = Dict(
            -5 => :dtriangle,
            0 => :square,
            5 => :utriangle,
        )
        action_colors = Dict(
            -5 => descend_color,
            0 => :white,
            5 => climb_color,
        )
        stroke_colors = Dict(
            -5 => :white,
            0 => :gray,
            5 => :white,
        )
        mark = [markers[a] for a in A]
        color = [action_colors[a] for a in A]
        msc = [stroke_colors[a] for a in A]
        ms = action_ms
        if !show_zero_actions
            idx = AI .== actionindex(pomdp, 0)
            deleteat!(mark, idx)
            deleteat!(color, idx)
            deleteat!(msc, idx)
            deleteat!(act_x, idx)
            deleteat!(act_y, idx)
        end
        scatter!(act_x, act_y; ms, label=false, mark, color, msc)
    end

    if show_collision_area
        plot!(rectangle(1, 2pomdp.collision_threshold, 0, -pomdp.collision_threshold), opacity=0.25, color=:crimson, label=false)
    end

    yl = ylims()

    if ismissing(ymin)
        ymin = min(yl[1], pomdp.h_rel_range[1])
    end
    if ismissing(ymax)
        ymax = max(yl[2], pomdp.h_rel_range[2])
    end

    ylims!(min(ymin, -abs(ymax)), max(abs(ymin), ymax))

    if show_aircraft
        overlay_aircraft!()
    end

    return plot!()
end

function overlay_aircraft!()
	img = load(joinpath(@__DIR__, "..", "img", "airplane.png"))
	xl = xlims()
	yl = ylims()
	ratio = sum(abs.(xl)) / (sum(abs.(yl)) + sum(abs.(xl)))
	width = 4
	height = width / ratio
    X = [0, width - 1]
    Y = [-height/2, height/2]
	return plot!(X, Y, reverse(img, dims=1), yflip=false, ratio=:none)
end

function generate_histories(pomdp::CollisionAvoidancePOMDP, policy::Policy, up::Updater, n::Int; parallel=false)
    fmap = parallel ? pmap : map
    return fmap(_->simulate(HistoryRecorder(), pomdp, policy, up), 1:n)
end

function plot_histories(pomdp, H::Vector{<:SimHistory}; kwargs...)
    for (i,h) in enumerate(H)
        isfirst = i==1
        plot_history(pomdp, h;
                     ymin=-350,
                     ymax=350,
                     show_actions=true,
                     show_obs=false,
                     show_collision_area=isfirst,
                     show_aircraft=isfirst,
                     show_belief=true,
                     fillalpha=0.1,
                     belief_lw=0,
                     show_zero_actions=false,
                     alpha=0.25,
                     hold=!isfirst,
                     kwargs...)
    end
    return plot!()
end

blur(img, σ) = imfilter(Float64.(img), ImageFiltering.Kernel.gaussian(σ))
@enum PlotTyle ValuePlot PolicyPlot PfailPlot

function cas_policy_plot(pomdp, up, policy, policy_lookup;
        plot_type=PolicyPlot,
        replan=false,
        n_runs=1,
        use_blur=true,
        use_mean=true,
        discrete_action_colors=false,
        coarse=true,
        verbose=false,
        dh_rel=0,
        a_prev=0,
        h_rel_max=350,
        fontfamily="Computer Modern")
    h_rel_range = coarse ? (-h_rel_max:35:h_rel_max) : (-h_rel_max:2:h_rel_max)
    f_dh_rel = ()->dh_rel # randn()
    f_a_prev = ()->a_prev # rand(actions(pomdp))
    τ_range = coarse ? (0:40) : (0:0.25:40)

    ds0 = initialstate(pomdp)
    b0 = initialize_belief(up, ds0)
    A = actions(pomdp)
    s = rand(ds0)
    policy_map = Matrix(undef, length(h_rel_range), length(τ_range))

    # @distributed
    @showprogress for (i,j) in collect(Iterators.product(eachindex(τ_range), eachindex(h_rel_range)))
        x = τ_range[i]
        y = h_rel_range[j]

        # s[1] = y
        # s[2] = f_dh_rel()
        # s[3] = f_a_prev()
        # s[4] = round(x)
        # o = rand(observation(pomdp, s))

        b = deepcopy(b0)
        b.ukf.μ[1] = y
        b.ukf.μ[2] = f_dh_rel()
        b.ukf.μ[3] = f_a_prev()
        b.ukf.μ[4] = x
        b.ukf.Σ .= 0

        # b = update(up, b, f_a_prev(), o)
        if plot_type == ValuePlot
            z = policy_lookup(policy.surrogate, b)
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
            z = policy_lookup(policy.surrogate, b)
        end
        verbose && @info b.ukf.μ z
        policy_map[j,i] = z
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

    if plot_type == PolicyPlot && use_blur
        Plots.heatmap(τ_range, h_rel_range, blur(policy_map, 1.5); xflip=true, label=false, size=(400,250), title=title_str, clims=(minimum(A), maximum(5)), kwargs...)
    else
        Plots.heatmap(τ_range, h_rel_range, policy_map; xflip=true, label=false, size=(400,250), title=title_str, kwargs...)
    end
    Plots.xlabel!(raw"time to closest approach ($\tau$)")
    Plots.ylabel!(raw"relative altitude ($h_\mathrm{rel}$)")

    # Plots.hline!([0], label=false, color=:black)
    return Plots.plot!(fontfamily=fontfamily, framestyle=:box)
end
