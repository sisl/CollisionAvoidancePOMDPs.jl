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
                      action_colors=["#8c1515", :white, "#007662"],
                      hold=false, ymin=missing, ymax=missing,
                      alpha=1, fillalpha=0.5, belief_lw=2, action_ms=4)
    X = get_taus(h)[1:t]

    plotf = hold ? plot! : plot

    plotf(size=(450, 300), xlims=(0, pomdp.τ_max),
          xlabel=raw"time to closest approach ($\tau$)", ylabel=raw"relative altitude ($h_\mathrm{rel}$)",
          fontfamily="Computer Modern", framestyle=:box)

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
        AI = map(a->actionindex(pomdp, a), get_actions(h)[1:t])
        markers = [:dtriangle, :square, :utriangle]
        stroke_colors = [:white, :gray, :white]
        mark = [markers[ai] for ai in AI]
        color = [action_colors[ai] for ai in AI]
        msc = [stroke_colors[ai] for ai in AI]
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

function generate_histories(pomdp::CollisionAvoidancePOMDP, policy::Policy, up::Updater, n::Int)
    return [simulate(HistoryRecorder(), pomdp, policy, up) for _ in 1:n]
end

function plot_histories(pomdp, H::Vector{<:SimHistory})
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
                     hold=!isfirst)
    end
    return plot!()
end
