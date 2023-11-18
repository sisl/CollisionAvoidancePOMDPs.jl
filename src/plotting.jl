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
                      show_actions=true, action_colors=["#8c1515", :white, "#007662"],
                      show_collision_area=true, show_aircraft=true,
                      ymin=missing, ymax=missing)
    X = get_taus(h)[1:t]

    plot(size=(450, 300), xlims=(0, pomdp.τ_max),
         xlabel=raw"time to closest approach ($\tau$)", ylabel=raw"relative altitude ($h_\mathrm{rel}$)",
         fontfamily="Computer Modern", framestyle=:box)

    hline!([0], label=false, c=:black, lw=0.5)

    # belief
    plot!(X, get_belief_mean_h_rel(h)[1:t], c=:gray, lw=2, ls=:dash,
          ribbon=get_belief_std_h_rel(h)[1:t], label=false)

    # true state
    plot!(X, get_h_rel(h)[1:t], label=false, xflip=true, c=:black, lw=1)

    if show_actions
        AI = map(a->actionindex(pomdp, a), get_actions(h))
        markers = [:dtriangle, :square, :utriangle]
        stroke_colors = [action_colors[1], :gray, action_colors[3]]
        mark = [markers[ai] for ai in AI]
        color = [action_colors[ai] for ai in AI]
        msc = [stroke_colors[ai] for ai in AI]
        ms = 3
    else
        mark = :circle
        color = :white
        msc = :black
        ms = 2
    end
    scatter!(X, get_obs_h_rel(h)[1:t]; ms, label=false, mark, color, msc)

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
	img = load("../img/airplane.png")
	xl = xlims()
	yl = ylims()
	ratio = sum(abs.(xl)) / (sum(abs.(yl)) + sum(abs.(xl)))
	width = 4
	height = width / ratio
    X = [0, width - 1]
    Y = [-height/2, height/2]
	return plot!(X, Y, reverse(img, dims=1), yflip=false, ratio=:none)
end
