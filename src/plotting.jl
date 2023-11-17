get_actions(h) = [step.a for step in h]
get_h_rel(h) = [step.s[1] for step in h]
get_dh_rel(h) = [step.s[2] for step in h]
get_a_prev(h) = [step.s[3] for step in h]
get_taus(h) = [step.s[4] for step in h]
get_obs_h_rel(h) = [step.o[1] for step in h]
get_belief_mean_h_rel(h) = [mean(step.b)[1] for step in h]
get_belief_std_h_rel(h) = [cov(step.b)[1,1] for step in h]

function plot_history(pomdp::CollisionAvoidancePOMDP, h::SimHistory, t=length(h);
                      ymin=missing, ymax=missing)
    X = get_taus(h)[1:t]

    plot(size=(450, 300), xlims=(0, pomdp.τ_max),
         xlabel=raw"time to closest approach ($\tau$)", ylabel=raw"relative altitude ($h_\mathrm{rel}$)",
         fontfamily="Computer Modern", framestyle=:box)
    hline!([0], label=false, c=:black, lw=0.5)

    plot!(X, get_belief_mean_h_rel(h)[1:t], c=:gray, lw=2,
          ribbon=get_belief_std_h_rel(h)[1:t], label=false)

    plot!(X, get_h_rel(h)[1:t], label=false, xflip=true, c=:crimson, lw=2, α=0.5)

    scatter!(X, get_obs_h_rel(h)[1:t], ms=2, label=false, c=:white)

    yl = ylims()

    if ismissing(ymin)
        ymin = min(yl[1], pomdp.h_rel_range[1])
    end
    if ismissing(ymax)
        ymax = max(yl[2], pomdp.h_rel_range[2])
    end

    return ylims!(min(ymin, -abs(ymax)), max(abs(ymin), ymax))
end