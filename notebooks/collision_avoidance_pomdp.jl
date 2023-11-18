### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 50e31b2d-5b8c-4235-ae10-fc9831b2d4d1
begin
	using Pkg
	Pkg.develop(path="..")
end

# ╔═╡ b5498630-4deb-4a24-9179-991292fbe630
begin
	Pkg.add("POMCPOW")
	using POMCPOW
end

# ╔═╡ a75d231f-afef-4d1e-9d5f-60f71e283331
using Revise, CollisionAvoidancePOMDPs

# ╔═╡ 9e7ab243-2f25-4dda-928c-554afabd216a
using MCTS

# ╔═╡ 239e6a17-1017-4b0e-b5cf-64c499e81693
using LocalApproximationValueIteration, LocalFunctionApproximation, GridInterpolations

# ╔═╡ 33817805-815e-43bd-83d4-395903ff7b36
using PlutoUI

# ╔═╡ adbd3d35-89db-4776-9fd3-56bd2d8e3044
pomdp = CollisionAvoidancePOMDP(
	reward_reversal=-25, reward_alert=-50, 				
	reward_collision=-200, ddh_max=3.0,
	h_rel_range=[-10, 10], dh_rel_range=[-1, 1],
	px=DiscreteNonParametric([0.1, 0.0, -0.1], [0.25, 0.5, 0.25])
);

# ╔═╡ 7dc77c34-30f1-4f6d-a427-95e35bf96c55
md"Use POMCOW: $(@bind use_pomcpow CheckBox(false))"

# ╔═╡ bb7e5be9-ca30-4c72-a8fd-327c69b2155a
md"Use Belief MCTS: $(@bind use_bmcts CheckBox(false))"

# ╔═╡ 8a13cf22-a3ed-4785-9e27-9b96653e34d0
POMDPs.isterminal(bmdp::GenerativeBeliefMDP, b::CASBelief) = isterminal(bmdp.pomdp, mean(b))

# ╔═╡ 1937f5ac-8504-440d-96ca-d3b3a07f2b63
up = CASBeliefUpdater(pomdp)

# ╔═╡ c6d5250c-85ad-4e23-88d5-be1f63b22349
mean(isfailure(pomdp, simulate(HistoryRecorder(), pomdp, FunctionPolicy(b->0), up)[end].sp) for _ in 1:100) # Nominal failure rate

# ╔═╡ a454baaa-eb4e-4c90-afdf-0506b195284a
mean(isfailure(pomdp, simulate(HistoryRecorder(), pomdp, RandomPolicy(pomdp), up)[end].sp) for _ in 1:100) # Nominal failure rate with random policy

# ╔═╡ 0d26e906-e24b-4419-a92f-07e2d81b64c4
if use_pomcpow
	solver = POMCPOWSolver(max_depth=pomdp.τ_max+1)
	policy = solve(solver, pomdp)
elseif use_bmcts
	bmdp_solver = DPWSolver(n_iterations=50, # 100,
		                    depth=Int(pomdp.τ_max+1),
							# enable_action_pw=false,
		                    exploration_constant=200.0)
	solver = BeliefMCTSSolver(bmdp_solver, up)
	policy = solve(solver, pomdp)
else
	policy = RandomPolicy(pomdp)
	# policy = FunctionPolicy(b->0)
end;

# ╔═╡ 6f1db216-080f-424f-89b7-134c37a1a655
@bind seed Slider(0:10, show_value=true)

# ╔═╡ d8b0b08c-3fdf-4c84-b803-a5dac74ccbfa
begin
	Random.seed!(seed)
	ds0 = initialstate(pomdp)
	b0 = initialize_belief(up, ds0)
	s0 = rand(b0)
	h = simulate(HistoryRecorder(), pomdp, policy, up, b0, s0)
end

# ╔═╡ ce09ad64-8211-4e99-a459-a19f08eabdcf
@bind t Slider(eachindex(h), show_value=true, default=length(h))

# ╔═╡ e4b44d0d-fa52-42e2-96bf-9719f85aa2e2
isfailure(pomdp, h[end].s)

# ╔═╡ 4c91339f-fef9-4e6f-abcf-1afe37c56f06
plt = plot_history(pomdp, h, t; ymin=-350, ymax=350, show_actions=true)

# ╔═╡ 5147c44d-78fb-4754-9609-b6036638e136
plt

# ╔═╡ 9f2e51e7-41f1-466d-b855-34237c910573
begin
	using Images
	img = load("../img/airplane.png")
	plot(plt)
	xl = xlims()
	yl = ylims()
	ratio = sum(abs.(xl)) / (sum(abs.(yl)) + sum(abs.(xl)))
	width = 4
	height = width / ratio
	plot!([0, width - 1], [-height/2, height/2], reverse(img, dims=1), yflip=false, aspect_ratio=:none)
end

# ╔═╡ 63a12560-8505-11ee-347d-ed393cefa98d
md"""
# Aircraft collision avoidance
"""

# ╔═╡ 5af8c4b2-d151-4110-8b25-96b7988b657c
md"""
# Plotting
"""

# ╔═╡ 0d4c9d80-4c4a-469c-8946-b6b41f0ec0bf
discounted_reward(h)

# ╔═╡ 1911419c-32c0-4e46-945b-8eaeab5bc243
get_actions(h)

# ╔═╡ 43012d73-9d05-4840-b474-afad58417fc9
md"""
## Value iteration
"""

# ╔═╡ 275e8629-70be-4c11-b88b-134d0bd11412
begin
	min_h = -100
	max_h = 100
	min_dh = -10
	max_dh = 10
end;

# ╔═╡ 725fb7d1-4f3f-4777-a856-d22db9b7f28a
discrete_length = 1000;

# ╔═╡ 316ce2b4-8ad9-4356-8c81-d73a121805f4
grid = RectangleGrid(range(min_h, stop=max_h, length=discrete_length),
	                 range(min_dh, stop=max_dh, length=discrete_length));

# ╔═╡ cbecaa42-df1d-4149-88ea-87e15c01cafe
interpolation = LocalGIFunctionApproximator(grid);

# ╔═╡ 1ea131a7-9102-45f7-8a72-c7bae2330e7d
lavi_solver = LocalApproximationValueIterationSolver(interpolation,
											         max_iterations=100,
	        	  	 						         is_mdp_generative=true,
												     verbose=true,
												     n_generative_samples=1);

# ╔═╡ fb547953-98c1-4a8a-8341-eb2b6378bc3e
bmdp = GenerativeBeliefMDP(pomdp, up);

# ╔═╡ de66dbb8-32a7-4ad0-8ddc-e99d02ef7a0f
# lavi_policy = solve(lavi_solver, bmdp);

# ╔═╡ 4f752943-2ddc-4766-8579-c52dbaf9c3d2
# begin

# 	default(fontfamily="Computer Modern", framestyle=:box)
	
# 	h_rel_range = -100:2:100
# 	dh_rel = 0
# 	a_prev = 0
# 	τ_range = 0:40
	
# 	policy_map = Matrix(undef, length(h_rel_range), length(τ_range))
# 	for (i,x) in enumerate(τ_range)
# 	    for (j,y) in enumerate(h_rel_range)
# 	        b = deepcopy(b0)
# 	        b.ukf.μ[1] = y
# 	        b.ukf.μ[2] = dh_rel # randn()
# 	        b.ukf.μ[3] = a_prev # rand(actions(pomdp))
# 	        b.ukf.μ[4] = x
# 	        # ai = action(lavi_policy, b)
# 	        ai = value(lavi_policy, b)
# 	        policy_map[j,i] = ai
# 	    end
# 	end
	
# 	Plots.heatmap(τ_range, h_rel_range, policy_map, xflip=true, label=false, c=cgrad(:jet, rev=true))
# 	Plots.xlabel!(raw"time to closest approach ($\tau$)")
# 	Plots.ylabel!(raw"relative altitude ($h_\mathrm{rel}$)")
# end

# ╔═╡ 693c69ee-9557-4b17-9e46-bf3bb37352a3
POMDPs.actionindex(bmdp::GenerativeBeliefMDP{CollisionAvoidancePOMDP, CASBeliefUpdater}, a::Real) = actionindex(bmdp.pomdp, a)

# ╔═╡ f4d941a0-a52c-4577-a0f9-444164f42266
POMDPs.convert_s(::Type{Vector{Float64}}, b::CASBelief, bmdp::GenerativeBeliefMDP{CollisionAvoidancePOMDP,CASBeliefUpdater}) = Float64.(mean(b)[1:2])

# ╔═╡ a06f3c9e-b55c-4013-8f68-c15c8c3f4fba
function POMDPs.convert_s(::Type{CASBelief}, v::Any, bmdp::GenerativeBeliefMDP{CollisionAvoidancePOMDP,CASBeliefUpdater})
	ukf = initialize_belief(bmdp.updater, initialstate(bmdp.pomdp)).ukf
	ukf.μ[1] = v[1]
	ukf.μ[2] = v[2]
	return CASBelief(ukf)
end

# ╔═╡ 3f6abaca-63be-4046-831c-5875bef14c4e
md"""
## Collision
"""

# ╔═╡ 051ae044-a5ca-4360-886d-ebfb134ae6c9
begin
	Random.seed!(seed)
	pomdp2 = CollisionAvoidancePOMDP(px=DiscreteNonParametric([0.0], [1.0]))
	up2 = CASBeliefUpdater(pomdp2)
	policy2 = FunctionPolicy(b->0)
	ds02 = initialstate(pomdp)
	b02 = initialize_belief(up, ds02)
	s02 = [0.0, 0.0, 0, pomdp2.τ_max]
	h2 = simulate(HistoryRecorder(max_steps=41), pomdp2, policy2, up2, b02, s02)
end

# ╔═╡ cf0639e9-ee9b-4bf6-87aa-da078fb1cb03
isfailure(pomdp2, h2[end].s)

# ╔═╡ 633a9f8d-40a9-4da0-829b-50bf44f9e173
plt2 = plot_history(pomdp2, h2, t; ymin=-350, ymax=350)

# ╔═╡ a7c46898-f051-4463-8448-4112b88babc6
isterminal(pomdp, mean(b0))

# ╔═╡ d05b74e2-5aea-4619-a6ff-48c5f8cbfb07
mean(b0)

# ╔═╡ 6776676b-cf5d-4112-9fc3-678c25a8316c
b0.ukf

# ╔═╡ 46e0f178-378b-4eb1-b3c1-fe4b90b6bd63
md"""
---
"""

# ╔═╡ 85b04aa4-bce4-4c48-8d29-2083b49785b3
1/ratio

# ╔═╡ b145ebb6-e827-4af6-9262-37a962a0fa90
width, height

# ╔═╡ 6fd348b4-6de4-4ed2-b9e4-fdc101c0cbbb
TableOfContents()

# ╔═╡ 7ae29367-8cc2-4d54-9711-f86b8b7871ff
function docs(sym::Symbol)
	Base.doc(Base.Docs.Binding(CollisionAvoidancePOMDPs, sym))
end

# ╔═╡ 3f1fcc8a-f810-46bf-9ecc-109180143a78
docs(:initialstate)

# ╔═╡ c2cd033a-d9c3-4568-beb4-02efe1492a02
docs(:UKFUpdater)

# ╔═╡ 2cfe158a-0f0d-42e6-bd91-4ca10114d29d
docs(:sigma_points)

# ╔═╡ 49d51e0c-ed4b-40e3-8e07-fe48350dabca
docs(:weights)

# ╔═╡ eaa79ae4-a4fa-4030-9dd2-452a24338394
docs(:unscented_transform)

# ╔═╡ f69e9d03-ff87-448c-a447-59b14518c047
docs(:update)

# ╔═╡ Cell order:
# ╟─63a12560-8505-11ee-347d-ed393cefa98d
# ╠═50e31b2d-5b8c-4235-ae10-fc9831b2d4d1
# ╠═a75d231f-afef-4d1e-9d5f-60f71e283331
# ╟─3f1fcc8a-f810-46bf-9ecc-109180143a78
# ╠═5147c44d-78fb-4754-9609-b6036638e136
# ╟─c2cd033a-d9c3-4568-beb4-02efe1492a02
# ╟─2cfe158a-0f0d-42e6-bd91-4ca10114d29d
# ╟─49d51e0c-ed4b-40e3-8e07-fe48350dabca
# ╟─eaa79ae4-a4fa-4030-9dd2-452a24338394
# ╟─f69e9d03-ff87-448c-a447-59b14518c047
# ╟─5af8c4b2-d151-4110-8b25-96b7988b657c
# ╠═b5498630-4deb-4a24-9179-991292fbe630
# ╠═9e7ab243-2f25-4dda-928c-554afabd216a
# ╠═adbd3d35-89db-4776-9fd3-56bd2d8e3044
# ╠═c6d5250c-85ad-4e23-88d5-be1f63b22349
# ╠═a454baaa-eb4e-4c90-afdf-0506b195284a
# ╟─7dc77c34-30f1-4f6d-a427-95e35bf96c55
# ╟─bb7e5be9-ca30-4c72-a8fd-327c69b2155a
# ╠═8a13cf22-a3ed-4785-9e27-9b96653e34d0
# ╠═0d26e906-e24b-4419-a92f-07e2d81b64c4
# ╠═1937f5ac-8504-440d-96ca-d3b3a07f2b63
# ╠═d8b0b08c-3fdf-4c84-b803-a5dac74ccbfa
# ╠═6f1db216-080f-424f-89b7-134c37a1a655
# ╠═ce09ad64-8211-4e99-a459-a19f08eabdcf
# ╠═e4b44d0d-fa52-42e2-96bf-9719f85aa2e2
# ╠═4c91339f-fef9-4e6f-abcf-1afe37c56f06
# ╠═0d4c9d80-4c4a-469c-8946-b6b41f0ec0bf
# ╠═1911419c-32c0-4e46-945b-8eaeab5bc243
# ╟─43012d73-9d05-4840-b474-afad58417fc9
# ╠═239e6a17-1017-4b0e-b5cf-64c499e81693
# ╠═275e8629-70be-4c11-b88b-134d0bd11412
# ╠═725fb7d1-4f3f-4777-a856-d22db9b7f28a
# ╠═316ce2b4-8ad9-4356-8c81-d73a121805f4
# ╠═cbecaa42-df1d-4149-88ea-87e15c01cafe
# ╠═1ea131a7-9102-45f7-8a72-c7bae2330e7d
# ╠═fb547953-98c1-4a8a-8341-eb2b6378bc3e
# ╠═de66dbb8-32a7-4ad0-8ddc-e99d02ef7a0f
# ╠═4f752943-2ddc-4766-8579-c52dbaf9c3d2
# ╠═693c69ee-9557-4b17-9e46-bf3bb37352a3
# ╠═f4d941a0-a52c-4577-a0f9-444164f42266
# ╠═a06f3c9e-b55c-4013-8f68-c15c8c3f4fba
# ╟─3f6abaca-63be-4046-831c-5875bef14c4e
# ╠═051ae044-a5ca-4360-886d-ebfb134ae6c9
# ╠═cf0639e9-ee9b-4bf6-87aa-da078fb1cb03
# ╠═633a9f8d-40a9-4da0-829b-50bf44f9e173
# ╠═a7c46898-f051-4463-8448-4112b88babc6
# ╠═d05b74e2-5aea-4619-a6ff-48c5f8cbfb07
# ╠═6776676b-cf5d-4112-9fc3-678c25a8316c
# ╟─46e0f178-378b-4eb1-b3c1-fe4b90b6bd63
# ╠═9f2e51e7-41f1-466d-b855-34237c910573
# ╠═85b04aa4-bce4-4c48-8d29-2083b49785b3
# ╠═b145ebb6-e827-4af6-9262-37a962a0fa90
# ╠═33817805-815e-43bd-83d4-395903ff7b36
# ╠═6fd348b4-6de4-4ed2-b9e4-fdc101c0cbbb
# ╠═7ae29367-8cc2-4d54-9711-f86b8b7871ff
