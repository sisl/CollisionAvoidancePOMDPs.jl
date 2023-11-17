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
	Pkg.develop(path="../CollisionAvoidancePOMDPs.jl")
end

# ╔═╡ a75d231f-afef-4d1e-9d5f-60f71e283331
using Revise, CollisionAvoidancePOMDPs

# ╔═╡ 33817805-815e-43bd-83d4-395903ff7b36
using PlutoUI

# ╔═╡ 63a12560-8505-11ee-347d-ed393cefa98d
md"""
# Aircraft collision avoidance
"""

# ╔═╡ adbd3d35-89db-4776-9fd3-56bd2d8e3044
pomdp = CollisionAvoidancePOMDP()

# ╔═╡ c33b5b0e-5687-48f6-99fd-78850bf603d9
md"""
$\pmb{\upmu}$
"""

# ╔═╡ 5af8c4b2-d151-4110-8b25-96b7988b657c
md"""
# Plotting
"""

# ╔═╡ 7dc77c34-30f1-4f6d-a427-95e35bf96c55
md"Use POMCOW: $(@bind use_pomcpow CheckBox(false))"

# ╔═╡ 0d26e906-e24b-4419-a92f-07e2d81b64c4
if use_pomcpow
	Pkg.add("POMCPOW")
	using POMCPOW
	solver = POMCPOWSolver()
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
	up = CASBeliefUpdater(pomdp)
	ds0 = initialstate(pomdp)
	b0 = initialize_belief(up, ds0)
	s0 = rand(b0)
	h = simulate(HistoryRecorder(), pomdp, policy, up, b0, s0)
end

# ╔═╡ ce09ad64-8211-4e99-a459-a19f08eabdcf
@bind t Slider(eachindex(h), show_value=true, default=length(h))

# ╔═╡ 4c91339f-fef9-4e6f-abcf-1afe37c56f06
plt = plot_history(pomdp, h, t; ymin=-350, ymax=350)

# ╔═╡ 5147c44d-78fb-4754-9609-b6036638e136
plt

# ╔═╡ 0d4c9d80-4c4a-469c-8946-b6b41f0ec0bf
discounted_reward(h)

# ╔═╡ e4b44d0d-fa52-42e2-96bf-9719f85aa2e2
isfailure(pomdp, h[end].s)

# ╔═╡ 1911419c-32c0-4e46-945b-8eaeab5bc243
get_actions(h)

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
	h2 = simulate(HistoryRecorder(), pomdp2, policy2, up2, b02, s02)
end

# ╔═╡ cf0639e9-ee9b-4bf6-87aa-da078fb1cb03
isfailure(pomdp2, h2[end].s)

# ╔═╡ 633a9f8d-40a9-4da0-829b-50bf44f9e173
plt2 = plot_history(pomdp2, h2, t; ymin=-350, ymax=350)

# ╔═╡ 46e0f178-378b-4eb1-b3c1-fe4b90b6bd63
md"""
---
"""

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
# ╠═adbd3d35-89db-4776-9fd3-56bd2d8e3044
# ╟─c2cd033a-d9c3-4568-beb4-02efe1492a02
# ╟─2cfe158a-0f0d-42e6-bd91-4ca10114d29d
# ╟─49d51e0c-ed4b-40e3-8e07-fe48350dabca
# ╟─eaa79ae4-a4fa-4030-9dd2-452a24338394
# ╠═c33b5b0e-5687-48f6-99fd-78850bf603d9
# ╟─f69e9d03-ff87-448c-a447-59b14518c047
# ╟─5af8c4b2-d151-4110-8b25-96b7988b657c
# ╟─7dc77c34-30f1-4f6d-a427-95e35bf96c55
# ╠═0d26e906-e24b-4419-a92f-07e2d81b64c4
# ╠═6f1db216-080f-424f-89b7-134c37a1a655
# ╠═d8b0b08c-3fdf-4c84-b803-a5dac74ccbfa
# ╠═ce09ad64-8211-4e99-a459-a19f08eabdcf
# ╠═4c91339f-fef9-4e6f-abcf-1afe37c56f06
# ╠═0d4c9d80-4c4a-469c-8946-b6b41f0ec0bf
# ╠═e4b44d0d-fa52-42e2-96bf-9719f85aa2e2
# ╠═1911419c-32c0-4e46-945b-8eaeab5bc243
# ╟─3f6abaca-63be-4046-831c-5875bef14c4e
# ╠═051ae044-a5ca-4360-886d-ebfb134ae6c9
# ╠═cf0639e9-ee9b-4bf6-87aa-da078fb1cb03
# ╠═633a9f8d-40a9-4da0-829b-50bf44f9e173
# ╟─46e0f178-378b-4eb1-b3c1-fe4b90b6bd63
# ╠═33817805-815e-43bd-83d4-395903ff7b36
# ╠═6fd348b4-6de4-4ed2-b9e4-fdc101c0cbbb
# ╠═7ae29367-8cc2-4d54-9711-f86b8b7871ff
