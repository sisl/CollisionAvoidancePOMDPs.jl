r_struct = r"@with_kw struct CollisionAvoidancePOMDP <: POMDP\{Vector\{Float64\}, Float64, Vector\{Float64\}\}(.|\n)*?end"

filename_pomdp = joinpath(@__DIR__, "..", "src", "pomdp.jl")
filename_readme = joinpath(@__DIR__, "..", "README.md")

pomdp = read(filename_pomdp, String)
readme = read(filename_readme, String)

m_pomdp = match(r_struct, pomdp)
m_readme = match(r_struct, readme)

readme = replace(readme, m_readme.match=>m_pomdp.match)
open(filename_readme, "w+") do f
    write(f, readme)
end
