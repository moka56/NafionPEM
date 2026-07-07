using Printf

function generate_nafion_chain(num_monomers::Int)
    atoms = Tuple{String, Float64, Float64, Float64}[]
    cc_bond, phi = 1.54, (180.0 - 109.5) * pi / 180.0
    x, y, z = 0.0, 0.0, 0.0
    
    for i in 1:num_monomers
        x = (i - 1) * cc_bond * cos(phi/2)
        y = (i % 2 == 0) ? 0.0 : cc_bond * sin(phi/2)
        push!(atoms, ("C", x, y, z))
        push!(atoms, ("F", x, y + 1.0, z + 1.0))
        push!(atoms, ("F", x, y - 1.0, z - 1.0))
        
        if i % 5 == 0
            push!(atoms, ("O", x, y + 1.4, z))
            push!(atoms, ("C", x, y + 2.8, z))
            push!(atoms, ("F", x + 1.0, y + 2.8, z + 1.0))
            push!(atoms, ("F", x - 1.0, y + 2.8, z - 1.0))
            push!(atoms, ("S", x, y + 4.5, z))
            push!(atoms, ("O", x + 1.2, y + 4.5, z + 1.0))
            push!(atoms, ("O", x - 1.2, y + 4.5, z - 1.0))
            push!(atoms, ("O", x, y + 4.5, z - 1.5))
            push!(atoms, ("H", x, y + 5.2, z - 1.5))
        end
    end
    
    open("nafion_chain.xyz", "w") do io
        println(io, length(atoms))
        println(io, "Nafion single chain generated via Julia")
        for a in atoms
            @printf(io, "%-2s %12.6f %12.6f %12.6f\n", a[1], a[2], a[3], a[4])
        end
    end
    println("SUCCESS: nafion_chain.xyz has been created!")
end

generate_nafion_chain(20)