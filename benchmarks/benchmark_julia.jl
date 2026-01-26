# Julia benchmark script for MixedModels.jl comparison
# Run with: julia benchmark_julia.jl
#
# Requires MixedModels.jl:
#   using Pkg; Pkg.add(["MixedModels", "DataFrames", "Random", "BenchmarkTools"])

using MixedModels
using DataFrames
using Random
using BenchmarkTools
using Statistics

function generate_lmm_data(n_obs::Int, n_groups::Int; seed::Int=42)
    Random.seed!(seed)

    group = rand(1:n_groups, n_obs)
    x1 = randn(n_obs)

    beta = [1.0, 0.5]
    group_effects = randn(n_groups) * 2.0

    y = beta[1] .+ beta[2] .* x1 .+ group_effects[group] .+ randn(n_obs)

    DataFrame(
        y = y,
        x1 = x1,
        group = categorical(string.(group))
    )
end

function benchmark_fit(df::DataFrame; n_evals::Int=10)
    formula = @formula(y ~ 1 + x1 + (1 | group))

    # Warmup
    fit(MixedModel, formula, df; REML=true)

    # Benchmark
    times = Float64[]
    for _ in 1:n_evals
        t = @elapsed fit(MixedModel, formula, df; REML=true)
        push!(times, t * 1000)  # Convert to ms
    end

    return times
end

function run_benchmarks()
    println("=" ^ 70)
    println("MixedModels.jl Benchmark")
    println("=" ^ 70)

    sizes = [
        (100, 10),
        (500, 25),
        (1000, 50),
        (5000, 100),
        (10000, 200),
        (50000, 500),
    ]

    results = Dict{String, Dict}()

    for (n_obs, n_groups) in sizes
        println("\nn_obs=$n_obs, n_groups=$n_groups")

        df = generate_lmm_data(n_obs, n_groups)

        try
            times = benchmark_fit(df; n_evals=5)
            mean_time = mean(times)
            std_time = std(times)

            results["$(n_obs)_$(n_groups)"] = Dict(
                "mean_ms" => mean_time,
                "std_ms" => std_time,
                "n_obs" => n_obs,
                "n_groups" => n_groups
            )

            println("  Mean: $(round(mean_time, digits=2))ms ± $(round(std_time, digits=2))ms")
        catch e
            println("  Error: $e")
            results["$(n_obs)_$(n_groups)"] = Dict(
                "error" => string(e),
                "n_obs" => n_obs,
                "n_groups" => n_groups
            )
        end
    end

    # Summary table
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println("Size                  Mean Time (ms)")
    println("-" ^ 70)

    for (key, data) in sort(collect(results), by=x->x[2]["n_obs"])
        n_obs = data["n_obs"]
        n_groups = data["n_groups"]
        if haskey(data, "mean_ms")
            println("n=$(n_obs), g=$(n_groups)          $(round(data["mean_ms"], digits=2))")
        else
            println("n=$(n_obs), g=$(n_groups)          ERROR")
        end
    end

    return results
end

# Export to JSON for comparison with Python benchmarks
function save_results(results, filename="julia_benchmark_results.json")
    open(filename, "w") do f
        # Simple JSON-like output (Julia's JSON.jl would be better)
        println(f, "{")
        entries = collect(results)
        for (i, (key, data)) in enumerate(entries)
            println(f, "  \"$key\": {")
            for (j, (k, v)) in enumerate(collect(data))
                comma = j < length(data) ? "," : ""
                if v isa String
                    println(f, "    \"$k\": \"$v\"$comma")
                else
                    println(f, "    \"$k\": $v$comma")
                end
            end
            comma = i < length(entries) ? "," : ""
            println(f, "  }$comma")
        end
        println(f, "}")
    end
    println("\nResults saved to $filename")
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_benchmarks()
    save_results(results)
end
