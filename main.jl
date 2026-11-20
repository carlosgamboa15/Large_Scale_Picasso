using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JuMP, Gurobi, JSON, Distributions, Polyhedra, CDDLib, Dualization, LinearAlgebra, Statistics, Plots, LaTeXStrings
using GLM, DataFrames, Random, NearestNeighbors, StatsPlots, CSV
using Plots.Measures
using SlurmClusterManager
using JLD2, Dates
using Distributed

const PROGRESS_FILE = joinpath(pwd(), "results", "progress.json")
const PROGRESS_HISTORY_FILE = joinpath(
    pwd(), "results", "progress_history.jsonl"
)
const PROGRESS_CHANNEL = RemoteChannel(
    () -> Channel{Any}(10000), 1
)

function save_progress(state)
    mkpath(dirname(PROGRESS_FILE))

    temporal = PROGRESS_FILE * ".tmp"

    open(temporal, "w") do io
        JSON.print(io, state, 4)
    end

    mv(temporal, PROGRESS_FILE; force=true)
end

function save_progress_event(event)
    mkpath(dirname(PROGRESS_HISTORY_FILE))

    open(PROGRESS_HISTORY_FILE, "a") do io
        JSON.print(io, event)
        println(io)
    end
end

function start_monitor()

    state = Dict{String,Any}(
        "step" => "initialization",
        "iteration" => 0,
        "subproblem_iteration" => 0,
        "last_update" => string(now()),
        "last_event" => nothing,
        "workers" => nworkers(),
        "subproblems" => Dict{String,Any}()
    )

    save_progress(state)

    return @async begin

        while true

            event = take!(PROGRESS_CHANNEL)
            event_type = event["type"]

            save_progress_event(event)

            state["last_update"] = string(now())
            state["last_event"] = event_type

            if event_type == "TERMINATE"
                save_progress(state)
                break

            elseif event_type == "STEP"

                state["step"] = event["stage"]
                state["iteration"] = event["iteration"]
                state["subproblem_iteration"] = 0
                state["subproblems"] = Dict{String,Any}()

                for key in ("benders", "master", "oracles", "cuts")
                    delete!(state, key)
                end

            elseif event_type == "SUBPROBLEM"

                event_iter = event["iteration"]

                if event_iter < state["subproblem_iteration"]
                    save_progress(state)
                    continue
                end

                if state["subproblem_iteration"] != event_iter
                    state["subproblems"] = Dict{String,Any}()
                    state["subproblem_iteration"] = event_iter
                end

                key = string(event["subproblem"])

                state["subproblems"][key] = event
                state["iteration"] = event_iter

            elseif event_type == "MASTER"

                state["master"] = event
                state["iteration"] = event["iteration"]

            elseif event_type == "ORACLES"

                state["oracles"] = event
                state["iteration"] = event["iteration"]

            elseif event_type == "CUTS"

                state["cuts"] = event
                state["iteration"] = event["iteration"]

            elseif event_type in (
                "BENDERS",
                "WARM_BENDERS",
                "BENDERS_WITHOUT_WARM_START"
            )

                state["iteration"] = event["iteration"]

                state["benders"] = Dict(
                    "LB" => event["LB"],
                    "UB" => event["UB"],
                    "gap" => event["gap"]
                )

            end

            save_progress(state)

        end

    end

end

println("Starting SLURM workers...")
flush(stdout)

addprocs(SlurmClusterManager.SlurmManager())

println("Julia processes: ", nprocs())
println("Workers Julia: ", nworkers())

@everywhere using Dates
@everywhere const WORKER_PROGRESS_CHANNEL = $PROGRESS_CHANNEL

monitor_task = start_monitor()

put!(PROGRESS_CHANNEL, Dict(
    "type" => "STEP",
    "stage" => "Benders inicial",
    "iteration" => 0
))

println("Number of workers: ", nworkers())

@everywhere println(
    "Process ", myid(), " executed in ", gethostname()
)

@everywhere include(joinpath(pwd(), "getData.jl"))

@everywhere begin

    using JuMP
    using Gurobi
    using  Dualization, LinearAlgebra

    import MathOptInterface as MOI

    const GUROBI_ENV = Gurobi.Env()

end

Data = Main.getData.Data

@everywhere function memory_rss_gb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmRSS:")
                kb = parse(Int, split(line)[2])
                return round(kb / 1024^2, digits=3)
            end
        end
    catch
    end

    return nothing
end

@everywhere function memory_peak_gb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmHWM:")
                kb = parse(Int, split(line)[2])
                return round(kb / 1024^2, digits=3)
            end
        end
    catch
    end

    return nothing
end

@everywhere   function report_oracle(state; extra=Dict{String,Any}())
        event = Dict{String,Any}(
            "type" => "SUBPROBLEM",
            "iteration" => iter,
            "subproblem" => n,
            "state" => state,
            "worker" => myid(),
            "node" => gethostname(),
            "start" => start_datetime,
            "finish" => nothing,
            "duration_seconds" => round(time() - start, digits=2),
            "memory_gb" => memory_rss_gb(),
            "memory_peak_gb" => memory_peak_gb()
        )

        merge!(event, extra)
        put!(WORKER_PROGRESS_CHANNEL, event)
end

function scenarios(Data)

    Y_b = Data.Y_b
    X_lag_b = Data.X_lag_b
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation

    Models = Vector{Any}(undef, length(bus_renewable_generation))
    errors = []

    for b in bus_renewable_generation

        b_ = ind_renewable_generation[b]

        y_vector = vec(Y_b[b_])
        x_vector = vec(X_lag_b[b_])

        df_regresion = DataFrame(Target = y_vector, Lag1 = x_vector)

        model = lm(@formula(Target ~ Lag1), df_regresion)
        
        Models[b_] = model

        forcast = predict(model)

        aux = reshape(df_regresion.Target .- forcast,24,Int(length(y_vector)/24))

        push!(errors, [s[:] for s in eachcol(aux)])

    end

    return Models, errors
   
end

function forecast(Data, zstar, Models)

    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation

    PredictionStar = []

    for b in bus_renewable_generation

        b_ = ind_renewable_generation[b]

        PredictionStar_ = predict(
            Models[b_],
            DataFrame(Lag1 = zstar[b_])
        )

        push!(PredictionStar, PredictionStar_  )

    end

    return PredictionStar

end

function uncertainty_support(Data, zstar, errors)

    z = Data.z
    Tf = Data.Tf
    N = Data.N
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    Tb = length(bus_renewable_generation)

    # Ξ_y = []

    # for b in bus_renewable_generation

    #     b_ = ind_renewable_generation[b]

    #     Z = [z[b_][n][i] for n ∈ sample, i ∈ collect(1:Tf)]
    #     errors = [errors[b_][n][i] for n ∈ sample, i ∈ collect(1:Tf)]

    #     tree = KDTree(permutedims(Z))

    #     k = 3

    #     idxs, dists = knn(tree,zstar[b_],k,true)

    #     Localerrors = errors[idxs,:]

    #     # Contextual dependent uncertainty set Ξ_y(z^*)

    #     Lower = zeros(Tf)

    #     Upper = zeros(Tf)

    #     α = 0.05

    #     for t in 1:Tf

    #         Lower[t] = quantile(Localerrors[:,t],α)

    #         Upper[t] = quantile(Localerrors[:,t],1-α)

    #     end

    #     push!(Ξ_y, Matrix([Lower[:] Upper[:]]))

    # end

    global_residual_scenarios = zeros(Tf * Tb, N)
    global_contextual_scenarios =  zeros(Tf * Tb, N)

    for b in bus_renewable_generation
        b_ = ind_renewable_generation[b]
        row_range = ((b_ - 1) * Tf + 1):(b_ * Tf)
    
        for n in 1:N
            vector_residual_24h = errors[b_][n] 
            vector_contextual_24h = z[b_][n]
            global_residual_scenarios[row_range, n] = vector_residual_24h
            global_contextual_scenarios[row_range, n] = vector_contextual_24h
        end
    end

    zstar_global = vcat(zstar...)
    Z = Matrix(global_contextual_scenarios')
    errors = Matrix(global_residual_scenarios')

    tree = KDTree(permutedims(Z))

    k = Data.N

    idxs, dists = knn(tree,zstar_global,k,true)

    Localerrors = errors[idxs,:]

    # Contextual dependent uncertainty set Ξ_y(z^*)

    Lower = zeros(Tf * Tb)

    Upper = zeros(Tf * Tb)

    α = 0.05

    for t in 1:(Tf * Tb)

        Lower[t] = quantile(Localerrors[:,t],α)

        Upper[t] = quantile(Localerrors[:,t],1-α)

    end

    Ξ_y_global = Matrix([Lower[:] Upper[:]])

    # Ξ_y = @views [ Ξ_y_global[((b - 1) * 24 + 1):(b * 24), :] for b in 1:Tb]

    return Ξ_y_global
    
end

@everywhere function economic_dispatch_problem(Data, u, v, w)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    Tl = Data.Tl
    bus_renewable_generation = Data.bus_renewable_generation
    Cp = Data.Cp
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    Fmax = Data.Fmax
    Cdef = Data.Cdef
    D = Data.D
    F = Data.F
    B = Data.B
    S = Data.S
    Tb, = size(B) # Total buses
    Bus = setdiff(collect(1:Tb), bus_renewable_generation) # Buses
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators
    L = collect(1:Tl) # transmission lines

    # Ignitial state
    p0 = zeros(number_of_generators)
    f0 = zeros(Tl)
    u0 = zeros(number_of_generators)

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Method" => 2, "Crossover" => 0))
    @variable(model, β[1:Tb, 1:T[end]]) # phase
    @variable(model, f[1:L[end], 1:T[end]]) # power flow
    @variable(model, p[1:Ig[end], 1:T[end]] >= 0) # generation
    @variable(model, pdef[1:Tb, 1:T[end]] >= 0) # deficit generation
    @variable(model, s[bus_renewable_generation, 1:T[end]] >= 0) # spillage
    @variable(model, s_positivo[b in bus_renewable_generation, t in T] >= 0)
    @variable(model, s_negativo[b in bus_renewable_generation, t in T] >= 0)

    @constraint(model, [t in T], f[:, t] .== S*β[:, t])
    @constraint(model, [b in Bus, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] == D[b][t]) # Power balance

    @constraint(model, [b in Bus, t in T],  pdef[b, t] ≤ D[b][t]) 
    # @constraint(model, π[b in bus_renewable_generation, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] - s[b, t] - s_positivo[b, t] + s_negativo[b, t] == 0.0) # Uncertainty
    @constraint(model, π[b in bus_renewable_generation, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] - s[b, t]  == 0.0) # Uncertainty

    @constraint(model, [b in bus_renewable_generation, t in T], pdef[b, t] ≤ D[b][t]) 
    @constraint(model, π_s[b in bus_renewable_generation, t in T], s[b, t] <=  0) # Uncertainty
    @constraint(model, [l in L, t in T], f[l,t] <= Fmax[l])
    @constraint(model, [l in L, t in T], f[l,t] >= -Fmax[l])
    @constraint(model, x1[i in Ig, t in T], p[i,t] <= Pmax[i]*u[i, t])
    @constraint(model, x2[i in Ig, t in T], p[i,t] >= Pmin[i]*u[i, t])

    @constraint(model, x3[i in Ig, t in T[2:end]], p[i, t] - p[i, t-1] <= RU[i]*u[i, t-1] + Pmax[i]*v[i, t])
    @constraint(model, x4[i in Ig, t in T[2:end]], p[i, t-1] - p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t])

    # Ignitial state constraints
    @constraint(model, x5[i in Ig, t in [1]], p[i, t] <= RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i])
    @constraint(model, x6[i in Ig, t in [1]], -p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i])

    M_penalizacion = 1e7

    # @objective(model, Min, sum(Cp[i]*p[i,t] for i in Ig, t in T) + sum(Cdef*pdef[b,t] for b in 1:Tb, t in T) + sum(0.1*Cdef*s[b,t] for b in bus_renewable_generation, t in T) + sum(M_penalizacion * (s_positivo[b,t] + s_negativo[b,t]) for b ∈ bus_renewable_generation, t ∈ T))
    @objective(model, Min, sum(Cp[i]*p[i,t] for i in Ig, t in T) + sum(Cdef*pdef[b,t] for b in 1:Tb, t in T) + sum(0.1*Cdef*s[b,t] for b in bus_renewable_generation, t in T) )

    return model
end

@everywhere function create_oracle_problem(Data, u, v, w)

    Tf = Data.Tf
    bus_renewable_generation = Data.bus_renewable_generation
    T = collect(1:Tf)

    # Construir el despacho económico con la solución
    # actual del maestro
    m = economic_dispatch_problem(Data, u, v, w)

    # Dualizar el problema primal
    m_oracle = dualize(
        m;
        dual_names = DualNames()
    )

    # Variables binarias de incertidumbre
    @variable(
        m_oracle,
        ηpos[b in bus_renewable_generation, t in T],
        Bin
    )

    @variable(
        m_oracle,
        ηneg[b in bus_renewable_generation, t in T],
        Bin
    )

    @constraint(
        m_oracle,
        [b in bus_renewable_generation, t in T],
        ηpos[b,t] + ηneg[b,t] <= 1
    )

    @constraint(
        m_oracle,
        positive[b in bus_renewable_generation, t in T],
        ηpos[b,t] + ηneg[b,t] <= 0
    )

    obj = objective_function(m_oracle)

    return m_oracle, obj

end

@everywhere function modify_oracle_linearized(Data, m_oracle, obj, y, Ξ_y, n, λ, PredictionStar)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    D = Data.D
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    RD = Data.RD
    RU = Data.RU
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    T = collect(1:Tf)
    Ig = collect(1:number_of_generators)
    Tb_renewable = length(bus_renewable_generation)

    y_n = y[n]

    lower = max.(Ξ_y[:,1], -vcat(PredictionStar...))
    upper = Ξ_y[:,2]

    yproj_aux = clamp.(y_n, lower, upper)
    Δpos_aux = upper - yproj_aux
    Δneg_aux = yproj_aux - lower

    yproj = @views [yproj_aux[((b - 1) * Tf + 1):(b * Tf)] for b in 1:Tb_renewable]

    Δpos = @views [Δpos_aux[((b - 1) * Tf + 1):(b * Tf)] for b in 1:Tb_renewable]
    Δneg = @views [Δneg_aux[((b - 1) * Tf + 1):(b * Tf)] for b in 1:Tb_renewable]

    @variable(m_oracle, ωpos[b ∈ bus_renewable_generation, t ∈ T])
    @variable(m_oracle, ωneg[b ∈ bus_renewable_generation, t ∈ T])
    @variable(m_oracle, τpos[b ∈ bus_renewable_generation, t ∈ T])
    @variable(m_oracle, τneg[b ∈ bus_renewable_generation, t ∈ T])

    ex = sum(m_oracle[Symbol("π[$b,$t]")] *( D[b][t] - (PredictionStar[ind_renewable_generation[b]][t] + yproj[ind_renewable_generation[b]][t])) -Δpos[ind_renewable_generation[b]][t] * m_oracle[:ωpos][b, t] + Δneg[ind_renewable_generation[b]][t] * m_oracle[:ωneg][b, t] + m_oracle[Symbol("π_s[$b,$t]")] * ( PredictionStar[ind_renewable_generation[b]][t] + yproj[ind_renewable_generation[b]][t]) + Δpos[ind_renewable_generation[b]][t] * m_oracle[:τpos][b, t] - Δneg[ind_renewable_generation[b]][t] * m_oracle[:τneg][b, t] - λ*Δpos[ind_renewable_generation[b]][t] * m_oracle[:ηpos][b, t] - λ*Δneg[ind_renewable_generation[b]][t] * m_oracle[:ηneg][b, t] for b ∈ bus_renewable_generation, t ∈ T)

    set_objective_function(m_oracle, obj + ex - λ * norm(yproj_aux - y_n, 1))

    for t in T
        for b in bus_renewable_generation

            r = ind_renewable_generation[b]

            REG = PredictionStar[r][t] + yproj[r][t]

            set_normalized_coefficient(
                m_oracle[:positive][b,t],
                m_oracle[:ηpos][b,t],
                -Δpos[r][t]
            )

            set_normalized_coefficient(
                m_oracle[:positive][b,t],
                m_oracle[:ηneg][b,t],
                Δneg[r][t]
            )

            set_normalized_rhs(
                m_oracle[:positive][b,t],
                REG
            )
        end
    end

    M = 1e6
    L, U = -M, M
    L_s, U_s = -M, 0.0

    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], L ≤ m_oracle[Symbol("π[$b,$t]")] ≤ U)
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], L_s <= m_oracle[Symbol("π_s[$b,$t]")] <= U_s)

    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], ωpos[b, t] ≥ L*m_oracle[:ηpos][b, t])
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], ωpos[b, t] ≤ U*m_oracle[:ηpos][b, t])
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], ωpos[b, t] ≥ m_oracle[Symbol("π[$b,$t]")] - U*(1 - m_oracle[:ηpos][b, t]))
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], ωpos[b, t] ≤ m_oracle[Symbol("π[$b,$t]")] - L*(1 - m_oracle[:ηpos][b, t]))

    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], ωneg[b, t] ≥ L*m_oracle[:ηneg][b, t])
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], ωneg[b, t] ≤ U*m_oracle[:ηneg][b, t])
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], ωneg[b, t] ≥ m_oracle[Symbol("π[$b,$t]")] - U*(1 - m_oracle[:ηneg][b, t]))
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], ωneg[b, t] ≤ m_oracle[Symbol("π[$b,$t]")] - L*(1 - m_oracle[:ηneg][b, t]))

    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], τpos[b, t] ≥ L_s*m_oracle[:ηpos][b, t])
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], τpos[b, t] ≤ U_s*m_oracle[:ηpos][b, t])
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], τpos[b, t] ≥ m_oracle[Symbol("π_s[$b,$t]")]- U_s*(1 - m_oracle[:ηpos][b, t]))
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], τpos[b, t] ≤ m_oracle[Symbol("π_s[$b,$t]")] - L_s*(1 - m_oracle[:ηpos][b, t]))

    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], τneg[b, t] ≥ L_s*m_oracle[:ηneg][b, t])
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], τneg[b, t] ≤ U_s*m_oracle[:ηneg][b, t])
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], τneg[b, t] ≥  m_oracle[Symbol("π_s[$b,$t]")] - U_s*(1 - m_oracle[:ηneg][b, t]))
    @constraint(m_oracle, [b ∈ bus_renewable_generation, t ∈ T], τneg[b, t] ≤ m_oracle[Symbol("π_s[$b,$t]")] - L_s*(1 - m_oracle[:ηneg][b, t]))

end

@everywhere function solve_oracle_problem(data, iter, n)

    start = time()
    start_datetime = string(now())

    report_oracle("RUNNING")

    Data = data.Data
    D = Data.D
    z = data.z
    y = data.y
    Ξ_y = data.Ξ_y

    u = data.u
    v = data.v
    w = data.w

    λ = data.λ
    μ = data.μ
    θ = data.θ

    zstar = data.zstar
    PredictionStar = data.PredictionStar

    Tf = Data.Tf
    number_of_generators = Data.Ig
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax

    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)

    T = collect(1:Tf)
    Ig = collect(1:number_of_generators)

    a = max.(Ξ_y[:, 1], -vcat(PredictionStar...))
    b = Ξ_y[:, 2]
    zstar_global = vcat(zstar...)

    violation = true

     log_dir = joinpath(
        pwd(), "results", "oracle_logs",
        String(formulation), "gurobi_logs"
    )
    mkpath(log_dir)

    log_file = joinpath(
        log_dir,
        "oracle_iter$(iter)_n$(n).log"
    )

  report_oracle("BUILDING")

 build_start = time()
    m_oracle, obj = create_oracle_problem(
        Data, data.u, data.v, data.w
    )

    set_optimizer(
        m_oracle,
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GUROBI_ENV),
            "OutputFlag" => 1,
            "LogToConsole" => 0,
            "LogFile" => log_file,
            "MIPGap" => 1e-3,
            "Threads" => 1,
            "Seed" => 1,
            "NonConvex" => 2
        )
    )

    time_build = time() - build_start

    report_oracle(
        "BUILT",
        extra=Dict(
            "time_build_seconds" => round(time_build, digits=2)
        )
    )

    report_oracle("MODIFYING")

    modify_start = time()

    modify_oracle_linearized(
        Data, m_oracle, obj,
        data.y, data.Ξ_y, n,
        data.λ, data.PredictionStar
    )

    time_modify = time() - modify_start

    report_oracle(
        "MODIFIED",
        extra=Dict(
            "time_modify_seconds" => round(time_modify, digits=2)
        )
    )

    report_oracle("OPTIMIZING")

    optimize_start = time()

    optimize!(m_oracle)

    time_optimize = time() - optimize_start

    status = termination_status(m_oracle)

    report_oracle(
        "OPTIMIZED",
        extra=Dict(
            "time_optimize_seconds" => round(time_optimize, digits=2),
            "termination_status" => string(status)
        )
    )

    if status != MOI.OPTIMAL
        error("Subproblem n = $n, Oracle Status: $(status)")
    end

    time_oracle = time_modify + time_optimize

    println(
        "Oracle n=$n | ",
        "build=$(round(time_build, digits=2)) s | ",
        "modify=$(round(time_modify, digits=2)) s | ",
        "optimize=$(round(time_optimize, digits=2)) s | ",
        "memory=$(memory_rss_gb()) GiB"
    )

    obj_val = objective_value(m_oracle)
    viol = μ[n] + θ - obj_val + λ*norm(zstar_global - z[n],1)
    yproj = clamp.(y[n], a, b)  
    Δpos = b .- yproj
    Δneg = yproj .- a
    ηpos_val = [value.(m_oracle[:ηpos])[b, t] for b ∈ bus_renewable_generation, t ∈ T]
    ηpos = vec(ηpos_val')
    ηneg_val = [value.(m_oracle[:ηneg])[b, t] for b ∈ bus_renewable_generation, t ∈ T]
    ηneg = vec(ηneg_val')
    y_opt = yproj .+ ηpos .* Δpos .- ηneg .* Δneg
    yopt = @views [y_opt[((b - 1) * Tf + 1):(b * Tf)] for b in 1:length(bus_renewable_generation)]

    const_term_x1, const_term_x2, const_term_x3, const_term_x4, const_term_x5, const_term_x6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    π =  Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T)
    π_s =  Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T)
    ηpos_solution = Dict("($b, $t)" => 0  for b in bus_renewable_generation, t in T)
    ηneg_solution = Dict("($b, $t)" => 0  for b in bus_renewable_generation, t in T)
    x1 = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x2 = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x3 = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x4 = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x5 = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])
    x6 = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])
    
    for t ∈ T

        for b in bus_renewable_generation

            π["($b, $t)"] = value(m_oracle[Symbol("π[$b,$t]")])
            π_s["($b, $t)"] = value(m_oracle[Symbol("π_s[$b,$t]")])
            ηpos_solution["($b, $t)"] = round(Int, value(m_oracle[:ηpos][b, t]))
            ηneg_solution["($b, $t)"] = round(Int, value(m_oracle[:ηneg][b, t]))

        end

        for i in Ig

            x1["($i, $t)"] = value(m_oracle[Symbol("x1[$i,$t]")])
            x2["($i, $t)"] = value(m_oracle[Symbol("x2[$i,$t]")])
            const_term_x1 += Pmax[i]*u[i, t]  * value(m_oracle[Symbol("x1[$i,$t]")])
            const_term_x2 += Pmin[i]*u[i, t]  * value(m_oracle[Symbol("x2[$i,$t]")])

            if t == 1

                x5["($i, $t)"] = value(m_oracle[Symbol("x5[$i,$t]")])
                x6["($i, $t)"] = value(m_oracle[Symbol("x6[$i,$t]")])
                const_term_x5 += (RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i]) * value(m_oracle[Symbol("x5[$i,$t]")])
                const_term_x6 += (RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i]) * value(m_oracle[Symbol("x6[$i,$t]")])

            else

                x3["($i, $t)"] = value(m_oracle[Symbol("x3[$i,$t]")])
                x4["($i, $t)"] = value(m_oracle[Symbol("x4[$i,$t]")])
                const_term_x3 += (RU[i]*u[i, t-1] + Pmax[i]*v[i, t])* value(m_oracle[Symbol("x3[$i,$t]")])
                const_term_x4 += (RD[i]*u[i, t] + Pmax[i]*w[i, t])* value(m_oracle[Symbol("x4[$i,$t]")])

            end

        end

    end

    const_term = copy(objective_value(m_oracle) + λ*norm(y_opt - y[n],1) - (const_term_x1 + const_term_x2 + const_term_x3 + const_term_x4 + const_term_x5 + const_term_x6)) - sum(π["($b, $t)"]*(D[b][t] - PredictionStar[ind_renewable_generation[b]][t] - yopt[ind_renewable_generation[b]][t]) + π_s["($b, $t)"]*(PredictionStar[ind_renewable_generation[b]][t] + yopt[ind_renewable_generation[b]][t]) for b ∈ bus_renewable_generation, t ∈ T)

   ϑ = objective_value(m_oracle) - λ*norm(zstar_global - z[n],1)

    if any(ηpos .> 0) || any(ηneg.> 0)
        println("Optimal and nominal are different for Subproblem n = $n")
    end

    if viol ≥ -1e-2
        violation = false
    end

    oracle_solution = (
        iteration = iter,
        scenario = n,
        yopt = yopt,
        π =  π,
        π_s=  π_s,
        x1 = x1,
        x2 = x2,
        x3 = x3,
        x4 = x4,
        x5 = x5,
        x6 = x6,
        ηpos = ηpos_solution,
        ηneg = ηneg_solution,
        ϑ = ϑ,
        const_term = const_term,
        time_oracle = time_oracle, 
        violation = violation
    )

    result = (
        oracle_solution = oracle_solution,
    )

    report_oracle(
        "COMPLETED",
        extra=Dict(
            "finish" => string(now()),
            "duration_seconds" => round(time() - start, digits=2),
            "time_build_seconds" => round(time_build, digits=2),
            "time_modify_seconds" => round(time_modify, digits=2),
            "time_optimize_seconds" => round(time_optimize, digits=2),
            "time_oracle_seconds" => round(time_oracle, digits=2)
        )
    )

    return result
    
end

function fill_oracle_dict(Data, result, oracle_dict)

    N = Data.N

    for n ∈ 1:N

        for t ∈ T

            for b in bus_renewable_generation

                oracle_dict["π"]["$n"]["($b, $t)"] = result[n].oracle_solution.π["($b, $t)"]
                oracle_dict["π_s"]["$n"]["($b, $t)"] = result[n].oracle_solution.π_s["($b, $t)"]

            end

            for i in Ig

                oracle_dict["x1"]["$n"]["($i, $t)"] = result[n].oracle_solution.x1["($i, $t)"]
                oracle_dict["x2"]["$n"]["($i, $t)"] = result[n].oracle_solution.x2["($i, $t)"]

                if t == 1

                    oracle_dict["x5"]["$n"]["($i, $t)"] = result[n].oracle_solution.x5["($i, $t)"]
                    oracle_dict["x6"]["$n"]["($i, $t)"] = result[n].oracle_solution.x6["($i, $t)"]

                else

                    oracle_dict["x3"]["$n"]["($i, $t)"] = result[n].oracle_solution.x3["($i, $t)"]
                    oracle_dict["x4"]["$n"]["($i, $t)"] = result[n].oracle_solution.x4["($i, $t)"]

                end

            end

        end

        oracle_dict["ϑ"][n] = result[n].oracle_solution.ϑ
        oracle_dict["const"]["$n"] = result[n].oracle_solution.const_term
        oracle_dict["time"]["$n"] = result[n].oracle_solution.time_oracle
        oracle_dict["y"]["$n"] = result[n].oracle_solution.yopt

    end

end

function create_master_problem(Data, ρ)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    N = Data.N
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators
    Cu = Data.Cu
    Csu = Data.Csu
    Csd = Data.Csd

    # Ignitial state
    u0 = zeros(number_of_generators)


    m_master = Model(
    optimizer_with_attributes(
            () -> Gurobi.Optimizer(GUROBI_ENV),
            "OutputFlag" => 0
        )
    )
    @variable(m_master, 0 <= v[1:Ig[end], 1:T[end]] <= 1) # start-up
    @variable(m_master, 0 <= w[1:Ig[end], 1:T[end]] <= 1) # shutdown
    @variable(m_master, u[1:Ig[end], 1:T[end]], Bin) # commit
    @variables(m_master, begin
        μ[1:N] ≥ 0
        θ
        λ ≥ 0
    end)

    @constraint(m_master, [i in Ig, t in T[2:end]], v[i, t] - w[i, t] == u[i, t] - u[i, t-1])
    @constraint(m_master, [i in Ig, t in T], v[i, t] <= u[i, t])
    @constraint(m_master, [i in Ig, t in T], w[i, t] <= 1 - u[i, t])

    # Ignitial state constraint
    @constraint(m_master, [i in Ig, t in [1]], v[i, t] - w[i, t] == u[i, t] - u0[i])

    @objective(m_master, Min, sum(Cu[i]*m_master[:u][i, t] for i in Ig, t in T) + sum(Csu[i]*m_master[:v][i, t] for i in Ig, t in T) +  sum(Csd[i]*m_master[:w][i, t] for i in Ig, t in T) + m_master[:λ]*ρ + m_master[:θ] + (1/(N*α))*sum(m_master[:μ]))

    return m_master
end

function update_master_Benders(Data, m_master, z, zstar, PredictionStar, y, oracle_dict, cuts_Benders)

    N = Data.N
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    number_of_generators = Data.Ig
    D = Data.D

    # Ignitial state
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)

    for n ∈ 1:N

        ex = @expression(m_master, AffExpr())

        y_opt = oracle_dict["y"]["$n"]
        π = oracle_dict["π"]["$n"]
        π_s = oracle_dict["π_s"]["$n"]
        x1 = oracle_dict["x1"]["$n"]
        x2 = oracle_dict["x2"]["$n"]
        x3 = oracle_dict["x3"]["$n"]
        x4 = oracle_dict["x4"]["$n"]
        x5 = oracle_dict["x5"]["$n"]
        x6 = oracle_dict["x6"]["$n"]
        const_term = oracle_dict["const"]["$n"]

        δ = norm(vcat(y_opt...) - y[n], 1) + norm(vcat(zstar...) - z[n], 1)

        γ = const_term + sum(π["($b, $t)"]*(D[b][t] - PredictionStar[ind_renewable_generation[b]][t] - y_opt[ind_renewable_generation[b]][t]) + π_s["($b, $t)"]*(PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t]) for b ∈ bus_renewable_generation, t ∈ T)

        for t in T

            for i in Ig

                add_to_expression!(ex, Pmax[i]*x1["($i, $t)"], m_master[:u][i, t])
                add_to_expression!(ex, Pmin[i]*x2["($i, $t)"], m_master[:u][i, t])

                if t == 1

                    add_to_expression!(ex, RU[i]*u0[i]*x5["($i, $t)"] + p0[i]*x5["($i, $t)"])
                    add_to_expression!(ex, Pmax[i]*x5["($i, $t)"], m_master[:v][i, t])

                    add_to_expression!(ex, RD[i]*x6["($i, $t)"], m_master[:u][i, t])
                    add_to_expression!(ex, Pmax[i]*x6["($i, $t)"], m_master[:w][i, t])
                    add_to_expression!(ex, -p0[i]*x6["($i, $t)"])

                else
                    add_to_expression!(ex, RU[i]*x3["($i, $t)"], m_master[:u][i, t-1])
                    add_to_expression!(ex, Pmax[i]*x3["($i, $t)"], m_master[:v][i, t])

                    add_to_expression!(ex, RD[i]*x4["($i, $t)"], m_master[:u][i, t])
                    add_to_expression!(ex, Pmax[i]*x4["($i, $t)"], m_master[:w][i, t])
                end
            end
        end

        @constraint(m_master, ex - δ*m_master[:λ] + γ ≤ m_master[:μ][n] + m_master[:θ])

        push!(cuts_Benders["$n"], (
            x1 = copy(x1),
            x2 = copy(x2),
            x3 = copy(x3),
            x4 = copy(x4),
            x5 = copy(x5),
            x6 = copy(x6),

            π = copy(π),
            π_s = copy(π_s),

            y_opt = vcat(y_opt...),

            const_term = const_term
        ))


    end
    
end

function deterministic_equivalent_problem(Data, y_list, PredictionStar)
    
    D = Data.D
    F = Data.F
    B = Data.B
    S = Data.S
    Tf = Data.Tf
    Tl = Data.Tl
    Cp = Data.Cp
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    Fmax = Data.Fmax
    Cdef = Data.Cdef
    number_of_generators = Data.Ig
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators
    Tb, = size(B) # Total buses
    L = collect(1:Tl) # transmission lines
    Bus = setdiff(collect(1:Tb), bus_renewable_generation) # Buses
    Cu = Data.Cu
    Csu = Data.Csu
    Csd = Data.Csd


    # Ignitial state
    p0 = zeros(number_of_generators)
    f0 = zeros(Tl)
    u0 = zeros(number_of_generators)

    m = length(y_list)

    model =  Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    @variable(model, cost_recourse[1:m])

    @variable(model, 0 <= v[1:Ig[end], 1:T[end]] <= 1) # start-up
    @variable(model, 0 <= w[1:Ig[end], 1:T[end]] <= 1) # shutdown
    @variable(model, u[1:Ig[end], 1:T[end]], Bin) # commit

    @constraint(model, [i in Ig, t in T[2:end]], v[i, t] - w[i, t] == u[i, t] - u[i, t-1])
    @constraint(model, [i in Ig, t in T], v[i, t] <= u[i, t])
    @constraint(model, [i in Ig, t in T], w[i, t] <= 1 - u[i, t])

    # Ignitial state constraint
    @constraint(model, [i in Ig, t in [1]], v[i, t] - w[i, t] == u[i, t] - u0[i])

    for k in 1:m

        y_opt = y_list[k]

        β = @variable(model, [1:Tb, 1:T[end]]) # phase
        f = @variable(model, [1:L[end], 1:T[end]]) # power flow
        p = @variable(model, [1:Ig[end], 1:T[end]], lower_bound = 0) # generation
        pdef = @variable(model, [1:Tb, 1:T[end]], lower_bound = 0) # deficit generation
        s = @variable(model, [bus_renewable_generation, 1:T[end]], lower_bound = 0) # spillage

        @constraint(model, [t in T], f[:, t] .== S*β[:, t])
        @constraint(model, [b in Bus, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] == D[b][t]) # Power balance
        @constraint(model, [b in Bus, t in T],  pdef[b, t] ≤ D[b][t]) 
        @constraint(model, [b in bus_renewable_generation, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] - s[b, t] == D[b][t] - maximum([PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0])) # Uncertainty
        @constraint(model, [b in bus_renewable_generation, t in T], pdef[b, t] ≤ D[b][t]) 
        @constraint(model, [b in bus_renewable_generation, t in T], s[b, t] <=  maximum([PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0])) # Uncertainty
        @constraint(model, [l in L, t in T], f[l,t] <= Fmax[l])
        @constraint(model, [l in L, t in T], f[l,t] >= -Fmax[l])
        @constraint(model, [i in Ig, t in T], p[i,t] <= Pmax[i]*u[i, t])
        @constraint(model, [i in Ig, t in T], p[i,t] >= Pmin[i]*u[i, t])

        @constraint(model, [i in Ig, t in T[2:end]], p[i, t] - p[i, t-1] <= RU[i]*u[i, t-1] + Pmax[i]*v[i, t])
        @constraint(model, [i in Ig, t in T[2:end]], p[i, t-1] - p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t])

        # Ignitial state constraints
        @constraint(model, [i in Ig, t in [1]], p[i, t] <= RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i])
        @constraint(model, [i in Ig, t in [1]], -p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i])

        @constraint(model, cost_recourse[k] == sum(Cp[i]*p[i,t] for i in Ig, t in T) + sum(Cdef*pdef[b,t] for b in 1:Tb, t in T) + sum(0.1*Cdef*s[b,t] for b in bus_renewable_generation, t in T))

    end

    @objective(model, Min, sum(Cu[i]*model[:u][i, t] for i in Ig, t in T) + sum(Csu[i]*model[:v][i, t] for i in Ig, t in T) +  sum(Csd[i]*model[:w][i, t] for i in Ig, t in T) + (1/m)*sum(cost_recourse[k] for k in 1:m))
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
        error(" Status: $(status)")
    end

    u = value.(model[:u])
    v = value.(model[:v])
    w = value.(model[:w])

    return u, v, w
   
end

function dual_vertices(Data, u, v, w, y_opt, PredictionStar) 

    D = Data.D
    Tf = Data.Tf
    T = collect(1:Tf)
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    number_of_generators = Data.Ig


    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)


    number_of_generators = Data.Ig
    Ig = collect(1:number_of_generators)
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation

    model = economic_dispatch_problem(Data, u, v, w)

    for b ∈ bus_renewable_generation, t ∈ T

        set_normalized_rhs(model[:π][b, t], D[b][t] - PredictionStar[ind_renewable_generation[b]][t] - y_opt[ind_renewable_generation[b]][t])
        set_normalized_rhs(model[:π_s][b, t], PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t])

    end
    m_oracle = dualize(model; dual_names = DualNames())
    set_optimizer(m_oracle, optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    optimize!(m_oracle)
    status = termination_status(m_oracle)
    if status != MOI.OPTIMAL
        error(" Status: $(status)")
    end

    π = Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T)
    π_s = Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T)
    x1 = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x2 = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x3 = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x4 = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x5 = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])
    x6 = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])
    const_term = 0.0

    obj_funcion = objective_function(m_oracle)

    const_term_x1, const_term_x2, const_term_x3, const_term_x4, const_term_x5, const_term_x6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for t ∈ T

        for b in bus_renewable_generation

            π["($b, $t)"] = value(m_oracle[Symbol("π[$b,$t]")])
            π_s["($b, $t)"] = value(m_oracle[Symbol("π_s[$b,$t]")])

        end

        for i in Ig

            x1["($i, $t)"] = value(m_oracle[Symbol("x1[$i,$t]")])
            x2["($i, $t)"] = value(m_oracle[Symbol("x2[$i,$t]")])
            const_term_x1 += Pmax[i]*u[i, t] * value(m_oracle[Symbol("x1[$i,$t]")])
            const_term_x2 += Pmin[i]*u[i, t] * value(m_oracle[Symbol("x2[$i,$t]")])

            if t == 1

                x5["($i, $t)"] = value(m_oracle[Symbol("x5[$i,$t]")])
                x6["($i, $t)"] = value(m_oracle[Symbol("x6[$i,$t]")])
                const_term_x5 += (RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i]) * value(m_oracle[Symbol("x5[$i,$t]")])
                const_term_x6 += (RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i]) * value(m_oracle[Symbol("x6[$i,$t]")])

            else

                x3["($i, $t)"] = value(m_oracle[Symbol("x3[$i,$t]")])
                x4["($i, $t)"] = value(m_oracle[Symbol("x4[$i,$t]")])
                const_term_x3 += (RU[i]*u[i, t-1] + Pmax[i]*v[i, t])* value(m_oracle[Symbol("x3[$i,$t]")])
                const_term_x4 += (RD[i]*u[i, t] + Pmax[i]*w[i, t]) * value(m_oracle[Symbol("x4[$i,$t]")])


            end

        end

    end


    const_term = objective_value(m_oracle) - (const_term_x1 + const_term_x2 + const_term_x3 + const_term_x4 + const_term_x5 + const_term_x6)

    return π, π_s, x1, x2, x3, x4, x5, x6, const_term, objective_value(m_oracle)

end

function initilization_restricted_master_Benders(Data, m_master, z, zstar, y, Ξ_y, PredictionStar)

    N = Data.N
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    Tf = Data.Tf   
    number_of_generators = Data.Ig 

    # Ignitial state
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)

    a = max.(Ξ_y[:,1], -vcat(PredictionStar...))
    b = Ξ_y[:,2]

    y_list = []
    idx = collect(1:N)

    for  n ∈ 1:N

        yproj = clamp.(y[n], a, b)
        push!(y_list, @views [yproj[((b - 1) * Tf + 1):(b * Tf)] for b in 1:length(bus_renewable_generation)]) 

    end   

    y_mean = mean(collect(y_list))

    u , v , w = deterministic_equivalent_problem(Data, [y_mean], PredictionStar)

    for n ∈ 1:N


        yproj = y_list[n]

        π, π_s, x1, x2, x3, x4, x5, x6, const_term = dual_vertices(Data, u, v, w, yproj, PredictionStar)

        ex = @expression(m_master, AffExpr())


        δ = norm(vcat(yproj...) - y[n], 1) +
            norm(vcat(zstar...) - z[n], 1)

        γ = const_term

        for t in T

            for i in Ig

                add_to_expression!(ex, Pmax[i]*x1["($i, $t)"], m_master[:u][i, t])
                add_to_expression!(ex, Pmin[i]*x2["($i, $t)"], m_master[:u][i, t])

                if t == 1

                    add_to_expression!(ex, RU[i]*u0[i]*x5["($i, $t)"] + p0[i]*x5["($i, $t)"])
                    add_to_expression!(ex, Pmax[i]*x5["($i, $t)"], m_master[:v][i, t])

                    add_to_expression!(ex, RD[i]*x6["($i, $t)"], m_master[:u][i, t])
                    add_to_expression!(ex, Pmax[i]*x6["($i, $t)"], m_master[:w][i, t])
                    add_to_expression!(ex, -p0[i]*x6["($i, $t)"])

                else
                    add_to_expression!(ex, RU[i]*x3["($i, $t)"], m_master[:u][i, t-1])
                    add_to_expression!(ex, Pmax[i]*x3["($i, $t)"], m_master[:v][i, t])

                    add_to_expression!(ex, RD[i]*x4["($i, $t)"], m_master[:u][i, t])
                    add_to_expression!(ex, Pmax[i]*x4["($i, $t)"], m_master[:w][i, t])
                end
            end
        end

       @constraint(m_master, ex - δ*m_master[:λ] + γ ≤ m_master[:μ][n] + m_master[:θ])

    end

    return nothing
 
end

function warm_start_solution_Benders(Data, m_master_warm, z, zstar_new, y, Ξ_y_new, PredictionStar_new,  cuts_Benders)

    a_new = max.(Ξ_y_new[:,1], -vcat(PredictionStar_new...))
    b_new = Ξ_y_new[:,2]    

    N = Data.N
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    number_of_generators = Data.Ig
    Tf = Data.Tf
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    D = Data.D

    # Ignitial state
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)

    for n ∈ 1:N

        for cut_data in cuts_Benders["$n"]

            ex = @expression(m_master_warm, AffExpr())

            x1 = cut_data.x1
            x2 = cut_data.x2
            x3 = cut_data.x3
            x4 = cut_data.x4
            x5 = cut_data.x5
            x6 = cut_data.x6

            π = cut_data.π
            π_s = cut_data.π_s

            yopt = cut_data.y_opt
            γ = cut_data.const_term

            yproj_adv = clamp.(yopt, a_new, b_new)
            y_opt = @views [yproj_adv[((b - 1) * Tf + 1):(b * Tf)] for b in 1:length(bus_renewable_generation)]

            γ += sum(π["($b, $t)"]*(D[b][t] - PredictionStar_new[ind_renewable_generation[b]][t] - y_opt[ind_renewable_generation[b]][t])+ π_s["($b, $t)"]*(PredictionStar_new[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t]) for b ∈ bus_renewable_generation, t ∈ T)

            δ = norm(yproj_adv - y[n], 1) + norm(vcat(zstar_new...) - z[n], 1)

            for t in T

                for i in Ig

                    add_to_expression!(ex, Pmax[i]*x1["($i, $t)"], m_master_warm[:u][i, t])
                    add_to_expression!(ex, Pmin[i]*x2["($i, $t)"], m_master_warm[:u][i, t])

                    if t == 1

                        add_to_expression!(ex, RU[i]*u0[i]*x5["($i, $t)"] + p0[i]*x5["($i, $t)"])
                        add_to_expression!(ex, Pmax[i]*x5["($i, $t)"], m_master_warm[:v][i, t])

                        add_to_expression!(ex, RD[i]*x6["($i, $t)"], m_master_warm[:u][i, t])
                        add_to_expression!(ex, Pmax[i]*x6["($i, $t)"], m_master_warm[:w][i, t])
                        add_to_expression!(ex, -p0[i]*x6["($i, $t)"])

                    else
                        add_to_expression!(ex, RU[i]*x3["($i, $t)"], m_master_warm[:u][i, t-1])
                        add_to_expression!(ex, Pmax[i]*x3["($i, $t)"], m_master_warm[:v][i, t])

                        add_to_expression!(ex, RD[i]*x4["($i, $t)"], m_master_warm[:u][i, t])
                        add_to_expression!(ex, Pmax[i]*x4["($i, $t)"], m_master_warm[:w][i, t])
                    end
                end
                
            end

            @constraint(m_master_warm, ex - δ*m_master_warm[:λ] + γ ≤ m_master_warm[:μ][n] + m_master_warm[:θ])

        end

    end

    val, time_ = @timed(begin
        optimize!(m_master_warm)
        status = termination_status(m_master_warm)
        if status != MOI.OPTIMAL
            error("problem $status")
        end
    end)

    u_warm = value.(m_master_warm[:u])
    v_warm = value.(m_master_warm[:v])
    w_warm = value.(m_master_warm[:w])
    λ_warm = value.(m_master_warm[:λ])
    θ_warm = value(m_master_warm[:θ])
    μ_warm = value.(m_master_warm[:μ])

    return u_warm, v_warm, w_warm, λ_warm, μ_warm, θ_warm, objective_value(m_master_warm), time_
 
end

function solve_master_problem(m_master, iter)

    master_start = time()

    put!(PROGRESS_CHANNEL, Dict(
        "type" => "MASTER",
        "iteration" => iter,
        "state" => "RUNNING",
        "start" => string(now())
    ))

    val, time_master = @timed(begin
        optimize!(m_master)
        status = termination_status(m_master)
        if status != MOI.OPTIMAL
            error("problem $status")
        end
    end) 

    put!(PROGRESS_CHANNEL, Dict(
    "type" => "MASTER",
    "iteration" => iter,
    "state" => "COMPLETED",
    "finish" => string(now()),
    "duration_seconds" => round(time() - master_start, digits=2)
))

    # Solution
    u = value.(m_master[:u])
    v = value.(m_master[:v])
    w = value.(m_master[:w])
    λ = value.(m_master[:λ])

    μ = value.(m_master[:μ])
    θ = value(m_master[:θ])

    return u, v, w, λ, μ, θ, objective_value(m_master), time_master
    
end

function critical_radius(Data, errors, Ξ_y, α, zstar; p = 1)

    N = Data.N
    z_aux = Data.z
    Tf = Data.Tf
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    Tb = length(bus_renewable_generation)

    global_residual_scenarios = zeros(Tf * Tb, N)
    global_contextual_scenarios =  zeros(Tf * Tb, N)

    for b in bus_renewable_generation
        b_ = ind_renewable_generation[b]
        row_range = ((b_ - 1) * Tf + 1):(b_ * Tf)
    
        for n in 1:N
            vector_residual_24h = errors[b_][n] 
            vector_contextual_24h = z_aux[b_][n]
            global_residual_scenarios[row_range, n] = vector_residual_24h
            global_contextual_scenarios[row_range, n] = vector_contextual_24h
        end
    end

    zstar_global = vcat(zstar...)
    y = [s[:] for s in eachcol(global_residual_scenarios)]
    z = [s[:] for s in eachcol(global_contextual_scenarios)]

    a = Ξ_y[:,1]
    b = Ξ_y[:,2]

    dist = zeros(N)

    for n in 1:N

        dz = norm(z[n] - zstar_global, 1)

        dy = 0.0
        for d in eachindex(a)
            if y[n][d] < a[d]
                dy += a[d] - y[n][d]
            elseif y[n][d] > b[d]
                dy += y[n][d] - b[d]
            end
        end

        dist[n] = dz + dy
    end

    sort!(dist)

    if α == 0
        return dist[1]
    end

    m = floor(Int, N*α)
    frac = N*α - m

    value = sum(dist[k] for k in 1:m)

    if frac > 0
        value += frac*dist[m+1]
    end

    return value/(N*α)

end

function CVaR(α, vec)

    S = collect(1:length(vec))
    p = (1/length(vec))*ones(length(vec))

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    @variable(model, θ)
    @variable(model, z[s in S] >= 0)

    @constraint(model, [s in S],
    z[s] >= vec[s] - θ)

    @objective(model, Min, θ + (1/α) * sum(p[s] * z[s] for s in S))

    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
        error("problem $status")
    end

    CVaR_value = objective_value(model)

    return CVaR_value

end

function reconstruct_sub_dict(dict, values_tuple)

    sub_dict = Dict{String, Float64}()
    for (idx, key) in enumerate(collect(keys(dict)))
        sub_dict[key] = values_tuple[idx]
    end

    return sub_dict

end

function Upper_Bound(Data, u, v, w, λ, vec, ρ,  α)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators

    Cu = Data.Cu
    Csu = Data.Csu
    Csd = Data.Csd

    UB = sum(Cu[i]*u[i, t] for i in Ig, t in T) + sum(Csu[i]*v[i, t] for i in Ig, t in T) +  sum(Csd[i]*w[i, t] for i in Ig, t in T) + ρ*λ + CVaR(α, vec)

    return UB

end

zstar = Data.zstar

Models, errors = scenarios(Data)

Ξ_y = uncertainty_support(Data, zstar, errors)

PredictionStar = forecast(Data, zstar, Models)


# Trimming parameters

α = 0.2
ρ = 1.1*critical_radius(Data, errors, Ξ_y, α, zstar) 


N = Data.N
z_aux = Data.z
bus_renewable_generation = Data.bus_renewable_generation
ind_renewable_generation = Data.ind_renewable_generation
Tb_renewable = length(bus_renewable_generation)
number_of_generators = Data.Ig
Tf = Data.Tf
Ig = collect(1:number_of_generators)
T = collect(1:Tf)
sample = collect(1:N)

global_residual_scenarios = zeros(Tf * Tb_renewable, N)
global_contextual_scenarios =  zeros(Tf * Tb_renewable, N)

for b in bus_renewable_generation

    b_ = ind_renewable_generation[b]
    row_range = ((b_ - 1) * Tf + 1):(b_ * Tf)

    for n in 1:N
        vector_residual_24h = errors[b_][n] 
        vector_contextual_24h = z_aux[b_][n]
        global_residual_scenarios[row_range, n] = vector_residual_24h
        global_contextual_scenarios[row_range, n] = vector_contextual_24h
    end

end

y = [s[:] for s in eachcol(global_residual_scenarios)]
z = [s[:] for s in eachcol(global_contextual_scenarios)]


oracle_dict = Dict("y" => Dict("$n" => [zeros(Tf) for b ∈ collect(1:Tb_renewable)] for n ∈ sample), "π" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "π_s" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "x1" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x2" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x3" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x4" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x5" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "x6" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "const" => Dict("$n" => 0.0 for n ∈ sample), "time" => Dict("$n" => 0.0 for n ∈ sample), "ϑ" => zeros(N))
cuts_Benders = Dict("$n" => NamedTuple[] for n in sample)
iter = 1

LowerBound = []
UpperBound = []
Time = []

m_master = create_master_problem(Data, ρ)
initilization_restricted_master_Benders(Data, m_master, z, zstar, y, Ξ_y, PredictionStar)
convergence = false
Max_iter = 400
cumulative_time = 0.0
UB_Benders = Inf
LB_Benders = -Inf

master_solution = (
    u = zeros(Ig[end], T[end]),
    v = zeros(Ig[end], T[end]),
    w = zeros(Ig[end], T[end]),
    λ = 0.0,
    θ = 0.0,
    μ = zeros(N)
)

master_solution_dict = Dict("$iter" => master_solution for iter ∈ 1:Max_iter)

gap = Inf

u_previous = nothing
v_previous = nothing
w_previous = nothing


delta_u = NaN
delta_v = NaN
delta_w = NaN

first_stage_history = NamedTuple[]

convergence = false

violation = fill(true, N)

while ~convergence && iter ≤ Max_iter  && gap ≥ 1e-3

    global delta_u, delta_v, delta_w
    global u_previous, v_previous, w_previous
    global master_solution, convergence
    global LB_Benders, UB_Benders
    global cumulative_time, iter, gap

    global u_Benders, v_Benders, w_Benders, λ_Benders, μ_Benders, θ_Benders, LB_Benders_, time_master_Benders = solve_master_problem(m_master, iter)

    u_current = Dict(
        (i,t) => u_Benders[i, t]
        for i in Ig, t in T
    )

    v_current = Dict(
        (i,t) => v_Benders[i, t]
        for i in Ig, t in T
    )

    w_current = Dict(
        (i,t) => w_Benders[i, t]
        for i in Ig, t in T
    )

    if u_previous !== nothing

        delta_u = sum(
            abs(round(u_current[(i,t)]) - u_previous[(i,t)])
            for i in Ig, t in T
        )

        delta_v = sum(
            abs(round(v_current[(i,t)]) - v_previous[(i,t)])
            for i in Ig, t in T
        )

        delta_w = sum(
            abs(round(w_current[(i,t)]) - w_previous[(i,t)])
            for i in Ig, t in T
        )

    end

    u_previous = Dict(k => round(val) for (k,val) in u_current)
    v_previous = Dict(k => round(val) for (k,val) in v_current)
    w_previous = Dict(k => round(val) for (k,val) in w_current)

    push!(first_stage_history, (
        iteration = iter,

        delta_u = delta_u,
        delta_v = delta_v,
        delta_w = delta_w
    ))

    master_solution = (
        u = round.(Int, u_Benders),
        v = v_Benders,
        w = w_Benders,
        λ = λ_Benders,
        θ = θ_Benders,
        μ = μ_Benders
    )
    oracle_data = (
        Data = Data,
        z = z,
        y = y,
        Ξ_y = Ξ_y,
        u = u_Benders,
        v = v_Benders,
        w = w_Benders,
        λ = λ_Benders,
        μ = μ_Benders,
        θ = θ_Benders,
        oracle_dict = oracle_dict,
        zstar = zstar,
        PredictionStar = PredictionStar
    )
    global result = pmap(
        n -> solve_oracle_problem(oracle_data, iter, n),
        1:N
    )

    for n ∈ 1:N 
        violation[n] = result[n].oracle_solution.violation
    end

    if sum(violation) < N 
        print("iteration = $iter, feasibility for $(N - sum(violation)) subproblems\n")
    end

    convergence = ~Base.any(violation)

    master_solution_dict["$iter"] = master_solution
    fill_oracle_dict(Data, result, oracle_dict)
    LB_Benders = max(LB_Benders, LB_Benders_)
    UB_Benders = minimum([UB_Benders, Upper_Bound(Data, u_Benders, v_Benders, w_Benders, λ_Benders, oracle_dict["ϑ"], ρ,  α)])
    update_master_Benders(Data, m_master, z, zstar, PredictionStar, y, oracle_dict, cuts_Benders)
    total_time_oracle = sum(oracle_dict["time"]["$n"][end] for n ∈ sample)
    cumulative_time += time_master_Benders + total_time_oracle
    total_time = time_master_Benders + total_time_oracle
    println("================================================")
    println(" Lower Bound: $LB_Benders\n") 
    println(" Upper Bound: $UB_Benders\n") 
    println(" λ: $λ_Benders\n") 
    println(" θ: $θ_Benders\n") 
    println(" μ: $(μ_Benders')\n") 
    println(" Iteration: $iter")
    println("================================================") 
    iter +=  1
    push!(LowerBound, LB_Benders)
    push!(UpperBound, UB_Benders)
    push!(Time,(time_master_Benders, total_time_oracle,total_time))

    gap = (UB_Benders - LB_Benders)/UB_Benders

    put!(PROGRESS_CHANNEL, Dict(
        "type" => "BENDERS",
        "iteration" => iter,
        "LB" => LB_Benders,
        "UB" => UB_Benders,
        "gap" => gap
    ))

end

df_first_stage = DataFrame(first_stage_history)

CSV.write("results/first_stage_changes.csv", df_first_stage)
jldsave("results/LowerBound.jld2"; LowerBound)
jldsave("results/UpperBound.jld2"; UpperBound)
jldsave("results/cuts_Benders.jld2"; cuts_Benders)
jldsave("results/master_solution_dict.jld2"; master_solution_dict)
jldsave("results/cumulative_time.jld2"; cumulative_time)
jldsave("results/m_master.jld2"; m_master)
jldsave("results/iterations.jlds"; iter)

println("Benders converged...")

put!(PROGRESS_CHANNEL, Dict(
    "type" => "STEP",
    "stage" => "Warm Start",
    "iteration" => 0
))

# New context

# cuts_Benders = load("results/cuts_Benders.jld2", "cuts_Benders")
ϵ = 0.1
zstar_new_ = vcat(zstar...) .+ (rand(length(vcat(zstar...))) .- 0.5) .* (2 * ϵ)
zstar_new = @views [zstar_new_[((b - 1) * Tf + 1):(b * Tf)] for b in 1:length(bus_renewable_generation)]
jldsave("results/zstar_new.jld2"; zstar_new)

Ξ_y_new = uncertainty_support(Data, zstar_new, errors)

PredictionStar_new = forecast(Data, zstar_new, Models)
jldsave("results/PredictionStar_new.jld2"; PredictionStar_new)

ρ_new = 1.1*critical_radius(Data, errors, Ξ_y_new, α, zstar_new) 

oracle_dict_warm = Dict("y" => Dict("$n" => [zeros(Tf) for b ∈ collect(1:Tb_renewable)] for n ∈ sample), "π" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "π_s" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "x1" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x2" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x3" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x4" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x5" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "x6" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "const" => Dict("$n" => 0.0 for n ∈ sample), "time" => Dict("$n" => 0.0 for n ∈ sample), "ϑ" => zeros(N))
cuts_Benders_warm = Dict("$n" => NamedTuple[] for n in sample)

iter_warm = 1

LowerBound_warm = []
UpperBound_warm = []
Time_warm = []
cuts_warm = Dict("$n" => [] for n ∈ sample)

m_master_warm = create_master_problem(Data, ρ_new)
initilization_restricted_master_Benders(Data, m_master_warm, z, zstar_new, y, Ξ_y_new, PredictionStar_new)
Max_iter = 350
cumulative_time_warm = 0.0
UB_Benders_warm= Inf

master_solution_warm = (
    u = zeros(Ig[end], T[end]),
    v = zeros(Ig[end], T[end]),
    w = zeros(Ig[end], T[end]),
    λ = 0.0,
    θ = 0.0,
    μ = zeros(N)
)

master_solution_dict_warm = Dict("$iter" => master_solution_warm for iter ∈ 1:Max_iter)

gap_warm = Inf

u_previous_warm = nothing
v_previous_warm = nothing
w_previous_warm = nothing


delta_u_warm = NaN
delta_v_warm = NaN
delta_w_warm = NaN

first_stage_history_warm = NamedTuple[]

convergence_warm = false

violation_warm = fill(true, N)

while ~convergence_warm &&  iter_warm ≤ Max_iter && gap_warm ≥1e-3

    global delta_u_warm, delta_v_warm, delta_w_warm
    global u_previous_warm, v_previous_warm, w_previous_warm
    global master_solution_warm, convergence_warm
    global LB_Benders_warm, UB_Benders_warm
    global cumulative_time_warm, iter_warm, gap_warm

    if iter_warm == 1
        global u_Benders_warm, v_Benders_warm, w_Benders_warm, λ_Benders_warm, μ_Benders_warm, θ_Benders_warm, LB_Benders_warm, time_master_Benders_warm = warm_start_solution_Benders(Data, m_master_warm, z, zstar_new, y, Ξ_y_new, PredictionStar_new,  cuts_Benders)
    else                                                                
        global u_Benders_warm, v_Benders_warm, w_Benders_warm, λ_Benders_warm, μ_Benders_warm, θ_Benders_warm, LB_Benders_warm, time_master_Benders_warm = solve_master_problem(m_master_warm, iter_warm)
    end

    u_current_warm = Dict(
        (i,t) => u_Benders_warm[i, t]
        for i in Ig, t in T
    )

    v_current_warm = Dict(
        (i,t) => v_Benders_warm[i, t]
        for i in Ig, t in T
    )

    w_current_warm = Dict(
        (i,t) => w_Benders_warm[i, t]
        for i in Ig, t in T
    )

    if u_previous_warm !== nothing

        delta_u_warm = sum(
            abs(round(u_current_warm[(i,t)]) - u_previous_warm[(i,t)])
            for i in Ig, t in T
        )

        delta_v_warm = sum(
            abs(round(v_current_warm[(i,t)]) - v_previous_warm[(i,t)])
            for i in Ig, t in T
        )

        delta_w_warm = sum(
            abs(round(w_current_warm[(i,t)]) - w_previous_warm[(i,t)])
            for i in Ig, t in T
        )

    end

    u_previous_warm = Dict(k => round(val) for (k,val) in u_current_warm)
    v_previous_warm = Dict(k => round(val) for (k,val) in v_current_warm)
    w_previous_warm = Dict(k => round(val) for (k,val) in w_current_warm)

    push!(first_stage_history_warm, (
        iteration = iter_warm,

        delta_u = delta_u_warm,
        delta_v = delta_v_warm,
        delta_w = delta_w_warm
    ))

    master_solution_warm = (
        u = round.(Int, u_Benders_warm),
        v = v_Benders_warm,
        w = w_Benders_warm,
        λ = λ_Benders_warm,
        θ = θ_Benders_warm,
        μ = μ_Benders_warm
    )
    oracle_data = (
        Data = Data,
        z = z,
        y = y,
        Ξ_y = Ξ_y_new,
        u = u_Benders_warm,
        v = v_Benders_warm,
        w = w_Benders_warm,
        λ = λ_Benders_warm,
        μ = μ_Benders_warm,
        θ = θ_Benders_warm,
        oracle_dict = oracle_dict_warm,
        zstar = zstar_new,
        PredictionStar = PredictionStar_new
    )
    global result = pmap(
        n -> solve_oracle_problem(oracle_data, iter_warm, n),
        1:N
    )

    for n ∈ 1:N 
        violation_warm[n] = result[n].oracle_solution.violation
    end

    if sum(violation_warm) < N 
        print("iteration = $iter_warm, feasibility for $(N - sum(violation_warm)) subproblems\n")
    end

    master_solution_dict_warm["$iter_warm"] = master_solution_warm
    fill_oracle_dict(Data, result, oracle_dict_warm)

    UB_Benders_warm = minimum([UB_Benders_warm, Upper_Bound(Data, u_Benders_warm, v_Benders_warm, w_Benders_warm, λ_Benders_warm, oracle_dict_warm["ϑ"], ρ_new, α)])
    update_master_Benders(Data, m_master_warm, z, zstar_new, PredictionStar_new, y, oracle_dict_warm, cuts_Benders_warm)
    total_time_oracle_warm = sum(oracle_dict_warm["time"]["$n"][end] for n ∈ sample)
    cumulative_time_warm += time_master_Benders_warm + total_time_oracle_warm
    total_time_warm = time_master_Benders_warm + total_time_oracle_warm
    println("================================================")
    println(" Lower Bound: $LB_Benders_warm\n") 
    println(" Upper Bound: $UB_Benders_warm\n") 
    println(" λ: $λ_Benders_warm\n") 
    println(" θ: $θ_Benders_warm\n") 
    println(" μ: $(μ_Benders_warm')\n") 
    println(" Iteration: $iter_warm")
    println("================================================") 
    push!(LowerBound_warm, LB_Benders_warm)
    push!(UpperBound_warm, UB_Benders_warm)
    push!(Time_warm, (time_master_Benders_warm, total_time_oracle_warm, total_time_warm))

    convergence_warm = ~Base.any(violation_warm)
    gap_warm = ( UB_Benders_warm - LB_Benders_warm)/UB_Benders_warm

    put!(PROGRESS_CHANNEL, Dict(
        "type" => "WARM_BENDERS",
        "iteration" => iter_warm,
        "LB" => LB_Benders_warm,
        "UB" => UB_Benders_warm,
        "gap" => gap_warm
    ))

    iter_warm +=  1

end

df_first_stage_warm = DataFrame(first_stage_history_warm)

CSV.write("results/first_stage_changes_warm.csv", df_first_stage_warm)
jldsave("results/LowerBound_warm.jld2"; LowerBound_warm)
jldsave("results/UpperBound_warm.jld2"; UpperBound_warm)
jldsave("results/cuts_Benders_warm.jld2"; cuts_Benders_warm)
jldsave("results/master_solution_dict_warm.jld2"; master_solution_dict_warm)
jldsave("results/cumulative_time_warm.jld2"; cumulative_time_warm)
jldsave("results/iterations_warm.jlds"; iter_warm)


println("Benders with warm star converged...")

put!(PROGRESS_CHANNEL, Dict(
    "type" => "STEP",
    "stage" => "Without Warm Start",
    "iteration" => 0
))

oracle_dict_new = Dict("y" => Dict("$n" => [zeros(Tf) for b ∈ collect(1:Tb_renewable)] for n ∈ sample), "π" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "π_s" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "x1" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x2" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x3" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x4" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x5" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "x6" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "const" => Dict("$n" => 0.0 for n ∈ sample), "time" => Dict("$n" => 0.0 for n ∈ sample), "ϑ" => zeros(N))
cuts_Benders_new = Dict("$n" => NamedTuple[] for n in sample)

iter_new = 1

LowerBound_new = []
UpperBound_new = []
Time_new = []
cuts_new = Dict("$n" => [] for n ∈ sample)

m_master_new = create_master_problem(Data, ρ_new)
initilization_restricted_master_Benders(Data, m_master_new, z, zstar_new, y, Ξ_y_new, PredictionStar_new)
Max_iter = 350
cumulative_time_new = 0.0
UB_Benders_new= Inf

master_solution_new = (
    u = zeros(Ig[end], T[end]),
    v = zeros(Ig[end], T[end]),
    w = zeros(Ig[end], T[end]),
    λ = 0.0,
    θ = 0.0,
    μ = zeros(N)
)

master_solution_dict_new = Dict("$iter" => master_solution_new for iter ∈ 1:Max_iter)

gap_new = Inf

u_previous_new = nothing
v_previous_new = nothing
w_previous_new = nothing


delta_u_new = NaN
delta_v_new = NaN
delta_w_new = NaN

first_stage_history_new = NamedTuple[]

convergence_new = false

violation_new = fill(true, N)

while ~convergence_new &&  iter_new ≤ Max_iter && gap_new ≥ 1e-3

    global delta_u_new, delta_v_new, delta_w_new
    global u_previous_new, v_previous_new, w_previous_new
    global master_solution_new, convergence_new
    global LB_Benders_new, UB_Benders_new
    global cumulative_time_new, iter_new, gap_new

    global u_Benders_new, v_Benders_new, w_Benders_new, λ_Benders_new, μ_Benders_new, θ_Benders_new, LB_Benders_new, time_master_Benders_new = solve_master_problem(m_master_new, iter_new)

    u_current_new = Dict(
        (i,t) => u_Benders_new[i, t]
        for i in Ig, t in T
    )

    v_current_new = Dict(
        (i,t) => v_Benders_new[i, t]
        for i in Ig, t in T
    )

    w_current_new = Dict(
        (i,t) => w_Benders_new[i, t]
        for i in Ig, t in T
    )

    if u_previous_new !== nothing

        delta_u_new = sum(
            abs(round(u_current_new[(i,t)]) - u_previous_new[(i,t)])
            for i in Ig, t in T
        )

        delta_v_new = sum(
            abs(round(v_current_new[(i,t)]) - v_previous_new[(i,t)])
            for i in Ig, t in T
        )

        delta_w_new = sum(
            abs(round(w_current_new[(i,t)]) - w_previous_new[(i,t)])
            for i in Ig, t in T
        )

    end

    u_previous_new = Dict(k => round(val) for (k,val) in u_current_new)
    v_previous_new = Dict(k => round(val) for (k,val) in v_current_new)
    w_previous_new = Dict(k => round(val) for (k,val) in w_current_new)

    push!(first_stage_history_new, (
        iteration = iter_new,

        delta_u = delta_u_new,
        delta_v = delta_v_new,
        delta_w = delta_w_new
    ))

    master_solution_new = (
        u = round.(Int, u_Benders_new),
        v = v_Benders_new,
        w = w_Benders_new,
        λ = λ_Benders_new,
        θ = θ_Benders_new,
        μ = μ_Benders_new
    )
    oracle_data = (
        Data = Data,
        z = z,
        y = y,
        Ξ_y = Ξ_y_new,
        u = u_Benders_new,
        v = v_Benders_new,
        w = w_Benders_new,
        λ = λ_Benders_new,
        μ = μ_Benders_new,
        θ = θ_Benders_new,
        oracle_dict = oracle_dict_new,
        zstar = zstar_new,
        PredictionStar = PredictionStar_new
    )
    global result = pmap(
        n -> solve_oracle_problem(oracle_data, iter_new, n),
        1:N
    )

    for n ∈ 1:N 
        violation_new[n] = result[n].oracle_solution.violation
    end

    if sum(violation_new) < N 
        print("iteration = $iter_new, feasibility for $(N - sum(violation_new)) subproblems\n")
    end

    master_solution_dict_new["$iter_new"] = master_solution_new
    fill_oracle_dict(Data, result, oracle_dict_new)

    UB_Benders_new = minimum([UB_Benders_new, Upper_Bound(Data, u_Benders_new, v_Benders_new, w_Benders_new, λ_Benders_new, oracle_dict_new["ϑ"], ρ_new, α)])
    update_master_Benders(Data, m_master_new, z, zstar_new, PredictionStar_new, y, oracle_dict_new, cuts_Benders_new)
    total_time_oracle_new = sum(oracle_dict_new["time"]["$n"][end] for n ∈ sample)
    cumulative_time_new += time_master_Benders_new + total_time_oracle_new
    total_time_new = time_master_Benders_new + total_time_oracle_new
    println("================================================")
    println(" Lower Bound: $LB_Benders_new\n") 
    println(" Upper Bound: $UB_Benders_new\n") 
    println(" λ: $λ_Benders_new\n") 
    println(" θ: $θ_Benders_new\n") 
    println(" μ: $(μ_Benders_new')\n") 
    println(" Iteration: $iter_new")
    println("================================================") 
    push!(LowerBound_new, LB_Benders_new)
    push!(UpperBound_new, UB_Benders_new)
    push!(Time_new, (time_master_Benders_new, total_time_oracle_new, total_time_new))

    convergence_new = ~Base.any(violation_new)
    gap_new =( UB_Benders_new - LB_Benders_new)/UB_Benders_new

    put!(PROGRESS_CHANNEL, Dict(
        "type" => "BENDERS_WITHOUT_WARM_START",
        "iteration" => iter_new,
        "LB" => LB_Benders_new,
        "UB" => UB_Benders_new,
        "gap" => gap_new
    ))

    iter_new +=  1

end

df_first_stage_new = DataFrame(first_stage_history_new)

CSV.write("results/first_stage_changes_new.csv", df_first_stage_new)
jldsave("results/LowerBound_new.jld2"; LowerBound_new)
jldsave("results/UpperBound_new.jld2"; UpperBound_new)
jldsave("results/cuts_Benders_new.jld2"; cuts_Benders_new)
jldsave("results/master_solution_dict_new.jld2"; master_solution_dict_new)
jldsave("results/cumulative_time_new.jld2"; cumulative_time_new)
jldsave("results/iterations_new.jlds"; iter_new)

println("Benders without warm star converged...")

put!(PROGRESS_CHANNEL, Dict(
    "type" => "STEP",
    "stage" => "COMPLETED",
    "iteration" => 0
))

put!(PROGRESS_CHANNEL, Dict(
    "type" => "TERMINATE"
))

wait(monitor_task)