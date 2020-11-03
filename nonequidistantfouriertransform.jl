using Base.Threads
using CUDA
using DelimitedFiles

import LinearAlgebra

function nedft(x :: Array{Float64, 1}, F :: Array{ComplexF64, 1})
    N = length(F)
    if length(x) != N
        error("Length mismatch")
    end
    
    f = zeros(ComplexF64, N)

    for j = 1:N
        for k = 1:N
            @inbounds f[j] += F[k] * exp(-2*π*1im*x[j]*(k-1-N/2))
        end
    end

    f
end

function nedft_simpl(x :: Array{Float64, 1}, F :: Array{ComplexF64, 1})
    N = length(F)
    if length(x) != N
        error("Length mismatch")
    end
    
    f = zeros(ComplexF64, N)

    for j = 1:N
        e_j = exp(-2*π*1im*x[j])^(-N/2)
        for k = N:-1:1
            @inbounds f[j] *= e_j
            @inbounds f[j] += F[k]
        end
    end

    f
end

function nedft_outer_threaded(x :: Array{Float64, 1}, F :: Array{ComplexF64, 1})
    N = length(F)
    if length(x) != N
        error("Length mismatch")
    end
    
    f = zeros(ComplexF64, N)

    @threads for j = 1:N
        for k = 1:N
            @inbounds f[j] += F[k] * exp(-2*π*1im*x[j]*(k-1-N/2))
        end
    end

    f
end

# GEFAHR --- Da Threads asynchron funktioniert Horner-style Multiplikation nicht, wenn man die k's aufteilt!!!
function nedft_inner_threaded(x :: Array{Float64, 1}, F :: Array{ComplexF64, 1})
    N = length(F)
    if length(x) != N
        error("Length mismatch")
    end
    
    f = zeros(ComplexF64, N)

    for j = 1:N
        @threads for k = 1:N
            @inbounds f[j] += F[k] * exp(-2*π*1im*x[j]*(k-1-N/2))
        end
    end

    f
end

function nedft_horner_threaded(x :: Array{Float64, 1}, F :: Array{ComplexF64, 1})
    N = length(F)
    
    if length(x) != N
        error("Length mismatch")
    end
    
    f :: Array{ComplexF64, 1} = zeros(ComplexF64, N)

    @threads for j = 1:N
        e_j = exp(-2*π*1im*x[j])
        for k = N:-1:1
            @inbounds f[j] = f[j] * e_j + F[k]
        end
        f[j] = f[j] * e_j^(-N/2) 
    end

    f
end

function nedft_cuda(x :: Array{Float64, 1}, F :: Array{ComplexF64, 1})
    N = length(x)
    
    # Copy data to GPU
    F_d = CuArray(F)
    x_d = CuArray(x)
    f_d = CUDA.zeros(ComplexF64, N)
    e_d = CUDA.zeros(Float64, N)

    CUDA.copyto!(e_d, x_d)
    
    s = -2*π*1im
    
    e_d = broadcast(CUDA.exp, s.*e_d)

    for k = N:-1:1
        CUDA.fill!(F_d, F[k])
        f_d .= f_d .* e_d .+ F_d
    end
    CUDA.copyto!(e_d, x_d)
    e_d = broadcast(CUDA.exp, (-N/2)*s.*e_d)
    f_d =  f_d .* e_d

    # f_d von GPU 
    Array(f_d)
end

# Precompile funcs
@time begin
    local N = 2^2                        # Length of input vector
    local F = rand(ComplexF64, N)        # Fourier Coefficients
    local x = rand(Float64, N) .- 0.5    # Sample points (-.5, .5]
    
    nedft(x,F)
    nedft_simpl(x, F)
    nedft_basic_threaded(x, F)
    nedft_horner_threaded(x, F)
    nedft_cuda(x, F)                     # Vorallem hier viel Overhead
end;

max_exp = 17
data = zeros(6,max_exp)

for exponent = 1:max_exp
    local N = 2^exponent                 # Length of input vector
    local F = rand(ComplexF64, N)        # Fourier Coefficients
    local x = rand(Float64, N) .- 0.5    # Sample points (-.5, .5]

    data[1, exponent] = @elapsed nedft(x,F)
    data[2, exponent] = @elapsed nedft_simpl(x, F)
    data[3, exponent] = @elapsed nedft_outer_threaded(x, F)
    data[4, exponent] = @elapsed nedft_horner_threaded(x, F)
    data[5, exponent] = @elapsed nedft_cuda(x, F)
end

# Daten als CSV speichern - eine Reihe pro Methode
writedlm("results.csv", vcat(["standard", "horner", "threaded", "threaded-horner", "cuda"], data), ",\t")