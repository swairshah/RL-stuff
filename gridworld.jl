using Distributions
using Random: AbstractRNG

abstract type AbstractSpace end
abstract type AbstractDiscreteSpace <: AbstractSpace end
struct DiscreteSpace{T<:Integer} <: AbstractDiscreteSpace
    low::T
    high::T
    n::T
    function DiscreteSpace(high::T, low=one(T)) where {T <: Integer}
        high >= low || throw(ArgumentError("$high must be >= $low"))
        new{T}(low, high, high - low + 1)
    end
end
Base.length(s::DiscreteSpace) = s.n
Base.eltype(s::DiscreteSpace{T}) where {T} = T
Base.in(x, s::DiscreteSpace{T}) where {T} = s.low <= x <= s.high
Base.:(==)(s1::DiscreteSpace, s2::DiscreteSpace) = s1.low == s2.low && s1.high == s2.high
Base.rand(rng::AbstractRNG, s::DiscreteSpace) = rand(rng, s.low:s.high)



abstract type  AbstractEnvironmentModel end
abstract type  AbstractDistributionBasedModel <: AbstractEnvironmentModel end
struct DeterministicDistributionModel <: AbstractDistributionBasedModel
    table::Array{
        Vector{NamedTuple{(:nextstate, :reward, :prob),Tuple{Int,Float64,Float64}}},
        2,
    }
end
observation_space(m::DeterministicDistributionModel) = DiscreteSpace(size(m.table, 1))
action_space(m::DeterministicDistributionModel) = DiscreteSpace(size(m.table, 2))

(m::DeterministicDistributionModel)(s::Int, a::Int) = m.table[s, a]
