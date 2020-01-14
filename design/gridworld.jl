using Random: AbstractRNG

# State Space Representation
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

# Environment
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

struct Observation{R,T,S,M<:NamedTuple}
    reward::R
    terminal::T
    state::S
    meta::M
end
Observation(; reward, terminal, state, kw...) =
    Observation(reward, terminal, state, merge(NamedTuple(), kw))
get_reward(obs::Observation) = obs.reward
get_terminal(obs::Observation) = obs.terminal
get_state(obs::Observation) = obs.state
get_legal_actions(obs::Observation) = obs.meta.legal_actions

# Value Stuff
abstract type AbstractApproximator end
abstract type AbstractVApproximator <: AbstractApproximator end
struct TabularVApproximator <: AbstractVApproximator
    table::Vector{Float64}
end
TabularVApproximator(ns::Int64, init::Float64 = 0.0) = TabularVApproximator(fill(init, ns))
(v::TabularVApproximator)(s::Int) = v.table[s]
function update!(v::TabularVApproximator, correction::Pair{Int, Float64})
    s, e = correction
    v.table[s] += e
end

# Policy Stuff
abstract type AbstractPolicy end
struct TabularRandomPolicy <: AbstractPolicy
    prob::Array{Float64, 2}
end
(π::TabularRandomPolicy)(s) = sample(Weights(π.prob[s, :]))
(π::TabularRandomPolicy)(obs::Observation) = π(get_state(obs))
get_prob(π::TabularRandomPolicy, s) = @view π.prob[s, :]
get_prob(π::TabularRandomPolicy, s, a) = π.prob[s, a]

function policy_evaluation!(
    ;
    V::AbstractVApproximator,
    π::AbstractPolicy,
    model::AbstractDistributionBasedModel,
    γ::Float64 = 0.9,
    θ::Float64 = 1e-4,
)
    states, actions = 1:length(observation_space(model)), 1:length(action_space(model))
    while true
        Δ = 0.0
        for s in states
            v = sum(
                a -> get_prob(π, s, a) *
                     sum(p * (r + γ * V(s′)) for (s′, r, p) in model(s, a)),
                actions,
            )
            error = v - V(s)
            update!(V, s => error)
            Δ = max(Δ, abs(error))
        end
        Δ < θ && break
    end
    V
end

const GridWorldLinearIndices = LinearIndices((4,4))
const GridWorldCartesianIndices = CartesianIndices((4,4))

isterminal(s::CartesianIndex{2}) = s == CartesianIndex(1,1) || s == CartesianIndex(4,4)

function nextstep(s::CartesianIndex{2}, a::CartesianIndex{2})
    ns = s + a
    if isterminal(s) || ns[1] < 1 || ns[1] > 4 || ns[2] < 1 || ns[2] > 4
        ns = s
    end
    r = isterminal(s) ? 0. : -1.0
    [(nextstate=GridWorldLinearIndices[ns], reward=r, prob=1.0)]
end

const GridWorldActions = [CartesianIndex(-1, 0),
                          CartesianIndex(1,0),
                          CartesianIndex(0, 1),
                          CartesianIndex(0, -1)]

const GridWorldEnvModel = DeterministicDistributionModel([nextstep(GridWorldCartesianIndices[s], a) for s in 1:16, a in GridWorldActions]);
n_state = length(observation_space(GridWorldEnvModel))
n_action = length(action_space(GridWorldEnvModel))

V, π = TabularVApproximator(16), TabularRandomPolicy(fill(0.25, 16, 4))
policy_evaluation!(V=V, π=π, model=GridWorldEnvModel, γ=1.0)

using Plots
using StatsBase:sample, Weights
heatmap(1:4, 1:4, reshape(V.table, 4,4), yflip=true)
