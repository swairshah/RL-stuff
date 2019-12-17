using Plots
using LinearAlgebra

const γ = 1
const SIZE = 4
const GRID = reshape(1:SIZE*SIZE, (SIZE, SIZE)) 

const ACTIONS = [
                 CartesianIndex(0, -1), # left
                 CartesianIndex(0, 1),  # right
                 CartesianIndex(-1, 0), # up
                 CartesianIndex(1,0)    # down
                ]

const ACTION_PROB = 0.25

const n_actions = 4
const n_states = 16

isterminal(s::CartesianIndex{2}) = 
        s == CartesianIndex(1,1) || s == CartesianIndex(4,4) 

function step(s::CartesianIndex{2}, a::CartesianIndex{2})
    ns = s + a
    if isterminal(s) || ns[1] < 1 || ns[1] > SIZE || ns[2] < 1 || ns[2] > SIZE
        ns = s
    end
    r = isterminal(s) ? 0.0 : -1.0

    (ns, r)
end

function PolicyEval!(value::Array{Float64, 2}, iters=100)
    for _ = 1:iters
        new_value = zeros(Float64, SIZE, SIZE)
        for ind in CartesianIndices((SIZE, SIZE))
            for action in ACTIONS
                next_ind, reward = step(ind, action)
                new_value[ind] += ACTION_PROB * (reward + γ * value[next_ind])
            end
        end
        value .= new_value
    end
    value
end

value = zeros(Float64, SIZE, SIZE)
valueiters!(value)

