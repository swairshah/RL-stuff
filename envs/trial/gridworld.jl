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

#function PolicyImprove(value::Array{Float64, 2})
#    policy = zeros(Float64, SIZE, SIZE)
#    for ind in CartesianIndices((SIZE, SIZE))

value = zeros(Float64, SIZE, SIZE)
PolicyEval!(value)
display(value); println()

#i = CartesianIndex(2, 2)
#for action in ACTIONS
#    println(action)
#    j, r = step(i, action)
#    println(j, r + γ * value[j])
#end

π = fill(0.25, n_states, n_actions)

println("random policy")
display(π)

linear = LinearIndices((SIZE, SIZE))
cartesian = CartesianIndices((SIZE, SIZE))

for s in linear
    chosen_a = argmax(π[s])

    action_vals = zeros(n_actions)
    for a in 1:n_actions
        (ns, r) = step(cartesian[s], ACTIONS[a])
        action_vals[a] = r + γ * value[ns]
    end

    best_a = argmax(action_vals)

    if chosen_a != best_a
        π[s, :] = Matrix{Float64}(I, n_actions, n_actions)[best_a, :]
    end
end

println("updated policy")
display(π)
