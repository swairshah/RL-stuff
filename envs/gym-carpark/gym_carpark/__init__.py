from gym.envs.registration import register

register(
    id='CarPark-v0',
    entry_point='gym_carpark.envs:CarPark',
)
