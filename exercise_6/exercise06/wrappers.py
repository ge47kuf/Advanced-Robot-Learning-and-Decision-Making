import jax.numpy as jp
from gymnasium.spaces import flatten_space
from gymnasium.vector import VectorEnv, VectorObservationWrapper


class FlattenJaxObservation(VectorObservationWrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self.single_observation_space = flatten_space(env.single_observation_space)
        self.observation_space = flatten_space(env.observation_space)

    def observations(self, observations: dict) -> dict:
        return jp.concatenate([v for v in observations.values()], axis=-1)
