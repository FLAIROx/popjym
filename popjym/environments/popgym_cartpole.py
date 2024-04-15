from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax


@struct.dataclass
class EnvState:
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    time: int


@struct.dataclass
class EnvParams:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1  # (masscart + masspole)
    length: float = 0.5
    polemass_length: float = 0.05  # (masspole * length)
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4


class NoisyStatelessCartPole(environment.Environment):
    """
    JAX Compatible version of CartPole-v1 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    def __init__(self, noise_sigma=0.0, max_steps_in_episode=200):
        super().__init__()
        self.obs_shape = (2,)
        self.noise_sigma = noise_sigma
        self.max_steps_in_episode = max_steps_in_episode

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        force = params.force_mag * action - params.force_mag * (1 - action)
        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (
            force + params.polemass_length * state.theta_dot**2 * sintheta
        ) / params.total_mass
        thetaacc = (params.gravity * sintheta - costheta * temp) / (
            params.length * (4.0 / 3.0 - params.masspole * costheta**2 / params.total_mass)
        )
        xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

        # Only default Euler integration option available here!
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        # Important: Reward is based on termination is previous step transition
        reward = 1.0 - prev_terminal
        reward = self.reward_transform(params, reward)

        # Update state dict and evaluate termination conditions
        state = EnvState(x, x_dot, theta, theta_dot, state.time + 1)
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(key, state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reward_transform(self, params: EnvParams, reward: float):
        return jnp.where(jnp.isclose(reward, 0), -1.0, 1.0 / self.max_steps_in_episode)

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        init_state = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        key, _ = jax.random.split(key)
        state = EnvState(
            x=init_state[0],
            x_dot=init_state[1],
            theta=init_state[2],
            theta_dot=init_state[3],
            time=0,
        )
        return self.get_obs(key, state, params), state

    def get_obs(self, key: chex.PRNGKey, state: EnvState, params: EnvParams) -> chex.Array:
        """Applies observation function to state."""
        obs = (
            jnp.array([state.x, state.theta])
            + jax.random.normal(key, shape=(2,)) * self.noise_sigma
        )
        obs = jnp.clip(obs, self.observation_space(params).low, self.observation_space(params).high)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done2 = jnp.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )

        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps_in_episode
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPole-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                params.theta_threshold_radians * 2,
            ]
        )
        return spaces.Box(-high, high, (2,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Dict(
            {
                "x": spaces.Box(-high[0], high[0], (), jnp.float32),
                "x_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
                "theta": spaces.Box(-high[2], high[2], (), jnp.float32),
                "theta_dot": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(self.max_steps_in_episode),
            }
        )


class StatelessCartPoleEasy(NoisyStatelessCartPole):
    def __init__(self):
        super().__init__(noise_sigma=0.0, max_steps_in_episode=200)


class StatelessCartPoleMedium(NoisyStatelessCartPole):
    def __init__(self):
        super().__init__(noise_sigma=0.0, max_steps_in_episode=400)


class StatelessCartPoleHard(NoisyStatelessCartPole):
    def __init__(self):
        super().__init__(noise_sigma=0.0, max_steps_in_episode=600)


class NoisyStatelessCartPoleEasy(NoisyStatelessCartPole):
    def __init__(self):
        super().__init__(noise_sigma=0.1, max_steps_in_episode=200)


class NoisyStatelessCartPoleMedium(NoisyStatelessCartPole):
    def __init__(self):
        super().__init__(noise_sigma=0.2, max_steps_in_episode=200)


class NoisyStatelessCartPoleHard(NoisyStatelessCartPole):
    def __init__(self):
        super().__init__(noise_sigma=0.3, max_steps_in_episode=200)
