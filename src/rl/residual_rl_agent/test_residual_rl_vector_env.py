from src.envs.wrappers import Pi0ObservationWrapper, QueryFrequencyWrapper
from src.rl.residual_rl_agent.residual_rl_vector_env import ResidualRLVectorEnv
import numpy as np
import gymnasium as gym

    
def test_residual_rl_vector_env():
    
    class MockLiberoEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                "agentview_image": gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
                "robot0_eef_pos": gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                "robot0_eef_quat": gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                "robot0_gripper_qpos": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            })
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            obs = {
                "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
                "robot0_eef_pos": np.zeros((3,), dtype=np.float32),
                "robot0_eef_quat": np.array([0., 0., 0., 1.], dtype=np.float32),
                "robot0_gripper_qpos": np.zeros((2,), dtype=np.float32),
            }
            return obs, {}

        def step(self, action):
            obs, _ = self.reset()
            return obs, 1.0, False, False, {"step": 1}

    def make_env_fn():
        env = MockLiberoEnv()
        env = Pi0ObservationWrapper(
            env=env,
            env_class="libero",
            task_description="test task",
            add_states=True,
            pi0_obs_prefix="pi0",
        )
        env = QueryFrequencyWrapper(
            env=env,
            query_frequency=1,
            discount=0.99,
            store_full_transitions=True,
        )
        return env
    env_fns = [make_env_fn for _ in range(4)]
    wrapped = ResidualRLVectorEnv(env_fns)
    
    print("Testing reset()...")
    obs, info = wrapped.reset()
    observation = obs["observation"]
    action = obs["action"]
    base_action = obs["base_action"]
    
    image = observation["pi0/image"]
    state = observation["pi0/state"]
    wrist_image = observation["pi0/wrist_image"]
    
    assert image.shape == (4, 1, 128, 128, 3)
    assert state.shape == (4, 1, 8)
    assert wrist_image.shape == (4, 1, 128, 128, 3)
    assert action.shape == (4, 1, 7)
    assert base_action.shape == (4, 1, 7)
    
    print("Testing step()...")
    residual_action = np.random.randn(4, 1, 7)
    next_obs, reward, terminate, truncate, info = wrapped.step(residual_action)
    next_observation = next_obs["observation"]
    action = next_obs["action"]
    base_action = next_obs["base_action"]
    
    next_image = next_observation["pi0/image"]
    next_state = next_observation["pi0/state"]
    next_wrist_image = next_observation["pi0/wrist_image"]
    
    assert next_image.shape == (4, 1, 128, 128, 3)
    assert next_state.shape == (4, 1, 8)
    assert next_wrist_image.shape == (4, 1, 128, 128, 3)
    assert action.shape == (4, 1, 7)
    assert base_action.shape == (4, 1, 7)

    assert reward.shape == (4, 1)
    assert terminate.shape == (4, 1)
    assert truncate.shape == (4, 1)
    assert info.shape == (4,)

    print("Test passed!")


if __name__ == "__main__":
    test_residual_rl_vector_env()