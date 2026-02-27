from src.envs.wrappers import Pi0ObservationWrapper, QueryFrequencyWrapper
from src.envs.venv import DSRLVectorEnv
import numpy as np
import gymnasium as gym

    
def test_dsrl_vector_env():
    
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
            query_frequency=5,
            discount=0.99,
            store_full_transitions=True,
        )
        return env
    env_fns = [make_env_fn for _ in range(4)]
    wrapped = DSRLVectorEnv(env_fns)
    
    obs, info = wrapped.reset()
    observation = obs["observation"]
    action = obs["action"]
    prefix_rep = obs["prefix_rep"]
    
    image = observation["pi0/image"]
    state = observation["pi0/state"]
    wrist_image = observation["pi0/wrist_image"]
    
    assert image.shape == (4, 5, 128, 128, 3)
    assert state.shape == (4, 5, 8)
    assert wrist_image.shape == (4, 5, 128, 128, 3)
    assert action.shape == (4, 5, 7)
    assert prefix_rep.shape == (4, 968, 2048)
    
    noise = np.random.randn(4, 10, 32)
    next_obs, reward, terminate, truncate, info = wrapped.step(noise)
    next_observation = next_obs["observation"]
    action = next_obs["action"]
    prefix_rep = next_obs["prefix_rep"]
    
    next_image = next_observation["pi0/image"]
    next_state = next_observation["pi0/state"]
    next_wrist_image = next_observation["pi0/wrist_image"]
    
    assert next_image.shape == (4, 5, 128, 128, 3)
    assert next_state.shape == (4, 5, 8)
    assert next_wrist_image.shape == (4, 5, 128, 128, 3)
    assert action.shape == (4, 5, 7)
    assert prefix_rep.shape == (4, 968, 2048)

    assert reward.shape == (4, 5)
    assert terminate.shape == (4, 5)
    assert truncate.shape == (4, 5)
    assert info.shape == (4,)

    print("Test passed!")


if __name__ == "__main__":
    test_dsrl_vector_env()