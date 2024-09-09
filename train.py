from stable_baselines3 import PPO
from torch import nn
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from CarlaEnv import CarlaLaneTrackingEnv

def main():
    # Generate Env
    env = CarlaLaneTrackingEnv()

    # Build PPO Model
    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[400, 300], vf=[400, 300]))
    model = PPO("CnnPolicy", env, batch_size=64, n_steps=4096, learning_rate=0.0001, policy_kwargs=policy_kwargs, verbose=2)

    # Training
    callback = CheckpointCallback(save_freq=4096 * 4, save_path="ck", name_prefix="ppo")
    model.learn(total_timesteps=8e5, progress_bar=True, callback=callback)
    model.save("ppo")

if __name__ == "__main__":
    main()