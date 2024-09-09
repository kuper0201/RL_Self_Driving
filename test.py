from stable_baselines3 import PPO
from CarlaEnv import CarlaLaneTrackingEnv

def main():
    # Generate Env
    env = CarlaLaneTrackingEnv()

    model = PPO.load('model/ppo')
    model.set_env(env)

    for _ in range(5): # 5 Episodes
        obs, _ = env.reset()
        done = False
        while not done:
            act, stat = model.predict(obs, deterministic=True)
            obs, rew, done, _, info = env.step(act)

if __name__ == "__main__":
    main()