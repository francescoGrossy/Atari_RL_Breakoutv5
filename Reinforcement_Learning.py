#Import dependencies
import gymnasium as gym
import ale_py
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

# Create the environment
env = gym.make("ALE/Breakout-v5", render_mode="human")

episodes = 5
for episode in range(1, episodes+1):
    obs, info = env.reset()
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        env.render()
    print(f"Episode: {episode}, Score: {score}")
# Close the environment
env.close()

# Create the environment split into 4 schemas to improve performance and learning
env = make_atari_env("ALE/Breakout-v5", n_envs=4, seed=0)
# Wrap the environment
env = VecFrameStack(env, n_stack=4)

#Log path
log_path = os.path.join("repos\Reinforcement Learning\Training", "Logs")

# Create the model
model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_path)

# Train the model
model.learn(total_timesteps=100000, tb_log_name="A2C")

# Save the model
a2a_path = os.path.join("repos\Reinforcement Learning\Training", "Saved Models", "A2C")
model.save(a2a_path)


# Load the model
model = A2C.load(a2a_path, env=env)



# Model evaluation, I am setting n_envs=1 to evaluate the model in a single environment and avoid environment noise
eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Model testing
episodes = 5
obs = env.reset()
for episode in range(1, episodes + 1):
    done = [False] * env.num_envs  #I am using a list to track the done status of each environment
    score = [0] * env.num_envs  # Points for each environment

    while not all(done):  # Keep on going until all environments are done
        action, _ = model.predict(obs)  # It predicts the action to take for each environment
        obs, rewards, dones, infos = env.step(action)  

        #Score update for each environment
        for i in range(env.num_envs):
            if not done[i]:  # If the environment is not done, update the score
                score[i] += rewards[i]

        done = dones  # Update the done status for each environment

    print(f"Episode: {episode}, Scores: {score}")

# Close the environment
env.close()