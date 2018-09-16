import gym
import time

env_name = "CartPole-v0"

env_name = "Riverraid-v0"

env_name = "Pong-v0"


env = gym.make(env_name)

for i_episode in range(2):
    observation = env.reset()
    print(env.action_space)

    count = 0
    while True:

        env.render()
        action = env.action_space.sample()


        time.sleep(0.05)
        observation, reward, done, info = env.step(3)
        count += 1
        # print(observation.shape)
        if reward != 0:
            print(reward)

        if done:
            print("Episode finished after {} timesteps".format(count))
            break


print("finish")
env.close()