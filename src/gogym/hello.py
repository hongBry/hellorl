import gym
import time

env_name = "CartPole-v0"
env_name = "Riverraid-v0"

env = gym.make(env_name)

for i_episode in range(2):
    observation = env.reset()
    print(env.action_space)


    for t in range(1000):
        env.render()
        action = env.action_space.sample()


        time.sleep(0.05)
        observation, reward, done, info = env.step(action)
        print(reward)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


print("finish")
env.close()