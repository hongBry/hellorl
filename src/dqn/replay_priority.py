# Author: hong
# Date  : 9/13/2018
# Time  : 13:42 PM
# FileName: replay_priority.py

# copy from https://github.com/seekloud/hellorl/blob/master/src/dqn/replay_buffer.py
# edit a little. (Prioritized Experience Replay)
# reference https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py#L18-L86


"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""
import numpy as np
import time

floatX = 'float32'

from src.dqn.config import DISCOUNT

class SumTree(object):
    """
        This SumTree code is a modified version and the original code is from:
        https://github.com/jaara/AI-blog/blob/master/SumTree.py
        Story data with its priority in the tree.
        """
    data_pointer = 0

    def __init__(self, capacity):
        assert (capacity & capacity - 1) == 0  #capacity == 2^k
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.size = 0
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        # self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # # [--------------data frame-------------]
        # #             size: capacity

    def add(self, p):
        tree_idx = self.data_pointer + self.capacity - 1
        # self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        self.size = min(self.size + 1, self.capacity)
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0


    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_p(self):
        return self.tree[0]  # the root


class ReplayPriorityBuffer(object):
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

    """

    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, height, width, channel, rng, max_steps=8192, phi_length=4):
        """Construct a DataSet.

        Arguments:
            width, height, channel - image size
            max_steps - the number of time steps to store
            phi_length - number of images to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches

        """
        # TODO: Specify capacity in number of state transitions, not
        # number of saved time steps.

        # Store arguments.
        self.width = width
        self.height = height
        self.channel = channel
        self.max_steps = max_steps
        self.phi_length = phi_length
        self.rng = rng

        # Allocate the circular buffers and indices.
        self.imgs = np.zeros((max_steps, phi_length + 1, channel, height, width), dtype='uint8')
        # self.next_imgs = np.zeros((max_steps, phi_length, channel, height, width), dtype='uint8')
        self.actions = np.zeros(max_steps, dtype='int32')
        self.rewards = np.zeros(max_steps, dtype=floatX)
        self.terminal = np.zeros(max_steps, dtype='bool')
        self.terminal[-1] = True  # set terminal for the first episode.
        self.R = np.zeros(max_steps, dtype=floatX)

        self.bottom = 0
        self.top = 0
        self.size = 0

        self.tree = SumTree(max_steps)



    def add_sample(self, img, action, reward, terminal, next_img):
        """Add a time step record.
        Arguments:
            img -- observed image (phi, chanel, height ,width), it will be changed to (channel, height, width)
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.imgs[self.top][:-1] = img
        self.imgs[self.top][-1] = next_img.transpose(2, 0, 1)

        self.actions[self.top] = action
        # self.rewards[self.top] = max(1, min(-1, reward))  # clip reward
        self.rewards[self.top] = reward  # clip reward
        self.terminal[self.top] = terminal
        self.R[self.top] = -1000.0

        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p)


        if terminal:
            self.R[self.top] = reward
            idx = self.top
            while True:
                idx -= 1
                if self.terminal[idx]:
                    break
                self.R[idx] = self.R[idx + 1] * DISCOUNT + self.rewards[idx]  # fix bug

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps


    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return self.size



    def priority_sample(self, batch_size):
        """Return corresponding imgs, actions, rewards, terminal status, treeIdx, and sample weights for
        batch_size chosen state transitions.

        """
        imgs = np.zeros((batch_size,
                         self.phi_length + 1,
                         self.channel,
                         self.height,
                         self.width),
                        dtype='uint8')
        actions = np.zeros((batch_size, 1), dtype='int32')
        rewards = np.zeros((batch_size, 1), dtype=floatX)
        terminal = np.zeros((batch_size, 1), dtype='bool')
        R = np.zeros((batch_size, 1), dtype=floatX)
        next_img = np.zeros((batch_size,
                         self.channel * self.phi_length,
                         self.height,
                         self.width),
                        dtype='uint8')

        count = 0
        tree_idx, IS_weight = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size,1))
        pri_seg = self.tree.total_p / batch_size
        self.beta = min(1., self.beta + self.beta_increment_per_sampling) # max = 1
        min_prob = np.min(self.tree.tree[-self.tree.capacity: self.tree.capacity + self.tree.size - 1]) / self.tree.total_p  # for later calculate ISweight
        # print(min_prob)

        while count < batch_size:
            # print('count:', count)
            # print('batch_size:', batch_size)
            # print('self.bottom:', self.bottom)
            # print('self.size:', self.size)
            # print('self.phi_length:', self.phi_length)

            # Randomly choose a time step from the replay memory.
            # print(self.tree.data_pointer, self.top)
            a, b = pri_seg * count, pri_seg * (count + 1)
            v = self.rng.uniform(a, b)

            idx, p, data_idx = self.tree.get_leaf(v)

            # index = data_idx + self.bottom - self.phi_length + 1

            # print(count,index,self.size,self.bottom)
            # all_indices = np.arange(index, index + self.phi_length + 1)
            #
            # if (self.size != self.max_steps and index < self.bottom) or (index >= self.bottom + self.size - self.phi_length):
                # print("random",index % self.max_steps,self.top % self.max_steps)
                # continue
            #     #
            #     index = self.rng.randint(self.bottom,
            #                              self.bottom + self.size - self.phi_length)
            #     end_index = index + self.phi_length - 1
            #     idx = end_index % self.size + self.tree.capacity - 1
            #     p = self.tree.tree[idx]


            # if index >= self.bottom + self.size - self.phi_length:
            #     # continue
            #     print("random")
            #     index = self.rng.randint(self.bottom,
            #                              self.bottom + self.size - self.phi_length)
            #     end_index = index + self.phi_length - 1
            #     idx = end_index % self.size + self.tree.capacity - 1
            #     p = self.tree.tree[idx]
            # index = self.rng.randint(self.bottom,
            #                          self.bottom + self.size - self.phi_length)

            # Both the before and after states contain phi_length
            # frames, overlapping except for the first and last.
            # all_indices = np.arange(index, index + self.phi_length + 1)
            # end_index = index + self.phi_length - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but its last frame (the
            # second-to-last frame in imgs) may be terminal. If the last
            # frame of the initial state is terminal, then the last
            # frame of the transitioned state will actually be the first
            # frame of a new episode, which the Q learner recognizes and
            # handles correctly during training by zeroing the
            # discounted future reward estimate.
            # if np.any(self.terminal.take(all_indices[0:-2], mode='wrap')) or self.R.take(end_index,
            #                                                                       mode='wrap') == -1000.0:
            # if np.any(self.terminal.take(all_indices[0:-2], mode='wrap')) :
            #
            #     print(end_index)
            #     continue

            # Add the state transition to the response.
            imgs[count] = self.imgs[data_idx]
            # next_img[count] = self.next_imgs[data_idx]
            actions[count] = self.actions.take(data_idx, mode='wrap')
            rewards[count] = self.rewards.take(data_idx, mode='wrap')
            terminal[count] = self.terminal.take(data_idx, mode='wrap')
            R[count] = self.R.take(data_idx, mode='wrap')

            prob = p / self.tree.total_p
            IS_weight[count][0] = np.power(prob / min_prob, -self.beta)

            tree_idx[count] = idx
            # print(end_index,self.tree.tree[idx])

            count += 1

        return imgs, actions, rewards, terminal, tree_idx, IS_weight

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)



def test1():
    rng = np.random.RandomState()
    buff = ReplayPriorityBuffer(5, 3, 1, rng, 64)

    for i in range(20):
        img = np.arange(i, i + 15, dtype=np.uint8).reshape(5, 3, 1)
        buff.add_sample(img, i, i + 0.1, False)

    img = np.arange(100, 100 + 15, dtype=np.uint8).reshape(5, 3, 1)
    buff.add_sample(img, 100, 100 + 0.1, True)

    for i in range(12):
        imgs, actions, rewards, terminal, tree_idx, sample_weight = buff.priority_sample(2)
        print(i,tree_idx)

        buff.batch_update(tree_idx, np.zeros(shape=(2,)))
        print(buff.tree.total_p)
        print(sample_weight)

    imgs, actions, rewards, terminal, tree_idx, sample_weight = buff.priority_sample(2)
    print(12,tree_idx)
    buff.batch_update(tree_idx, np.array([1.0,2]))
    print(buff.tree.total_p)

    imgs, actions, rewards, terminal, tree_idx, sample_weight = buff.priority_sample(2)
    print(13, tree_idx)
    buff.batch_update(tree_idx, np.ones(shape=(2,)))
    print(buff.tree.total_p)

    imgs, actions, rewards, terminal, tree_idx, sample_weight = buff.priority_sample(2)
    print(14, tree_idx)
    buff.batch_update(tree_idx, np.array([1,0.8]))
    print(buff.tree.total_p)

    imgs, actions, rewards, terminal, tree_idx, sample_weight = buff.priority_sample(2)
    print(15, tree_idx)
    buff.batch_update(tree_idx, np.zeros(shape=(2,)))
    print(buff.tree.total_p)



    # imgs, actions, rewards, terminal, tree_idx, sample_weight = buff.priority_sample(2)
    # print(tree_idx)
    #
    # buff.batch_update(tree_idx,np.zeros(shape=(2,)))
    # imgs, actions, rewards, terminal, tree_idx, sample_weight = buff.priority_sample(2)
    # print(tree_idx)
    #
    # buff.batch_update(tree_idx, np.zeros(shape=(2,)))
    # imgs, actions, rewards, terminal, tree_idx, sample_weight = buff.priority_sample(2)
    # print(tree_idx)


    # print('-------------------')
    # print(buff.imgs)
    # print('-------------------')
    # print(buff.actions)
    # print('-------------------')
    # print(buff.rewards)
    # print('-------------------')
    # print(buff.terminal)
    # print('-------------------')
    # print(buff.R)
    #
    # print('-------------------')
    # print('-------------------')
    # print('-------------------')
    # print('-------------------')
    # print(imgs)
    # print('-------------------')
    # print(actions)
    # print('-------------------')
    # print(rewards)
    # print('-------------------')
    # print(terminal)
    # print('-------------------')
    # print(tree_idx)
    # print('-------------------')
    # print(sample_weight)

    # print(R)


if __name__ == '__main__':
    test1()
