# policy gradient算法实现

### 目标
1. 完成 policy gradient 的基本功能;
2. 在 atari 游戏：'Riverraid' 中进行实验;


### policy gradient 算法流程
policy gradient 算法是强化学习中policy based方法的一种，和dqn，q-learning不同的是
policy gradient 算法虽然也接受环境信息（state），然是它输出不是各个action的value，
只是各个action的概率。而dqn应该是让机器去学会一个评估action的价值（value）函数，
然后通过贪婪算法去选取获得最高价值的action.


policy gradient 算法的流程：
```angular2html
1.先构建一个策略网络（policy network），输入每一帧游戏的画面，输出是各个动作的概率.
2.for eposide in eposides do:
        用policy network_theta去玩一局游戏，收集游戏的数据：
        （s0，a0，r1，s1, a1, r2 ...., rT）;
        计算R(t) = R(t) + distcount * R (t + 1)
        for t = 1 to T -1 do:
            theta = theta + alpha * gradient( (log p(at | st) * R(t))
        end for
  end for.

```


### 2018/09/18 version for pong
1.使用minibatch的方式进行训练，效果不好。

2.参考 [https://github.com/mrahtz/tensorflow-rl-pong]的游戏图片预处理方案和state输入方案

3.大约在300 eposide 网络开始慢慢收敛
600个eposide(2个多小时) 能打败gym bot

