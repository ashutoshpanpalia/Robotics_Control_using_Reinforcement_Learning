# Robotics Control using Reinforcement Learning
## Abstract:
In this repo, I experiment with the real of reinforcement learning (RL) algorithms to solve robotics control problems. The work spans Q-Learning, Deep Q-Networks (DQN), Advantage Actor-Critic (A2C), and Deep Deterministic Policy Gradient (DDPG).

## Introduction:
In the last few years, Reinforcement Learning (RL) has become a popular nonlinear control method. RL has a powerful potential to control systems with high non-linearity and complex dynamics. In this repo, we explore several RL algorithms and their effectiveness in solving different control problems. The following algorithms will be discussed along with their application on different control environment:

1. Q-Learning
2. DQN
3. DQN with PER
4. A2C discrete action space
5. A2C continuous action sapce
6. DDPG

I will begin by discussing the results of the above algorithm before diving into a detailed analysis of both the algorithm and the contorl problem statement it tackles
# Results:
https://github.com/ashutoshpanpalia/CSE-546_Final_project/assets/43078289/f666223f-1f4b-4ab4-b460-3b465fd45e41

https://github.com/ashutoshpanpalia/CSE-546_Final_project/assets/43078289/992e13a7-e604-4a20-8dd1-7a0e0ff8c4d8

https://github.com/ashutoshpanpalia/CSE-546_Final_project/assets/43078289/79ff6b94-0768-432c-878f-6fee142c2723



# Discusion of algorithms
## Q-Learning
4x4 Grid implying a 16 state environment
Terminal reward at (3,3): 1000; 
Intermediary(decaying with time steps) rewards at
(0,3)=300
(1,0)=10
(2,2) =10

Actions taken by Agent:
Action 0: Going Down
Action 1: Going up
Action 2: Going right
Action 3: Going left
Objective of the Agent is to maximize the reward starting from position (0,0)

![image](https://github.com/ashutoshpanpalia/Robotics_Control_using_Reinforcement_Learning/assets/43078289/1d343d7a-8f83-46b1-85ea-5ea9bbe3d566)

![image](https://github.com/ashutoshpanpalia/Robotics_Control_using_Reinforcement_Learning/assets/43078289/ef129ff0-c01b-4ec6-8799-ac745dab3765)



## Deep Q-Network (DQN)
The DQN algorithm was first described in Human-level control through deep reinforcement learning paper by DeepMind. I implemented the DQN algorithm and also implemented the improved version of DQN through integration of Prioritized Experience Replay (PER). All the algorithms are built from scratch. The following environments were solved:
1. CartPole Gymnasium Environment
2. Lunar Lander Gymnasium Environment

Discussion:

Target and Q-Network: DQN utilizes two neural networks, the Q-network and the Q-target, to stabilize the learning process. While the Q-network is updated frequently using experiences from the replay memory, the Q-target is updated less often to maintain stability and prevent divergence.

Experience Replay: Experience replay stores experiences in a memory buffer, enabling the agent to learn from a diverse set of past experiences. This approach improves sample efficiency and allows for the reuse of experiences multiple times during training.

Loss Calculation and Backpropagation: DQN calculates the loss using mean-squared error between the Q-values from the Q-target and the rewards plus discounted future rewards from the Q-network. Backpropagation is then used to update the neural network weights and biases.

Priority Experience Replay (PER): PER prioritizes experiences based on their significance, allowing the agent to learn more efficiently by replaying important transitions more frequently. Implementing PER leads to faster learning and better final policy quality compared to uniform experience replay.

## Advantage Actor Critic algorithm (A2C)
A2C algorithm is the synchronous version of A3C (Asynchronous Advantage Actor Critic) algorithm. The A2C algorithm is a combination of value-based and policy based methods to learn policy. And it also have variance less than Monte Carlo policy gradient method.

A2C have 2 network:

Actor: Actor takes observation as input and outputs the action. This is the policy that the agent
has to learn and use during greedy policy execution.

Critic: Critic approximates how good the action is and in what direction the actor network need
to be updated.

A2C was implemented using Advantage value as the critic.

Advantage = Reward + γ * V(next state) - V(current state)

Actor Loss= -( log(π(a|s)) * Advantage)

Critic Loss = MSE(R + γ * V(next state) and V(current state))

Discussion:

A2C algorithm was successful to solve the different discrete environments (Cartpole, Acrobot). After training, the greedy-policy followed was by choosing the action with the highest probability from the trained actor-network. The challenges faced during training were due to learning instability. The algorithm would train to get the optimal policy but would immediately ‘forget’ the policy and the rewards would drop suddenly. This issue is mitigate development by incorporating early-stopping while training.

A2C drawback for complex continuous environments: The inability of A2C to explore due to shrinking of standard deviation led to learning of a particular action instead of a generalized solution. This also occurred because of exploring poor
transitions (Transition with high negative reward) a lot more frequently than good transitions(Transition with positive rewards), which cause the network to learn to find a solution
of local minima and not explore high rewarding future states and action. This is one of the major challenges of the policy gradient method using Actor Critic methods, which have been solved by
adding replay buffers to store high priority transitions and sample good actions frequently in the DDPG algorithm. Consequently, A2C struggled to discover the optimal policy in continuous
environments like Pusher, MountainCar continuous, and Bipedal.



### NOTE: This was done as a part of the coursework of CSE 546 at the University at Buffalo. The source code is not available publicly to avoid academic integrity violations. Please reach out to author to discuss teh source code.
