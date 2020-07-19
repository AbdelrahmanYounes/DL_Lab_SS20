import tensorflow as tf
import numpy as np
from agent.replay_buffer import ReplayBuffer
import torch
def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.99, batch_size=128, epsilon=0.1, tau=0.01, lr=5e-4, history_length=0):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)
        
        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions


    def train(self, state, action, next_state, reward, terminal,env="CartPole-v0"):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets and loss 
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones =self.replay_buffer.next_batch(self.batch_size)

        batch_actions = torch.from_numpy(batch_actions).float().cuda()

        batch_rewards = torch.from_numpy(batch_rewards).float().cuda()
        batch_dones = torch.from_numpy(batch_dones).float().cuda()
        print()
        if env == 'CarRacing-v0':
            #print("enter")
            batch_states = torch.tensor(batch_states).permute(0, 3, 1, 2).cuda()
            batch_next_states = torch.tensor(batch_next_states).permute(0, 3, 1, 2).cuda()
        else:
            batch_states = torch.from_numpy(batch_states).float().cuda()
            batch_next_states = torch.from_numpy(batch_next_states).float().cuda()


        # Normal DQN
        #td_target = batch_rewards + self.gamma * (torch.max(self.Q_target(batch_next_states),dim=1)[0])*(1-batch_dones)
        #Double Q Learning to solve the problem of overestimation bias
        td_target = batch_rewards + self.gamma * (self.Q_target(batch_next_states)[torch.arange(self.batch_size).long(), torch.argmax(self.Q(batch_next_states), dim=1)]) * (1 - batch_dones)
        output = self.Q(batch_states)[torch.arange(self.batch_size).long(), batch_actions.long()]
        loss = self.loss_function(output,td_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        soft_update(self.Q_target, self.Q, self.tau)




    def act(self, state, deterministic,epsilon,env="CartPole-v0"):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()


        if deterministic or r > epsilon:
            # TODO: take greedy action (argmax)
            # action_id = ...
            #output = self.Q(torch.from_numpy(state).float().cuda())
            #action_id = torch.argmax(output, dim=1)
            if env == 'CarRacing-v0':

                action_id = torch.argmax(self.Q(torch.tensor(state.reshape(-1,96,96,1)).permute(0, 3, 1, 2).cuda()),dim=1).item()
            else:

                action_id = np.argmax(self.Q(torch.from_numpy(state).float().cuda()).cpu().detach().numpy())

        else:

            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            # action_id = ...
            action_id = np.random.randint(self.num_actions)
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
