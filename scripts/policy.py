from typing_extensions import dataclass_transform
import numpy as np
import os
from collections import deque
import gc
import random
import pandas as pd
from scipy.special import softmax
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
import Networks
import envs.exploration_env as robot


class DeepQ(object):
    def __init__(self, case_path, model_name):
        # define the local file path
        self.case_path = case_path
        self.weights_path = "../data/torch_weights/" + self.case_path
        self.reward_data_path = "../data/reward_data/" + self.case_path
        self.object_path = '../data/training_object_data/' + self.case_path
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        if not os.path.exists(self.reward_data_path):
            os.makedirs(self.reward_data_path)
        if not os.path.exists(self.object_path):
            os.makedirs(self.object_path)
        data_all = pd.DataFrame({"Step": [], "Reward": []})
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)

        # setup parameters for RL
        self.BATCH = 64  # 64
        self.REPLAY_MEMORY = 1e4  # 1e4
        self.GAMMA = 0.99
        self.OBSERVE = 5e3  # 5e3
        self.EXPLORE = 1e6  # 5e5
        self.epoch = 1e4  # 1e4
        
        if model_name == "GCN":
            self.TARGET_UPDATE = 15000#1000
        else:
            self.TARGET_UPDATE = 9000

        # exploration and exploitation trad-off parameters
        # e-greedy trad-off
        self.FINAL_EPSILON = 0  # final value of epsilon
        self.INITIAL_EPSILON = 0.9#0.9~0
        self.max_grad_norm = 0.5

        # setup environment parameters
        self.map_size = 40
        # setup replay memory
        self.buffer = deque()
        # setup training
        self.step_t = 0
        self.epsilon = self.INITIAL_EPSILON
        self.temp_loss = 0
        self.total_reward = np.empty([0, 0])

    def running(self, model, modelTarget, test=False):
        data_all = pd.read_csv(self.reward_data_path + "reward_data.csv")
        temp_i = 0
        Test = test
        method = "bayesian"  # bayesian, e-greedy
        env = robot.ExplorationEnv(self.map_size, 0, Test)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = model
        target_net = modelTarget
        target_net.eval()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-5)

        temp_reward_data = []
        temp_loss_data = []
        while temp_i < self.epoch:
            self.step_t += 1
            temp_i += 1
            # e-greedy scale down epsilon
            if self.epsilon > self.FINAL_EPSILON and self.step_t > self.OBSERVE:
                self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE
            # get the input data (X, A)
            adjacency, featrues, globals_features, fro_size = env.graph_matrix()
            node_size = adjacency.shape[0]
            key_size = node_size - fro_size
            s_t = self.data_process([adjacency, featrues])

            # get the output reward (Y)
            all_actions = env.actions_all_goals()
            rewards = env.rewards_all_goals(all_actions)

            # choose an action
            if method == "e-greedy":
                readout_t = self.test(s_t, 0.0, device, policy_net).cpu().detach().numpy()#0.9~0 状态 epsilon 设备 网络 
                a_t = np.zeros([node_size])
                if random.random() <= self.epsilon:
                    # Random Action
                    state = "explore"
                    action_index = random.randrange(fro_size)
                    a_t[key_size + action_index] = 1
                else:
                    # Policy Action
                    state = "exploit"
                    action_index = np.argmax(readout_t[-fro_size:])
                    a_t[key_size + action_index] = 1
            elif method == "bayesian":
                readout_t = self.test(s_t, self.epsilon, device, policy_net).cpu().detach().numpy()
                state = "bayesian"
                a_t = np.zeros([node_size])
                action_index = np.argmax(readout_t[-fro_size:])
                a_t[key_size + action_index] = 1

            # choose an action
            actions = all_actions[key_size + action_index]

            # get reward
            r_t = rewards[key_size + action_index]

            # move to next view point
            for act in actions:
                _, done, _ = env.step(act)

            # terminal for Q networks
            # cur_ter = env.loop_clo and r_t == 1
            current_done = done or env.loop_clo

            # get next state
            adjacency, featrues, globals_features, fro_size1 = env.graph_matrix()
            s_t1 = self.data_process([adjacency, featrues])

            # save to buffer
            self.buffer.append((s_t, a_t, r_t, s_t1, current_done, fro_size1))
            if len(self.buffer) > self.REPLAY_MEMORY:
                self.buffer.popleft()#拿出最左边的一个

            # training step
            if self.step_t > self.OBSERVE:
                # updata target network
                if self.step_t % self.TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # sample a minibatch to train on
                minibatch = random.sample(self.buffer, self.BATCH)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]
                s_j_loader = DataLoader(s_j_batch, batch_size=self.BATCH)
                s_j1_loader = DataLoader(s_j1_batch, batch_size=self.BATCH)
                for batch in s_j_loader:
                    s_j_batch = batch
                for batch1 in s_j1_loader:
                    s_j1_batch = batch1

                r_batch = [d[2] for d in minibatch]
                readout_j1_batch = self.test(s_j1_batch, 0.0, device, target_net)
                readout_j1_batch = readout_j1_batch.cpu().detach().numpy()
                a_batch = np.array([])
                y_batch = np.array([])
                start_p = 0
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    action_space = minibatch[i][5]
                    act = minibatch[i][1]
                    a_batch = np.append(a_batch, act)
                    node_space = len(act)

                    temp_y = np.zeros(node_space)
                    index = np.argmax(act)
                    # if terminal, only equals reward
                    if terminal:
                        temp_y[index] = r_batch[i]
                    else:
                        temp_range = readout_j1_batch[start_p:start_p + node_space]
                        temp_range = temp_range[-action_space:]
                        max_q = np.max(temp_range)
                        temp_y[index] = r_batch[i] + self.GAMMA * max_q
                    start_p += node_space
                    y_batch = np.append(y_batch, temp_y)

                # perform gradient step
                self.train(s_j_batch, a_batch, y_batch, device, policy_net, optimizer)
                temp_loss_data.append([self.step_t, self.temp_loss])

            print("TIMESTEP", self.step_t, "/ STATE", state, "/ EPSILON", self.epsilon,
                  "/ Q_MAX %e" % np.max(readout_t), "/ EXPLORED", env.status(), "/ REWARD", r_t,
                  "/ Terminal", current_done, "\n")

            if done:
                del env
                gc.collect()
                env = robot.ExplorationEnv(self.map_size, 0, Test)
                done = False

            data_all = data_all.append({"Step": self.step_t, "Reward": r_t}, ignore_index=True)
            self.total_reward = np.append(self.total_reward, r_t)

            # save progress every 50000 iterations
            if self.step_t % 5e4 == 0:
                torch.save(policy_net.state_dict(), self.weights_path + 'MyModel.pt')
            if self.step_t > 1000:
                new_average_reward = np.average(self.total_reward[len(self.total_reward) - 1000:])
                if self.step_t % 1e2 == 0:
                    temp_reward_data.append([self.step_t, new_average_reward])

        np.savetxt(self.object_path + "temp_reward.csv", temp_reward_data, delimiter=",")
        np.savetxt(self.object_path + "temp_loss.csv", temp_loss_data, delimiter=",")
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)
        torch.save(policy_net.state_dict(), self.object_path + 'Model_Policy.pt')
        torch.save(target_net.state_dict(), self.object_path + 'Model_Target.pt')

    def data_process(self, data):
        s_a, s_x = data
        edge_index = []
        edge_attr = []
        edge_set = set()
        for a_i in range(np.shape(s_a)[0]):
            for a_j in range(np.shape(s_a)[1]):
                if (a_i, a_j) in edge_set or (a_j, a_i) in edge_set \
                        or s_a[a_i][a_j] == 0:
                    continue
                edge_index.append([a_i, a_j])
                edge_attr.append(s_a[a_i][a_j])
                if a_i != a_j:
                    edge_index.append([a_j, a_i])
                    edge_attr.append(s_a[a_j][a_i])
                edge_set.add((a_i, a_j))
                edge_set.add((a_j, a_i))
        edge_index = torch.tensor(np.transpose(edge_index), dtype=torch.long)
        x = torch.tensor(s_x, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        state = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return state

    def cost(self, pred, target, action):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        readout_action = torch.mul(pred_flat, action)
        loss = torch.pow(readout_action - target_flat, 2).sum() / self.BATCH
        return loss

    def train(self, data, action, y, device, model, optimizer):
        model.train()
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data, 0.5, batch=data.batch)
        y = torch.tensor(y).to(device)
        action = torch.tensor(action).to(device)
        loss = self.cost(out, y, action)
        self.temp_loss = loss.item()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
        optimizer.step()

    def test(self, data, prob, device, model):#状态 epsilon 设备 网络 
        model.eval()
        data = data.to(device)
        pred = model(data, prob)
        return pred


class A2C(object):
    def __init__(self, case_path):
        # define the local file path
        self.case_path = case_path
        self.weights_path = "../data/torch_weights/" + self.case_path
        self.reward_data_path = "../data/reward_data/" + self.case_path
        self.object_path = '../data/training_object_data/' + self.case_path
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        if not os.path.exists(self.reward_data_path):
            os.makedirs(self.reward_data_path)
        if not os.path.exists(self.object_path):
            os.makedirs(self.object_path)
        data_all = pd.DataFrame({"Step": [], "Reward": [], "Loss":[], "Entropy":[], "Terminal":[]})
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)

        # setup parameters for RL
        self.GAMMA = 0.99
        self.EXPLORE = 1e6  # 5e5
        self.epoch = 1e4 # 1e4
        self.nstep = 40
        self.ent_coef = 0.01#损失计算的输出熵系数
        self.vf_coef = 0.25#损失计算的价值函数系数
        self.max_grad_norm = 0.5

        # setup memory
        self.buffer = deque()
        # setup environment parameters
        self.map_size = 40
        # setup training
        self.step_t = 0
        self.temp_loss = 0
        self.entro = 0
        self.total_reward = np.empty([0, 0])

    def running(self, actor, critic, test=False):
        data_all = pd.read_csv(self.reward_data_path + "reward_data.csv")
        temp_i = 0
        Test = test
        env = robot.ExplorationEnv(self.map_size, 0, Test)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = actor
        value_net = critic
        params = list(policy_net.parameters()) + list(value_net.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-5)

        temp_reward_data = []
        temp_loss_data = []
        while temp_i < self.epoch:#10000
            self.step_t += 1
            temp_i += 1
            # get the input data (X, A)
            adjacency, featrues, globals_features, fro_size, vehicle_position = env.graph_matrix()#邻接矩阵 特征
            node_size = adjacency.shape[0]#读取邻接矩阵第一维度的长度,相当于行数
            key_size = node_size - fro_size#关键节点=节点总数-边界节点
            s_t, b_t = self.data_process([adjacency, featrues], device)#s_t: Data(x=[8, 5]特征, edge_index=[2, 22], edge_attr=[22]) b_t: tensor([0,0,0,0,0,0,0,0])
            mask = np.zeros([node_size])#array([0,0,0,0,0,0,0,0])
            mask[-fro_size:] = 1#最后几个改成1   array([0,0,0,0,0,0,1,1])

            # get the output reward (Y)
            all_actions = env.actions_all_goals()#[[],[],[],...,[[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]]
            rewards = env.rewards_all_goals(all_actions)#所有动作对应的奖励 array([ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.])  array([ 0.,  0.,  0.,  0.,  0.,  0.,  -1., 1.])

            # choose an action   #s_t: Data(x=[8, 5], edge_index=[2, 22], edge_attr=[22]) b_t: tensor([0,0,0,0,0,0,0,0]) mask: array([0,0,0,0,0,0,1,1])
            readout_t = self.test(s_t, b_t, mask, device, policy_net).view(-1).cpu().detach().numpy()#策略网络array([0.51,0.49]) 选择两个边界节点的概率
            val = self.test(s_t, b_t, mask, device, value_net).item()#价值网络取值0.0047

            action_index = np.random.choice(fro_size, 1, p=readout_t)[0]#按概率取到选择的动作编号
            action_index = key_size + action_index#all_actions集合里面的编号

            a_t = np.zeros([node_size])#array([0., 0., 0., 0., 0., 0., 0., 0.])
            a_t[action_index] = 1#将要选择的动作编号置1

            # choose an action
            actions = all_actions[action_index]#选择动作集[[0,0,0],[0,0,0],[0,0,0],[0,0,0]....]

            # get reward
            r_t = rewards[action_index]#该组动作对应的奖励  如果能得到最大奖励的点不是与当前位姿相关联的点 也不可以选

            # move to the next view point
            for act in actions:
                _, done, _ = env.step(act)

            # terminal for RL value calculation   RL值计算终端
            current_done = done or env.loop_clo#探索了大于85%或者成环

            # get next state
            adjacency, featrues, globals_features, fro_size1, vehicle_position1 = env.graph_matrix()
            s_t1, b_t1 = self.data_process([adjacency, featrues], device)
            mask = np.zeros([adjacency.shape[0]])
            mask[-fro_size1:] = 1

            last_value = self.test(s_t1, b_t1, mask, device, value_net).item()

            # save to buffer
            self.buffer.append((s_t, a_t, r_t, s_t1, current_done, fro_size, val))

            # training step
            if len(self.buffer) == self.nstep:
                # get the batch variables
                s_j_batch = [d[0] for d in self.buffer]
                s_j1_batch = [d[3] for d in self.buffer]
                s_j_loader = DataLoader(s_j_batch, batch_size=self.nstep)
                for batch in s_j_loader:
                    s_j_batch = batch
                r_batch = [d[2] for d in self.buffer]
                value_j = [d[6] for d in self.buffer]

                discount_rewards = []
                ret = last_value
                for i in reversed(range(len(self.buffer))):
                    terminal = self.buffer[i][4]
                    ret = r_batch[i] + self.GAMMA * ret * (1.0-terminal)
                    discount_rewards.append(ret)
                discount_rewards = discount_rewards[::-1]

                a_batch = np.array([])
                y_adv_batch = np.array([])
                mask_batch = np.array([])
                for i in range(0, len(self.buffer)):
                    action_space = self.buffer[i][5]
                    act = self.buffer[i][1]
                    a_batch = np.append(a_batch, act)
                    node_space = len(act)
                    temp_mask = np.zeros(node_space)
                    temp_mask[-action_space:] = 1
                    temp_y = np.zeros(node_space)
                    index = np.argmax(act)
                    # get policy loss
                    temp_y[index] = discount_rewards[i] - value_j[i]
                    y_adv_batch = np.append(y_adv_batch, temp_y)
                    mask_batch = np.append(mask_batch, temp_mask)

                # perform gradient step
                self.train(s_j_batch, a_batch, mask_batch, discount_rewards, y_adv_batch,
                            device, policy_net, value_net, optimizer)
                temp_loss_data.append([self.step_t, self.temp_loss])
                self.buffer.clear()

            print("TIMESTEP", self.step_t,
                  "/ Loss", self.temp_loss, 
                  "/ EXPLORED", env.status(), "/ REWARD", r_t, "/ Terminal", current_done, "\n")#"/ Entropy", self.entro,

            if done:
                del env
                gc.collect()
                env = robot.ExplorationEnv(self.map_size, 0, Test)
                done = False

            data_all = data_all.append({"Step": self.step_t, "Reward": r_t, "Loss": self.temp_loss, "Entropy": self.entro, "Terminal": current_done}, ignore_index=True)
            self.total_reward = np.append(self.total_reward, r_t)

            # save progress every 50000 iterations
            if self.step_t % 5e4 == 0:
                torch.save(policy_net.state_dict(), self.weights_path + 'MyModel.pt')
            if self.step_t > 1000:
                new_average_reward = np.average(self.total_reward[len(self.total_reward) - 1000:])
                if self.step_t % 1e2 == 0:
                    temp_reward_data.append([self.step_t, new_average_reward])

        np.savetxt(self.object_path + "temp_reward.csv", temp_reward_data, delimiter=",")
        np.savetxt(self.object_path + "temp_loss.csv", temp_loss_data, delimiter=",")
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)
        torch.save(policy_net.state_dict(), self.object_path + 'Model_Policy.pt')
        torch.save(value_net.state_dict(), self.object_path + 'Model_Value.pt')

    def data_process(self, data, device):
        s_a, s_x = data#a邻接矩阵 x特征
        edge_index = []#编号
        edge_attr = []#值
        edge_set = set()#边的集合
        for a_i in range(np.shape(s_a)[0]):#读取邻接矩阵第一维度的长度,相当于行数
            for a_j in range(np.shape(s_a)[1]):
                if (a_i, a_j) in edge_set or (a_j, a_i) in edge_set or s_a[a_i][a_j] == 0:#如果已提取边的特征或者没有边相连就进入下一次循环
                    continue
                edge_index.append([a_i, a_j])#加入边的编号[0,1]
                edge_attr.append(s_a[a_i][a_j])#加入边的值
                if a_i != a_j:#非对角线
                    edge_index.append([a_j, a_i])#加入对称的边的编号
                    edge_attr.append(s_a[a_j][a_i])#加入对称的边的值
                edge_set.add((a_i, a_j))#加入边的集合{(0,1)}
                edge_set.add((a_j, a_i))#加入边的集合{(1,0)}
        edge_index = torch.tensor(np.transpose(edge_index), dtype=torch.long)#tensor([[0,1,0,2,0,3...],[1,0,2,0,3,0...]])
        x = torch.tensor(s_x, dtype=torch.float)#特征array--->tensor
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)#tensor([5.3639,...])
        state = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)#Data返回tensor的维度
        batch = torch.zeros(np.shape(s_a)[0], dtype=int).to(device)#tensor([0,0,0,0,0,0,0,0])
        return state, batch

    def policy_cost(self, prob, advantages, action, mask):
        prob_flat = prob.view(-1)
        advantages_flat = advantages.view(-1)
        advantages_flat = torch.masked_select(advantages_flat, mask)
        action = torch.masked_select(action, mask)
        log_prob = prob_flat.log()
        policy_loss = -torch.mul(log_prob, advantages_flat)
        policy_loss = torch.mul(policy_loss, action).sum() / self.nstep
        return policy_loss

    def value_cost(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        loss = F.mse_loss(pred_flat, target_flat)
        return loss

    def entropy_loss(self, prob):
        prob_flat = prob.view(-1).detach()
        entro = -torch.mul(prob_flat.log(), prob_flat).sum() / self.nstep#-(logp1+logp2+...+logpn)/n
        self.entro = entro.item()
        return entro

    def train(self, data, action, mask, dis_reward, y_adv,
              device, modelA, modelC, optimizer):
        modelA.train()
        modelC.train()
        data = data.to(device)
        mask = torch.tensor(mask, dtype=bool).to(device)
        optimizer.zero_grad()
        actor_out = modelA(data, mask, batch=data.batch)
        critic_out = modelC(data, mask, batch=data.batch)
        eps = 1e-35
        actor_out = actor_out + eps
        y_adv = torch.tensor(y_adv).to(device).to(torch.float32)
        dis_reward = torch.tensor(dis_reward).to(device).to(torch.float32)
        action = torch.tensor(action).to(device).to(torch.float32)
        actor_loss = self.policy_cost(actor_out, y_adv, action, mask)
        critic_loss = self.value_cost(critic_out, dis_reward)#值和回报之间的区别A
        entropy_loss = self.entropy_loss(actor_out)
        loss = actor_loss - entropy_loss * self.ent_coef + critic_loss * self.vf_coef#actor_loss + 均方误差 * β - entropy_loss * η 
        self.temp_loss = loss.item()
        loss.backward()
        params = list(modelA.parameters()) + list(modelC.parameters())
        for param in params:
            param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
        optimizer.step()
    #data: Data(x=[8, 5], edge_index=[2, 22], edge_attr=[22]) batch: tensor([0,0,0,0,0,0,0,0]) mask: array([0,0,0,0,0,0,1,1]) device: cpu model: policyGCN
    def test(self, data, batch, mask, device, model):
        model.eval()
        data = data.to(device)
        mask = torch.tensor(mask, dtype=bool).to(device)#array([false,false,false,false,false,false,true,true])转成布尔类型
        pred = model(data, mask, batch)
        return pred


if __name__ == "__main__":
    case_path = "test_case"
    training = A2C(case_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modela = Networks.PolicyGCN()
    modelc = Networks.ValueGCN()
    modela.to(device)
    modelc.to(device)
    training.running(modela, modelc)
