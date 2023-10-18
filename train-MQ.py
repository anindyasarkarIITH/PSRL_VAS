"""
Train MQ-VAS Policy
"""
import os
import random
import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import argparse
from torch.autograd import Variable
#from tensorboard_logger import configure, log_value
from torch.distributions import Bernoulli
from torch.distributions.categorical import Categorical
from copy import deepcopy as c

from utils_c import utils, utils_detector
from constants import base_dir_metric_cd, base_dir_metric_fd
from constants import num_actions

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser(description='PolicyNetworkTraining')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') 
parser.add_argument('--data_dir', default='/home/research/data_path/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--img_size', type=int, default=448, help='PN Image Size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=8, help='Number of Workers')
parser.add_argument('--test_epoch', type=int, default=2, help='At every N epoch test the network')
parser.add_argument('--parallel', action='store_true', default=True, help='use multiple GPUs for training')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--beta', type=float, default=0.1, help='Coarse detector increment')
parser.add_argument('--sigma', type=float, default=0.5, help='cost for patch use')
args = parser.parse_args("")

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)

def coord(x):
    x_= x//6 + 1  #7
    y_= x%6       #7
    return (x_,y_)
     
## Adaptive-vas training function
def train(epoch):
    # select a random search budget within a range at the start of every training epochs
    search_budget = random.randrange(12,19,3) 
    batch_query = 3
    # set the agent in training mode
    agent.train()
    # initialize lists that holds the record of search performance
    rewards, total_reward, policies = [], [], []
    ## Iterate over training data
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs = Variable(inputs)
        if not args.parallel:
            inputs = inputs.cuda()

        # stores the information of previous search queries
        search_info = torch.zeros((int(inputs.shape[0]), num_actions)).cuda() 
        #Stores the information of other queried location in a batch
        loc_query_batch = torch.zeros((int(inputs.shape[0]), num_actions)).cuda()
        # stores the information of previously selected grids as target 
        mask_info = torch.ones((int(inputs.shape[0]), num_actions)).cuda()
        #store the information about the remaining query
        query_info = torch.zeros(int(inputs.shape[0])).cuda()
        
        # Start an episode
        policy_loss = []; search_history = []; ce_loss = []
        p_loss = torch.zeros(int(inputs.shape[0])).cuda()
        store_policy_out = []
        adv_store = [] ; log_prob_store = []
        # Find the loss for only the policy network
        loss_static = nn.BCEWithLogitsLoss() #nn.BCELoss()#CrossEntropyLoss()
        
        # Iterate untill search budget diminishes
        for step_ in range(search_budget):
            
            if (step_%batch_query == 0):
                updated_search_info = search_info
                
            query_remain = search_budget - step_
            #store the information about the remaining query
            query_left = torch.add(query_info, query_remain).cuda()
            # agents performs based on the following information: input, previous search history and the number of query left, other queried loc
            logit, grid_prob = agent.forward(inputs, updated_search_info, query_left, loc_query_batch) 
            grid_prob_net = grid_prob.view(grid_prob.size(0), -1)
            alpha_hp = np.clip(args.alpha + epoch * 0.001, 0.6, 0.95)
            grid_prob_out = grid_prob_net*alpha_hp + (1-alpha_hp) * (1-grid_prob_net)
            
            loss_cls = loss_static(grid_prob_net.float(), targets.float().cuda()) 
            ce_loss.append(loss_cls)           
            # Apply a softmax in order to obtain a probability distribution over grids
            probs = F.softmax(logit, dim = 1)
            # we mask the probability distribution and assign 0 probability to the grids that is already selected by the agent 
            mask_probs = probs * mask_info.clone() 
            # Sample the policies from the Categorical distribution characterized by agent
            distr = Categorical(mask_probs)
            policy_sample = distr.sample()
            store_policy_out.append(policy_sample)

            # Random policy - used as baseline policy in the training step
            policy_map = torch.randint(0, num_actions, (int(inputs.shape[0]),)) 
            
            # Find the reward for the baseline policy
            reward_map = utils.compute_reward(targets, policy_map.data, args.beta, args.sigma)
            # Find the reward for the sampled policy
            reward_sample = utils.compute_reward(targets, policy_sample.data, args.beta, args.sigma)
            rewards.append(reward_sample)
            for sample_id in range(int(inputs.shape[0])):
                # Update the search history based on the current reward 
                loc_query_batch[sample_id, int(policy_sample[sample_id].data)] = 1
                if (int(reward_sample[sample_id]) == 1):
                     search_info[sample_id, int(policy_sample[sample_id].data)] = int(reward_sample[sample_id])
                else:
                     search_info[sample_id, int(policy_sample[sample_id].data)] = -1
                # Update the mask info based on the current action taken by the agent
                mask_info[sample_id, int(policy_sample[sample_id].data)] = 0
            # Compute the advantage value
            advantage = reward_sample.cuda().float() - reward_map.cuda().float()
            adv_store.append(advantage)
            # Find the loss for only the policy network
            loss = -distr.log_prob(policy_sample)           
            log_prob_store.append(loss)
            # Final loss according to REINFORCE objective to train the policy/agent
            loss = loss * Variable(advantage).expand_as(policy_sample)
            policy_loss.append(loss)
            
        b = [] ; c = [] ; policy_loss = []; temp = []
        temp = torch.zeros(int(inputs.shape[0])).cuda()
        for t in range(search_budget)[::-1]:   
            adv_store[t] = adv_store[t] + (0.01) * temp  #0.01
            temp = adv_store[t]     
        for log_prob, disc_return in zip(log_prob_store, adv_store): # returns
            policy_loss.append(log_prob * Variable(disc_return).expand_as(log_prob))   #added Variable        
        # Cut the episode length by cost budget,by assigning the loss = grad_log_pi*furure_discounted_reward = 0
        for sample in range(int(inputs.shape[0])):
            flag = False
            remain_cost = 200
            last_visited_grid = {0: [], 1: [], 2: []} # the number of id = batch query number
            for time_step in range(len(store_policy_out)):
                agent_id = int(time_step % batch_query)
                if flag == False:
                    grid_id = int(store_policy_out[time_step][sample]) 
                    p1, p2 = coord(grid_id)
                    if len(last_visited_grid[agent_id]) == 0:
                        p1_last, p2_last = coord(grid_id)           
                        distance = abs(p1-p1_last) + abs(p2 - p2_last)
                        last_visited_grid[agent_id].append((p1, p2))
                    else:
                        p1_last = last_visited_grid[agent_id][-1][0]
                        p2_last = last_visited_grid[agent_id][-1][1]
                        distance = abs(p1-p1_last) + abs(p2 - p2_last)
                        last_visited_grid[agent_id].append((p1, p2))
                    remain_cost = remain_cost - distance
                    p1_last, p2_last = p1, p2
                    if remain_cost < 0:
                        policy_loss[time_step][sample] = 0
                        flag = True
                else:
                    policy_loss[time_step][sample] = 0
        loss_rl = torch.cat(policy_loss).mean() 
        loss_ce = torch.mean(torch.stack(ce_loss))
        loss = loss_rl + (0.1) * loss_ce  
        # update the policy network parameters 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # store the search result in the following lists
        batch_reward = torch.cat(rewards).mean() 
        total_reward.append(batch_reward.cpu())
        policies.append(policy_sample.data.cpu())
    
    reward = utils.performance_stats_search(policies, total_reward)
    with open('log.txt','a') as f:
        f.write('Train: %d | Rw: %.2f \n' % (epoch, reward))
    
    print('Train: %d | Rw: %.2E | CE: %.2E | RL: %.2E' % (epoch, reward, loss_ce, loss_rl))

# test the agent's performance on adaptive-VAS setting    
def test(epoch, best_sr): 
    search_budget = random.randrange(12,19,3) 
    batch_query = 3
    # set the agent in evaluation mode
    agent.eval()
    # initialize lists to store search outcomes
    targets_found, metrics, policies, set_labels, num_targets, num_search = list(), [], [], [], list(), list()
    acc_steps = []; tpr_steps =[];
    # iterate over the test data
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs = Variable(inputs, volatile=True)
        if not args.parallel:
            inputs = inputs.cuda()
        
        # stores the information of previous search queries
        search_info = torch.zeros((int(inputs.shape[0]), num_actions)).cuda()
        #Stores the information of other queried location in a batch
        loc_query_batch = torch.zeros((int(inputs.shape[0]), num_actions)).cuda()
        # stores the information of previously selected grids as target 
        mask_info = torch.ones((int(inputs.shape[0]), num_actions)).cuda()
        #store the information about the remaining query
        query_info = torch.zeros(int(inputs.shape[0])).cuda()

        # Start an episode
        policy_loss = []; search_history = []; reward_history = [];
        
        for step_ in range(search_budget): 
            if (step_%batch_query == 0):
                updated_search_info = search_info 
                
            query_remain = search_budget - step_
            # number of query left
            query_left = torch.add(query_info, query_remain).cuda()
            # action taken by agent
            logit, grid_prob = agent.forward(inputs, updated_search_info, query_left, loc_query_batch) 
            grid_prob = F.sigmoid(grid_prob)
            grid_prob_net = grid_prob.view(grid_prob.size(0), -1)
            
            # get the prediction of target from the agents intermediate output
            policy_pred = grid_prob_net.data.clone()
            policy_pred[policy_pred<0.5] = 0.0
            policy_pred[policy_pred>=0.5] = 1.0
            policy_pred = Variable(policy_pred)
            
            ### prediction result statistics
            acc, tpr = utils.acc_calc(targets, policy_pred.data)
            acc_steps.append(acc)
            tpr_steps.append(tpr)
            
            # get the probability distribution over grids
            probs = F.softmax(logit, dim=1)
            # assign 0 probability to those grids that is already queried by agent
            mask_probs = probs * mask_info.clone()  
            # Sample the grid that corresponds to highest probability of being target
            policy_sample = torch.argmax(mask_probs, dim=1) 
            # compute the reward for the agent's action
            reward_update = utils.compute_reward(targets, policy_sample.data, args.beta, args.sigma)
            # get the outcome of an action in order to compute ESR/SR 
            reward_sample = utils.compute_reward_batch(targets, policy_sample.data, args.beta, args.sigma)
            
            # Update search info and mask info after every query
            for sample_id in range(int(inputs.shape[0])):
                # Update the search history based on the current reward 
                loc_query_batch[sample_id, int(policy_sample[sample_id].data)] = 1
                if (int(reward_sample[sample_id]) == 1):
                     search_info[sample_id, int(policy_sample[sample_id].data)] = int(reward_sample[sample_id])
                else:
                     search_info[sample_id, int(policy_sample[sample_id].data)] = -1
                # Update the mask info based on the current action taken by the agent
                mask_info[sample_id, int(policy_sample[sample_id].data)] = 0   
            # store the episodic reward in the list
            reward_history.append(reward_sample)   
        # obtain the number of target grid in each sample (maximum value is search budget)
        for target_id in range(int(targets.shape[0])):
            temp = int(targets[target_id,:].sum())
            if temp > search_budget:
                num_targets.append(search_budget)
            else:
                num_targets.append(temp)

        # concat the episodic reward over the samples in a batch
        batch_reward = torch.cat(reward_history).sum() 
        targets_found.append(batch_reward)    
    ## compute ESR
    temp_recall = torch.sum(torch.stack(targets_found))
    recall = temp_recall / sum(num_targets)  
    final_acc = torch.mean(torch.stack(acc_steps))
    final_tpr = torch.mean(torch.stack(tpr_steps))
    
    # store the log in different log file
    if (search_budget == 12):
        with open('log12.txt','a') as f:
            f.write('Test - Recall: %.2f | ACC: %.2f | TPR: %.2f \n' % (recall, final_acc, final_tpr))
    elif (search_budget == 15):
        with open('log15.txt','a') as f:
            f.write('Test - Recall: %.2f | ACC: %.2f | TPR: %.2f \n' % (recall, final_acc, final_tpr))
    else:
        with open('log18.txt','a') as f:
            f.write('Test - Recall: %.2f | ACC: %.2f | TPR: %.2f \n' % (recall, final_acc, final_tpr))
        if (recall> best_sr):
            print ("best_SR for SB 18 is:", recall)
            best_sr = recall
            # save the model --- agent
            agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
            state = {
              'agent': agent_state_dict,
              'epoch': epoch,
              'reward': recall,
            }
            # uncomment the following line and provide a path where you want to save the trained model
            torch.save(state,"/storage1/model_vas_xview99_sc_adapt_F_batch")
    
    print('Test - Recall: %.2E | SB: %.2F' % (recall,search_budget))
    return best_sr
    
#--------------------------------------------------------------------------------------------------------#
# Get the data from dataloader
trainset, testset = utils.get_datasetVIS(args.img_size, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torchdata.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers)

# initialize the agent
agent = utils.Model_search_Arch_Adapt_batch()  
# ---- Load the pre-trained model ----------------------
""" uncomment this if we want to start training from a pretrained model
checkpoint = torch.load("path_stored_pretrained_model")
agent.load_state_dict(checkpoint['agent'])                                                      
start_epoch = checkpoint['epoch'] + 1
print('loaded agent from %s' % args.load)
"""

start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print('loaded agent from %s' % args.load)

# Parallelize the models if multiple GPUs available - Important for Large Batch Size to Reduce Variance
if args.parallel:
    agent = nn.DataParallel(agent)
agent.cuda()

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)
best_sr = 0.0

# Start training and testing
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch % args.test_epoch == 0:        
        best_sr = test(epoch, best_sr)
        


