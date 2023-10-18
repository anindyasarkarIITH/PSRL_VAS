
"""
Test MPS-VAS Policy
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
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
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
    x_= x//6 + 1   #7, 5, 9  8
    y_= x%6        #7, 6, 9  6
    return (x_,y_)    
    
# test the agent's performance on VAS setting    
def test(epoch, best_sr): 
    search_budget = random.randrange(12,19,3)  #35, 50, 75
    num_image = 0
    # set the agent in evaluation mode
    search_Agent.eval()
    pred_Agent.eval()
    # initialize lists to store search outcomes
    targets_found, metrics, policies, set_labels, num_targets, num_search = list(), [], [], [], list(), list()
    acc_steps = []; tpr_steps =[];
    # iterate over the test data
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        num_image += 1
        ## create a copy of the pretrained agent
        p_Agent = c(pred_Agent)
        optimizer = optim.Adam(p_Agent.parameters(), lr=0.0002) #1 foe 12,15 2 for 18,15 0.0002
        inputs = Variable(inputs, volatile=True)
        if not args.parallel:
            inputs = inputs.cuda()
        # stores the information of previous search queries
        search_info = torch.zeros((int(inputs.shape[0]), num_actions)).cuda()
        # Active Target Information representation
        act_target_label = torch.zeros((int(inputs.shape[0]), num_actions)).cuda()
        final_target_label = torch.zeros((int(inputs.shape[0]), num_actions)).cuda()
        # stores the information of previously selected grids as target 
        mask_info = torch.ones((int(inputs.shape[0]), num_actions)).cuda()
        #store the information about the remaining query
        query_info = torch.zeros(int(inputs.shape[0])).cuda()

        # Start an episode
        policy_loss = []; search_history = []; reward_history = [];
        travel_cost = remain_cost =75
        
        # Find the loss for only for the prediction policy network
        loss_static = nn.BCEWithLogitsLoss()
        
        for step_ in range(search_budget): 
            query_remain = search_budget - step_
            # number of query left
            query_left = torch.add(query_info, query_remain).cuda()
            # action taken by agent
            grid_prob_ = p_Agent.forward(inputs, search_info, query_left)
            logit = search_Agent.forward(inputs, search_info, query_left, grid_prob_)
            grid_prob_net = grid_prob_.view(grid_prob_.size(0), -1)
            
            grid_prob = F.sigmoid(grid_prob_net)
            
            # get the prediction of target from the agents intermediate output
            policy_pred = grid_prob_net.data.clone()
            policy_pred[policy_pred<0.5] = 0.0
            policy_pred[policy_pred>=0.5] = 1.0
            policy_pred = Variable(policy_pred)
            
            acc, tpr = utils.acc_calc(targets, policy_pred.data)
            acc_steps.append(acc)
            tpr_steps.append(tpr)
            
            # get the probability distribution over grids
            probs = F.softmax(logit, dim=1)
            
            # assign 0 probability to those grids that is already queried by agent
            mask_probs = probs * mask_info.clone()  
            
            # Sample the grid that corresponds to highest probability of being target
            policy_sample = torch.argmax(mask_probs, dim=1) 
            
            ###### cost travel
            p1, p2 = coord(int(policy_sample))
            if (step_ == 0):
                p1_last, p2_last = coord(int(policy_sample))
            
            distance = abs(p1-p1_last) + abs(p2 - p2_last)
            remain_cost = remain_cost - distance
            p1_last, p2_last = p1, p2
            if remain_cost < 0:
                break
            ################# travel
            
            # compute the reward for the agent's action
            reward_update = utils.compute_reward(targets, policy_sample.data, args.beta, args.sigma)

            # get the outcome of an action in order to compute ESR/SR 
            reward_sample = utils.compute_reward_batch(targets, policy_sample.data, args.beta, args.sigma)
            
            # Update search info and mask info after every query
            for sample_id in range(int(inputs.shape[0])):
                # update the search info based on the reward
                if int(reward_update[sample_id]) == 1:
                     search_info[sample_id, int(policy_sample[sample_id].data)] = int(reward_update[sample_id])
                else:
                     search_info[sample_id, int(policy_sample[sample_id].data)] = -1
                        
                if (int(reward_update[sample_id]) == 1):
                    act_target_label[sample_id, int(policy_sample[sample_id].data)] = 1
                else: 
                    act_target_label[sample_id, int(policy_sample[sample_id].data)] = 0
                
                # update the mask info based on the current action
                mask_info[sample_id, int(policy_sample[sample_id].data)] = 0
                
                for out_idx in range(num_actions):
                    if (mask_info[sample_id, out_idx] == 0):
                        final_target_label[sample_id, out_idx] = act_target_label[sample_id, out_idx].data
                    else:
                        final_target_label[sample_id, out_idx] = grid_prob[sample_id, out_idx].data

            # store the episodic reward in the list
            reward_history.append(reward_sample)
            
            loss_cls = loss_static(grid_prob_net.float(), final_target_label.float().cuda()) 
            # update the policy network parameters 
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
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
        
    ## compute ESR/ANT
    temp_recall = torch.sum(torch.stack(targets_found))
    recall = temp_recall / sum(num_targets)  
    sr_budget = temp_recall / num_image
    print ("travel sr:", sr_budget)
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
            
    
    print('Test - Recall: %.2E | SB: %.2F' % (recall,search_budget))
    return best_sr
 
  

#--------------------------------------------------------------------------------------------------------#
#trainset, testset = utils.get_dataset(args.img_size, args.data_dir)
trainset, testset = utils.get_datasetVIS_Classwise(args.img_size, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# initialize the agent
pred_Agent = utils.Model_search_Arch_Adapt_pred()
search_Agent = utils.Model_search_Arch_Adapt_search_meta()

# ---- Load the pre-trained model ----------------------
search_checkpoint = torch.load("/storage1//model_vas_dota36_lv_adapt_F_meta_search")
search_Agent.load_state_dict(search_checkpoint['agent'])                                                      
start_epoch = search_checkpoint['epoch'] + 1
print('loaded agent from %s' % args.load)

pred_checkpoint = torch.load("/storage1/model_vas_dota36_lv_adapt_F_meta_pred")
pred_Agent.load_state_dict(pred_checkpoint['agent'])  


# Parallelize the models if multiple GPUs available - Important for Large Batch Size to Reduce Variance
if args.parallel:
    
    pred_Agent = nn.DataParallel(pred_Agent)
    search_Agent = nn.DataParallel(search_Agent)

pred_Agent.cuda()
search_Agent.cuda()

best_sr = 0.0

# Start training and testing
for epoch in range(1):
    
    best_sr = test(epoch, best_sr)
        
