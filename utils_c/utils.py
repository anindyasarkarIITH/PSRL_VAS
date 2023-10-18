import os
import torch
import torchvision.transforms as transforms
import torchvision.models as torchmodels
import torch.nn.functional as F
import numpy as np
import shutil
#import torch_geometric.data as data
#from torch_geometric.nn import knn_graph
import json

from utils_c import utils_detector
from utils_c.ResNet import ResNetCifar as ResNet
from dataset.dataloader import CustomDatasetFromImages, CustomDatasetFromImagesTest, CustomDatasetFromImagesTestFM, CustomDatasetFromImagesTestVIS,CustomDatasetFromImagesTest_Classwise
from constants import base_dir_groundtruth, base_dir_detections_cd, base_dir_detections_fd, base_dir_metric_cd, base_dir_metric_fd
from constants import num_windows, img_size_fd, img_size_cd
#from vit_pytorch import ViT

device = torch.device('cpu')
if torch.cuda.is_available():
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
print (device)

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def read_json(filename):
    with open(filename) as dt:
        data = json.load(dt)
    return data

def xywh2xyxy(x):
    y = np.zeros(x.shape)
    y[:,0] = x[:, 0] - x[:, 2] / 2.
    y[:,1] = x[:, 1] - x[:, 3] / 2.
    y[:,2] = x[:, 0] + x[:, 2] / 2.
    y[:,3] = x[:, 1] + x[:, 3] / 2.
    return y

def get_detected_boxes(policy, file_dirs, metrics, set_labels):
    for index, file_dir_st in enumerate(file_dirs):
        counter = 0
        for xind in range(num_windows):
            for yind in range(num_windows):
                # ---------------- Read Ground Truth ----------------------------------
                outputs_all = []
                gt_path = '{}/{}_{}_{}.txt'.format(base_dir_groundtruth, file_dir_st, xind, yind)
                if os.path.exists(gt_path):
                    gt = np.loadtxt(gt_path).reshape([-1, 5])
                    targets = np.hstack((np.zeros((gt.shape[0], 1)), gt))
                    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                    # ----------------- Read Detections -------------------------------
                    if policy[index, counter] == 1:
                        preds_dir = '{}/{}_{}_{}'.format(base_dir_detections_fd, file_dir_st, xind, yind)
                        targets[:, 2:] *= img_size_fd
                        if os.path.exists(preds_dir):
                            preds = np.loadtxt(preds_dir).reshape([-1,7])
                            outputs_all.append(torch.from_numpy(preds))
                    else:
                        preds_dir = '{}/{}_{}_{}'.format(base_dir_detections_cd, file_dir_st, xind, yind)
                        targets[:, 2:] *= img_size_cd
                        if os.path.exists(preds_dir):
                            preds = np.loadtxt(preds_dir).reshape([-1,7])
                            outputs_all.append(torch.from_numpy(preds))
                    set_labels += targets[:, 1].tolist()
                    metrics += utils_detector.get_batch_statistics(outputs_all, torch.from_numpy(targets), 0.5)
                else:
                    continue
                counter += 1

    return metrics, set_labels

def read_offsets(image_ids, num_actions):
    offset_fd = torch.zeros((len(image_ids), num_actions)).cuda()
    offset_cd = torch.zeros((len(image_ids), num_actions)).cuda()
    for index, img_id in enumerate(image_ids):
        offset_fd[index, :] = torch.from_numpy(np.loadtxt('{}/{}'.format(base_dir_metric_fd, img_id)).flatten())
        offset_cd[index, :] = torch.from_numpy(np.loadtxt('{}/{}'.format(base_dir_metric_cd, img_id)).flatten())

    return offset_fd, offset_cd

def performance_stats(policies, rewards):
    # Print the performace metrics including the average reward, average number
    
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return reward, num_unique_policy, variance, policy_set

def performance_stats_search(policies, rewards):
    # Print the performace metrics including the average reward, average number
    
    policies = torch.cat(policies, 0)
    
    reward = sum(rewards)
    

    return reward

def compute_reward(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    #print ("target:", targets.size())
    #print (policy.size())
    ## compute reward for correct search
    reward = torch.zeros(int(targets.shape[0])).to(device)
    for sample_id in range(int(targets.shape[0])):
        
        if (targets[sample_id, int(policy[sample_id])] == 1):
            reward[sample_id] = 1 #2
        else:
            reward[sample_id] = 0 #1
    #print (reward.size())
    return reward

def compute_reward_(targets, policy):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    #print ("target:", targets.size())
    #print (policy.size())
    ## compute reward for correct search
    reward = torch.zeros(int(targets.shape[0])).to(device)
    for sample_id in range(int(targets.shape[0])):
        
        if (targets[sample_id, int(policy[sample_id])] == 1):
            reward[sample_id] = 1
        else:
            reward[sample_id] = -1
    #print (reward.size())
    return reward    

def compute_reward_latest(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    # for incorrect search reward is 0 instead of 1 (trial)
    reward = torch.zeros(int(targets.shape[0])).cuda()
    for sample_id in range(int(targets.shape[0])):
        if (targets[sample_id, int(policy[sample_id])] == 1):
            reward[sample_id] = 1
        else:
            reward[sample_id] = 0 #-1
    
    return reward

def compute_reward_greedy(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    temp_re = torch.eq(targets.cuda(), policy).long()
    temp_re[temp_re==0] = -1
    reward = torch.sum(temp_re, dim=1).unsqueeze(1).float()
    return reward



def compute_reward_batch(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    reward = torch.zeros(int(targets.shape[0])).cuda()
    for sample_id in range(int(targets.shape[0])):
        if (targets[sample_id, int(policy[sample_id])] == 1):
            reward[sample_id] = 1
        else:
            reward[sample_id] = 0
    
    return reward

def compute_reward_batch_topk(targets, policy, batch_query, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    reward = torch.zeros(int(targets.shape[0]), batch_query).cuda()
    for sample_id in range(int(targets.shape[0])):
        for batch_idx in range(batch_query):
            #print ("!!!",policy)
            if (targets[sample_id, int(policy[sample_id][batch_idx])] == 1):
                reward[sample_id, batch_idx] = 1
            else:
                reward[sample_id, batch_idx] = 0
    
    return reward    
    

def compute_reward_search(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    if (targets[0,int(policy)]== 0):
        reward = torch.tensor(-1).float()
    else:
        reward = torch.tensor(1).float()
    
    return reward.reshape(1)
def compute_reward_search_test(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    if (targets[0,int(policy)]== 0):
        reward = torch.tensor(0).float()
    else:
        reward = torch.tensor(1).float()
    
    return reward.reshape(1)
   
    
def compute_reward_test(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    
    temp_re = torch.mul(targets.cuda(), policy)
    target_found = torch.sum(temp_re)
    num_targets = torch.sum(targets.cuda())
    total_search = torch.sum(policy.cuda())
    
    return target_found, num_targets, total_search 

def acc_calc(targets, policy):
    
    correct = torch.sum(policy.cuda() == targets.cuda())
    total = targets.shape[0] * targets.shape[1]
    val = correct/total
    num_targets = torch.sum(targets.cuda())
    confusion_vector = policy.cuda() / targets.cuda()
    true_positives = torch.sum(confusion_vector == 1).item()
    tpr = true_positives/num_targets
    return val, tpr
 
def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(img_size), #Scale
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size), #Scale
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test

def get_dataset(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_chip300.csv', transform_train) 
    
    testset = CustomDatasetFromImagesTest(root+'train_chip300.csv', transform_test) 
    

    return trainset, testset

def get_dataset_DOTA_TTT(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_dota.csv', transform_train) 
    
    testset = CustomDatasetFromImagesTest(root+'train_dota.csv', transform_test) 
    

    return trainset, testset

def get_datasetFM(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_dota_ship.csv', transform_train) 
    
    testset = CustomDatasetFromImagesTestFM(root+'train_dota_ship.csv', transform_test) 

    return trainset, testset

#'/home/research/'
def get_datasetVIS(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)

    # use first 70% data for training
    trainset = CustomDatasetFromImages(root+'train_sc_xview_99_.csv', transform_train) 
    #trainset = CustomDatasetFromImages(root+'train_dota.csv', transform_train) 
    
    # use remaining 30% data for testing
    testset = CustomDatasetFromImagesTest(root+'train_sc_xview_99_.csv', transform_test) 
    #testset = CustomDatasetFromImagesTest(root+'train_dota.csv', transform_test) 
    
    return trainset, testset
#/home/research/
def get_datasetVIS_Classwise(img_size, root='/home/research/'):
    transform_train, transform_test = get_transforms(img_size)

    # use first 70% data for training train_helicoptar_xview49_500.csv  train_helipad_xview49_500.csv
    trainset = CustomDatasetFromImages(root+'train_sb_xview_99_.csv', transform_train) 
    
    # use remaining 30% data for testing  train_helipad_xview49_500.csv
    testset = CustomDatasetFromImagesTest_Classwise(root+'train_sb_xview_99_.csv', transform_test) 
    
    return trainset, testset

def get_datasetVISx(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_dota_ship.csv', transform_train) 
      
    testset = CustomDatasetFromImagesTestVIS(root+'train_dota_ship.csv', transform_test) 

    return trainset, testset

def get_datasetVIS_(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train_dota.csv', transform_train) 
    
    
    testset = CustomDatasetFromImagesTestVIS(root+'train_dota.csv', transform_test) 
    

    return trainset, testset

def get_datasetTTA(img_size, root='/home/research/Visual_Active_Search_Project/'):
    transform_train, transform_test = get_transforms(img_size)
    
    trainset = CustomDatasetFromImages(root+'train_building_tta.csv', transform_train)
    
    testset = CustomDatasetFromImagesTestVIS(root+'train_building_tta.csv', transform_test)
    
    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def node_embedding():
    res34_model = torchmodels.resnet34(pretrained=True)
    agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
    for param in agent.parameters():
        param.requires_grad = False
    
    return agent
        
def get_model_():
    '''
    res34_model = torchmodels.resnet34(pretrained=True)
    agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
    for param in agent.parameters():
        param.requires_grad = False
    '''
    agent = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, 49)
    
    return agent
"""
def get_model(num_output):
    agent = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)
    
    return agent
"""
## policy network using multi-head attention
class Model_search_Arch_MHA(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_MHA, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 30, 1)  # 1*1 convolution
        self.pointwise = torch.nn.Conv2d(60, 3, 1, 1)
        self.v = ViT(
                image_size = 14,
                patch_size = 2,
                num_classes = 30,
                dim = 512,
                depth = 2,
                heads = 4,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1)


    def forward(self, x, search_info):
        feat_ext = self.agent(x)
        reduced_feat =  F.relu(self.conv1(feat_ext))
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)
        
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ### MHA
        logits = self.v(combined_feat)
        
        return logits
    

def build_model():
    #from ResNet import ResNetCifar as ResNet
    print('Building model...')

    def gn_helper(planes):
        return torch.nn.GroupNorm(8, planes)
    norm_layer = gn_helper

    net = ResNet(26, 1, channels=3, classes=30, norm_layer=norm_layer).cuda()
    return net
    
# feat ext
class Feat_Ext(torch.nn.Module):
    def __init__(self):
        super(Feat_Ext, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60 
        ## Input feature extractor
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        #self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 49, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
    def forward(self, x):
        # Input feature extraction
        feat_ext = self.agent(x)

        reduced_feat = self.maxpool(feat_ext)  #apply maxpool stride = 2
        
        return reduced_feat


# MPS-VAS (meta) (lv dota 36 grid size) 
class Model_search_Arch_Adapt_pred(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_Adapt_pred, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60 
        ## Input feature extractor
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 36, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   
        
        self.pointwise = torch.nn.Conv2d(72, 3, 1, 1) #61
        
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 90),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(90, 36),    #60
        )
        
    def forward(self, x, search_info, query_left):
        # Input feature extraction
        feat_ext = self.agent(x)
        # feature squezing using 1x1 conv 
        
        reduced_feat_resnet =  F.relu(self.conv1(feat_ext))
        
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
         
        # tile remaining query budget information (b)
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 7, 7)
        #print (query_info_tile.shape)
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)  #, query_info_tile
        
        ## apply 1x1 conv on the combined feature representation
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ## apply 1*1 conv on the search info feature representation
        #search_feat_map = F.relu(self.pointwise_search_info(search_info_tile))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape)
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        #print (logits.shape, )
        grid_prob = logits #F.sigmoid(logits)
        
        return grid_prob        

    
# MPS-VAS  (meta) (lv dota 36 grid size)
class Model_search_Arch_Adapt_search_meta(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_Adapt_search_meta, self).__init__()
        
        
        self.pointwise_search_info = torch.nn.Conv2d(36, 1, 1, 1)
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(72, 49),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(49, 36),    #60
        )

    def forward(self, x, search_info, query_left, grid_prob):
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 6, 6)
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, 6, 6)
        
        ## apply 1*1 conv on the search info feature representation
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile))
        
        rl_rep = torch.cat((map_grid_prob, search_feat_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl  
        

"""
         
# MPS-VAS (meta) (sc xview 99 grid size) 
class Model_search_Arch_Adapt_pred(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_Adapt_pred, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60 
        ## Input feature extractor
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 99, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
       
        
        self.pointwise = torch.nn.Conv2d(198, 3, 1, 1) #61
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 128),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(128, 99),    #60
        )
       
    def forward(self, x, search_info, query_left):
        # Input feature extraction
        feat_ext = self.agent(x)
        # feature squezing using 1x1 conv
        
        
        reduced_feat_resnet =  F.relu(self.conv1(feat_ext))
        
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2
        
        
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        
        
        # tile remaining query budget information (b)
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 7, 7)
        #print (query_info_tile.shape)
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)  #, query_info_tile
        
        ## apply 1x1 conv on the combined feature representation
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape)
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        #print (logits.shape, )
        grid_prob = logits #F.sigmoid(logits)
        
        return grid_prob        

    
# MPS-VAS (meta) (sc xview 99 grid size)
class Model_search_Arch_Adapt_search_meta(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_Adapt_search_meta, self).__init__()
        
        
        self.pointwise_search_info = torch.nn.Conv2d(99, 1, 1, 1)
        #self.pointwise_pred = torch.nn.Conv2d(98, 3, 1, 1) #60
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(198, 164),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(164, 99),    #60
        )

    def forward(self, x, search_info, query_left, grid_prob):
        
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 11, 9)
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, 11, 9)
        #target = target.to(torch.float32)
        #map_grid_prob = target.view(target.shape[0], 1, 7, 7)
        ## apply 1*1 conv on the search info feature representation
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile))
        
        rl_rep = torch.cat((map_grid_prob, search_feat_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl  
        
        
      

# PSVAS (small car xview 49 grid size)
class Model_search_Arch_Adapt(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_Adapt, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60 
        ## Input feature extractor
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 49, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.pointwise = torch.nn.Conv2d(98, 3, 1, 1) #61
        
        self.pointwise_search_info = torch.nn.Conv2d(49, 1, 1, 1)
        #self.pointwise_pred = torch.nn.Conv2d(98, 3, 1, 1) #60
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 90),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(90, 49),    #60
        )
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(98, 49),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(49, 49),    #60
        )

    def forward(self, x, search_info, query_left):
        # Input feature extraction
        feat_ext = self.agent(x)
        # feature squezing using 1x1 conv
        
        
        reduced_feat_resnet =  F.relu(self.conv1(feat_ext))
        
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2
        
        
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        
        
        # tile remaining query budget information (b)
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 7, 7)
        #print (query_info_tile.shape)
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)  #, query_info_tile
        
        ## apply 1x1 conv on the combined feature representation
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ## apply 1*1 conv on the search info feature representation
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape)
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        #print (logits.shape, )
        grid_prob = logits #F.sigmoid(logits)
        #grid_prob = torch.mul(logits, mask_info.clone()) 
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, 7, 7)
        #target = target.to(torch.float32)
        #map_grid_prob = target.view(target.shape[0], 1, 7, 7)
        rl_rep = torch.cat((map_grid_prob, search_feat_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl, grid_prob    


# PSVAS (large vehicle dota 64 grid size)
class Model_search_Arch_Adapt(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_Adapt, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60 
        ## Input feature extractor
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 64, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.pointwise = torch.nn.Conv2d(128, 3, 1, 1) #61
        
        self.pointwise_search_info = torch.nn.Conv2d(64, 1, 1, 1)
        #self.pointwise_pred = torch.nn.Conv2d(98, 3, 1, 1) #60
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 90),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(90, 64),    #60
        )
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(128, 84),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(84, 64),    #60
        )

    def forward(self, x, search_info, query_left):
        # Input feature extraction
        feat_ext = self.agent(x)
        # feature squezing using 1x1 conv
        
        
        reduced_feat_resnet =  F.relu(self.conv1(feat_ext))
        #print (reduced_feat_resnet.shape)
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2
        #print (reduced_feat.shape)  # 64, 7 , 7
        
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        #print (search_info_tile.shape) #64, 7, 7
        
        search_info_tile_search_module = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 8, 8)
        
        # tile remaining query budget information (b)
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 7, 7)
        #print (query_info_tile.shape) # 1, 7, 7
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)  #, query_info_tile
        
        ## apply 1x1 conv on the combined feature representation
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ## apply 1*1 conv on the search info feature representation
        #search_feat_map = F.relu(self.pointwise_search_info(search_info_tile))
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile_search_module))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape)
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        #print (logits.shape, )
        grid_prob = logits #F.sigmoid(logits)
        #grid_prob = torch.mul(logits, mask_info.clone()) 
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, 8, 8)
        #target = target.to(torch.float32)
        #map_grid_prob = target.view(target.shape[0], 1, 7, 7)
        rl_rep = torch.cat((map_grid_prob, search_feat_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl, grid_prob 
    

"""
# PSVAS (large vehicle dota 36 grid size)
class Model_search_Arch_Adapt(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_Adapt, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60 
        ## Input feature extractor
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 36, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.pointwise = torch.nn.Conv2d(72, 3, 1, 1) #61
        
        self.pointwise_search_info = torch.nn.Conv2d(36, 1, 1, 1)
        #self.pointwise_pred = torch.nn.Conv2d(98, 3, 1, 1) #60
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 90),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(90, 36),    #60
        )
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(72, 48),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(48, 36),    #60
        )

    def forward(self, x, search_info, query_left):
        # Input feature extraction
        feat_ext = self.agent(x)
        # feature squezing using 1x1 conv
        
        
        reduced_feat_resnet =  F.relu(self.conv1(feat_ext))
        #print (reduced_feat_resnet.shape)
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2
        #print (reduced_feat.shape)  # 64, 7 , 7
        
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        #print (search_info_tile.shape) #64, 7, 7
        
        search_info_tile_search_module = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 6, 6)
        
        # tile remaining query budget information (b)
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 6, 6)
        #print (query_info_tile.shape) # 1, 7, 7
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)  #, query_info_tile
        
        ## apply 1x1 conv on the combined feature representation
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ## apply 1*1 conv on the search info feature representation
        #search_feat_map = F.relu(self.pointwise_search_info(search_info_tile))
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile_search_module))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape)
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        #print (logits.shape, )
        grid_prob = logits #F.sigmoid(logits)
        #grid_prob = torch.mul(logits, mask_info.clone()) 
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, 6, 6)
        #target = target.to(torch.float32)
        #map_grid_prob = target.view(target.shape[0], 1, 7, 7)
        rl_rep = torch.cat((map_grid_prob, search_feat_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl, grid_prob        
 
"""
# PSVAS (large vehicle xview 99 grid size)
class Model_search_Arch_Adapt(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_Adapt, self).__init__()
        resnet_embedding_sz = 512
        pointwise_in_channels = 60 
        ## Input feature extractor
        res34_model = torchmodels.resnet34(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res34_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 99, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.pointwise = torch.nn.Conv2d(198, 3, 1, 1) #61
        
        self.pointwise_search_info = torch.nn.Conv2d(99, 1, 1, 1)
        #self.pointwise_pred = torch.nn.Conv2d(98, 3, 1, 1) #60
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 128),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(128, 99),    #60
        )
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(198, 164),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(164, 99),    #60
        )

    def forward(self, x, search_info, query_left):
        # Input feature extraction
        feat_ext = self.agent(x)
        # feature squezing using 1x1 conv
        
        
        reduced_feat_resnet =  F.relu(self.conv1(feat_ext))
        #print (reduced_feat_resnet.shape)
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2
        #print (reduced_feat.shape)  # 64, 7 , 7
        
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        #print (search_info_tile.shape) #64, 7, 7
        
        search_info_tile_search_module = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 11, 9)
        
        # tile remaining query budget information (b)
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 11, 9)
        #print (query_info_tile.shape) # 1, 7, 7
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)  #, query_info_tile
        
        ## apply 1x1 conv on the combined feature representation
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ## apply 1*1 conv on the search info feature representation
        #search_feat_map = F.relu(self.pointwise_search_info(search_info_tile))
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile_search_module))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape)
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        #print (logits.shape, )
        grid_prob = logits #F.sigmoid(logits)
        #grid_prob = torch.mul(logits, mask_info.clone()) 
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, 11, 9)
        #target = target.to(torch.float32)
        #map_grid_prob = target.view(target.shape[0], 1, 7, 7)
        rl_rep = torch.cat((map_grid_prob, search_feat_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl, grid_prob   

# PSVAS  (small car xview 49 grid size)
class Model_search_Arch_Adapt_search(torch.nn.Module):
    def __init__(self):
        super(Model_search_Arch_Adapt_search, self).__init__()
        
        self.pointwise_search_info = torch.nn.Conv2d(49, 1, 1, 1)
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(98, 49),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(49, 49),    #60
        )

    def forward(self, x, search_info, query_left, grid_prob):
        
         # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, 7, 7)
        #target = target.to(torch.float32)
        #map_grid_prob = target.view(target.shape[0], 1, 7, 7)
        ## apply 1*1 conv on the search info feature representation
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile))
        
        rl_rep = torch.cat((map_grid_prob, search_feat_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl 
"""


