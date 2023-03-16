import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.model_utils import *
from .raflow import *
from .cmflow import *
from .cmflow_t import *


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1 and classname.find('Conv1d_p') == -1:
        nn.init.kaiming_normal_(m.weight.data)
        
def init_model(args):
    
    if args.model in ['cmflow', 'cmflow_t', 'raflow']:
        if args.model in ['raflow']:
            net = RaFlow(args).cuda()
        if args.model in ['cmflow']:
            net = CMFlow(args).cuda()
        if args.model in ['cmflow_t']:
            net = CMFlow_T(args).cuda()
            
        if args.eval or args.load_checkpoint:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
            print("Successfully load model parameters!")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Experiment with", torch.cuda.device_count(), "GPUs!")
            
        return net
    
    else:
        raise Exception('Not implemented')
        
        
