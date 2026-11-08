"""Typed optional editable surface. No callable names, paths or executable values."""
import math
import re

PPO = {'gamma':('gamma',float,0,1),'lam':('lam',float,0,1),'clipParam':('clip_param',float,.01,.5),'entropyCoef':('entropy_coef',float,0,1),'valueLossCoef':('value_loss_coef',float,0,10),'desiredKl':('desired_kl',float,.00001,1),'maxGradNorm':('max_grad_norm',float,.01,10),'numLearningEpochs':('num_learning_epochs',int,1,20),'numMiniBatches':('num_mini_batches',int,1,64)}
GAINS={'kpFirmware':('kp_fw',float,1,1000),'voltageMinimum':('vin_min',float,1,12),'delayMinLag':('delay_min_lag',int,0,20),'delayMaxLag':('delay_max_lag',int,0,20)}
GROUPS={'ppo','network','rewards','curriculum','observations','events','physics','gains','calibration','commands','teacherCheckpoints'}


def numeric(value, kind=float, low=-1000000, high=1000000):
    if type(value) not in ((int,) if kind is int else (int,float)) or not math.isfinite(value) or not low<=value<=high: raise ValueError('WORKBENCH_CONFIG_INVALID')


def numeric_tree(value,depth=0):
    if depth>8:raise ValueError('WORKBENCH_CONFIG_INVALID')
    if type(value) is dict and len(value)<=128:
        for k,v in value.items():
            if not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,79}',k):raise ValueError('WORKBENCH_CONFIG_INVALID')
            numeric_tree(v,depth+1)
    elif type(value) is list and len(value)<=128:
        for v in value:numeric_tree(v,depth+1)
    elif type(value) is bool:pass
    else:numeric(value)


def validate_groups(config):
    for group,value in config.items():
        if group not in GROUPS:continue
        if type(value) is not dict or len(value)>128:raise ValueError('WORKBENCH_CONFIG_INVALID')
        if group in ('ppo','gains'):
            fields=PPO if group=='ppo' else GAINS
            if value.keys()-fields.keys():raise ValueError('WORKBENCH_CONFIG_INVALID')
            for key,n in value.items():numeric(n,*fields[key][1:])
        elif group=='network':
            if value.keys()-{'actorHiddenDims','criticHiddenDims','activation','initialStd'}:raise ValueError('WORKBENCH_CONFIG_INVALID')
            for k,v in value.items():
                if k.endswith('HiddenDims'):
                    if type(v) is not list or not 1<=len(v)<=4:raise ValueError('WORKBENCH_CONFIG_INVALID')
                    for n in v:numeric(n,int,16,1024)
                elif k=='activation':
                    if v not in ('elu','relu','tanh'):raise ValueError('WORKBENCH_CONFIG_INVALID')
                else:numeric(v,float,.01,5)
        elif group=='teacherCheckpoints':
            if value.keys()-{'walk','stand'}:raise ValueError('WORKBENCH_CONFIG_INVALID')
            for ref in value.values():
                if type(ref) is not dict or set(ref)!={'jobId','checkpointSha256','configDigest'}:raise ValueError('WORKBENCH_CONFIG_INVALID')
                if not re.fullmatch('[-A-Za-z0-9_]{1,80}',ref['jobId']):raise ValueError('WORKBENCH_CONFIG_INVALID')
                for key in ('checkpointSha256','configDigest'):
                    if not re.fullmatch('sha256:[a-f0-9]{64}',ref[key]):raise ValueError('WORKBENCH_CONFIG_INVALID')
        else:numeric_tree(value)
