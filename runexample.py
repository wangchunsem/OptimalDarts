import os
import sys
import numpy as np
import time

import function_board as fb
import function_tool as ft
from evaluate_score_probability_player import evaluate_score_probability
import function_get_aiming_grid
import function_solve_dp
#import function_solve_zsg
import function_solve_zsg_gpu as function_solve_zsg

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)
np.seterr(divide='ignore')

import warnings
warnings.filterwarnings('ignore')

#%%
data_parameter_dir = fb.data_parameter_dir
result_dir = './result'        
playerID_list = [1,2]
gamelist = [[1,2]]

## evaluate the hitting probability for each aiming location on the 1mm grid 
evaluate_score_probability(playerID_list)

## save the small action set which contains 984 aiming locations (grid_version='v2', used in the frist draft of the paper)
function_get_aiming_grid.save_aiming_grid_v2(playerID_list)


## save the action set which contains 90,785 aiming locations (grid_version='circleboard', used in the current draft of the paper)
function_get_aiming_grid.save_aiming_grid_circleboard(playerID_list)

#%%
## Choose the action set. 'v2' for testing purpose.
#grid_version = 'v2'
grid_version = 'circleboard'
postfix=''

#gpu_device = None ## Using CPU
gpu_device = 0 ## Using GPU 

## solve the single player dart game
for playerID in playerID_list:
    name_pa = 'player{}'.format(playerID)   
    res1 = function_solve_dp.solve_singlegame(name_pa, data_parameter_dir=data_parameter_dir, grid_version=grid_version, result_dir=result_dir, postfix=postfix, gpu_device=gpu_device)
    print()

#%%
## solve ZSG dart game between Player A and B
grid_version_pa = grid_version
grid_version_pb = grid_version
postfix='_{}'.format(grid_version)
for [pa, pb] in gamelist:
    name_pa = 'player{}'.format(pa)
    name_pb = 'player{}'.format(pb)
        
    resA = function_solve_zsg.solve_zsg_optA_fixNS(name_pa, name_pb, data_parameter_dir=data_parameter_dir, grid_version_pa=grid_version_pa, grid_version_pb=grid_version_pb, dp_policy_folder=result_dir, result_dir=result_dir, postfix=postfix, gpu_device=gpu_device)
    print(time.asctime( time.localtime(time.time())))
    print()
    
    if (pa!=pb):
        resB = function_solve_zsg.solve_zsg_optB_fixNS(name_pa, name_pb, data_parameter_dir=data_parameter_dir, grid_version_pa=grid_version_pa, grid_version_pb=grid_version_pb, dp_policy_folder=result_dir, result_dir=result_dir, postfix=postfix, gpu_device=gpu_device)
        print(time.asctime( time.localtime(time.time())))    
        print()
    
    resboth = function_solve_zsg.solve_zsg_optboth(name_pa, name_pb, data_parameter_dir=data_parameter_dir, grid_version_pa=grid_version_pa, grid_version_pb=grid_version_pb, dp_policy_folder=result_dir, result_dir=result_dir, postfix=postfix, gpu_device=gpu_device)
    print(time.asctime(time.localtime(time.time())))
    print()
