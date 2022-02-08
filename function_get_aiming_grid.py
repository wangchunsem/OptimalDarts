import os
import sys
import numpy as np
import time

import function_board as fb
import function_tool as ft


#%%
R = fb.R ## radius of the dartboard 170
grid_num = fb.grid_num ## 341

## 2-dimension probability grid
def load_aiming_grid(playername_filename, data_parameter_dir=fb.data_parameter_dir, grid_version=fb.grid_version, count_bull=True):
    """
    Load 2-dimensional numpy arrays of hitting probability from files
    Each row of aiming_grid is the (x-index, y-index) of an aiming location. 
    For each aiming location, the corresponding row in prob_grid_singlescore (same row index as that in aiming_grid) contains the hitting probability of score S1,...,S20.    
    (prob_grid_doublescore for D1,...,D20, prob_grid_triplescore for T1,...,T20,, prob_grid_bullscore for SB,DB)
    prob_grid_normalscore has 61 columns and contains the aggregated hitting probability of score 0,1...,60. For example, P(score 18) = P(S18) + P(D9) + P(T6)

    """
    
    if playername_filename.startswith('player'):
        filename = data_parameter_dir+'/grid_{}/{}_gaussin_prob_grid_{}.pkl'.format(grid_version, playername_filename, grid_version)
    else:    
        filename = playername_filename    

    result_dic = ft.load_pickle(filename, printflag=True)
    aiming_grid = result_dic['aiming_grid']
    prob_grid_normalscore = result_dic['prob_grid_normalscore'] 
    prob_grid_singlescore = result_dic['prob_grid_singlescore']
    prob_grid_doublescore = result_dic['prob_grid_doublescore']
    prob_grid_triplescore = result_dic['prob_grid_triplescore']
    prob_grid_bullscore = result_dic['prob_grid_bullscore']
    
    ## default setting counts bull score
    if count_bull:
        prob_grid_normalscore[:,fb.score_SB] += prob_grid_bullscore[:,0]
        prob_grid_normalscore[:,fb.score_DB] += prob_grid_bullscore[:,1]
    else:
        print('bull score in NOT counted in prob_grid_normalscore')
        
    return [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore]

## 3-dimension probability grid
def load_prob_grid(playername_filename, data_parameter_dir=fb.data_parameter_dir, grid_version='full'):
    """
    Load 3-dimensional numpy arrays of size 341*341*si (the 340mmX340mm square grid enclosing the dartboard).
    Generate prob_grid_normalscore which has 61 columns and contains the aggregated hitting probability of score 0,1...,60. For example, P(score 18) = P(S18) + P(D9) + P(T6)
    """
    
    if playername_filename.startswith('player'):
        filename = data_parameter_dir+'/grid_{}/{}_gaussin_prob_grid.pkl'.format(grid_version, playername_filename)
    else:    
        filename = playername_filename        
    
    prob_grid_dict = ft.load_pickle(filename, printflag=True)    
    prob_grid_singlescore = prob_grid_dict['prob_grid_singlescore']
    prob_grid_doublescore = prob_grid_dict['prob_grid_doublescore']
    prob_grid_triplescore = prob_grid_dict['prob_grid_triplescore']
    prob_grid_bullscore = prob_grid_dict['prob_grid_bullscore']
    
    ## prob_grid_singlescore has 61 columns and contains the aggregated hitting probability of score 0,1...,60. For example, P(score 18) = P(S18) + P(D9) + P(T6)
    ## Bull score is NOT included yet !!
    prob_grid_normalscore = np.zeros((grid_num, grid_num, 61))
    for temp_s in range(1,61):
        if temp_s <= 20:
            prob_grid_normalscore[:,:,temp_s] = prob_grid_singlescore[:,:,temp_s-1]
        if temp_s%2 == 0 and temp_s <= 40:
            prob_grid_normalscore[:,:,temp_s] = prob_grid_normalscore[:,:,temp_s] + prob_grid_doublescore[:,:,temp_s//2-1]
        if temp_s%3 == 0:
            prob_grid_normalscore[:,:,temp_s] = prob_grid_normalscore[:,:,temp_s] + prob_grid_triplescore[:,:,temp_s//3-1]
    ## prob of hitting zero
    prob_grid_normalscore[:,:,0] =  np.maximum(0, 1-prob_grid_normalscore[:,:,1:].sum(axis=2)-prob_grid_bullscore.sum(axis=2))
    
    return [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore]


## generate action set v2
def get_aiming_grid_v2(playername_filename, data_parameter_dir=fb.data_parameter_dir, grid_version='full'):
    """
    Select 984 aiming locations to build a small action set.
    Eighteen targets in each of the double and treble regions are permitted for a total of 720 double / treble targets. 
    A small square enclosing the SB and DB regions contains 9 × 9 = 81 additional targets. 
    Three targets located in each of the inner and outer and single regions are permitted. 
    This leads to a total of 720+81+120 = 921 common targets for each of the players. 
    For each particular player, we also include the target within each region, e.g. T20, S12 etc., that has the highest probability of being hit for that player. 
    Finally we also include the single point on the board that has the highest expected score which is generally in T20. 
    This means an additional 63 targets for each player.
    Totally 921+63=984 points.                
    """    
    
    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = load_prob_grid(playername_filename, data_parameter_dir=data_parameter_dir, grid_version=grid_version)    
    temp_num = 1000
    aiming_grid = np.zeros((temp_num,2), dtype=np.int32)
        

    ## x,y location in [-170,170]. x,y index in [0,340].
    ## center of DB is the first one 
    temp_index = 0
    aiming_grid[temp_index,0] = R
    aiming_grid[temp_index,1] = R
    
    ## square enclosing SB. 4mm*4mm grid
    for temp_x in range(-16,16+1,4):
        for temp_y in range(-16,16+1,4):
            if (temp_x==0 and temp_y==0):
                continue
            else:
                temp_index += 1
                aiming_grid[temp_index] = [temp_x+R, temp_y+R]

    ## inner single area and outer single area
    rgrid = [58,135]
    theta_num = 60
    for theta in range(theta_num):
        theta = np.pi*(2.0*theta/theta_num)
        for temp_r in rgrid:
            temp_index += 1 
            temp_x = int(np.round(np.cos(theta)*temp_r))
            temp_y = int(np.round(np.sin(theta)*temp_r))
            aiming_grid[temp_index] = [temp_x+R, temp_y+R]
    
    ## double and triple area
    rgrid = [100, 103, 106,  163, 166, 169]
    theta_num = 120
    for theta in range(theta_num):
        theta = np.pi*(2.0*theta/theta_num)
        for temp_r in rgrid:
            temp_index += 1 
            temp_x = int(np.round(np.cos(theta)*temp_r))
            temp_y = int(np.round(np.sin(theta)*temp_r))
            aiming_grid[temp_index] = [temp_x+R, temp_y+R]
    
    ## all players share the above common points. 
    ## the following points are different among players
            
    ## maximize probability point for each score region
    ## single area
    for temp_s in range(fb.singlescorelist_len):
        temp_index += 1        
        temp_s_argmax = np.argmax(prob_grid_singlescore[:,:,temp_s]) 
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]
    ## double area
    for temp_s in range(fb.doublescorelist_len):
        temp_index += 1        
        temp_s_argmax = np.argmax(prob_grid_doublescore[:,:,temp_s]) 
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]    
    ## triple area
    for temp_s in range(fb.triplescorelist_len):
        temp_index += 1        
        temp_s_argmax = np.argmax(prob_grid_triplescore[:,:,temp_s]) 
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]    
    ## bull    
    for temp_s in range(fb.bullscorelist_len):
        temp_index += 1
        temp_s_argmax = np.argmax(prob_grid_bullscore[:,:,temp_s])
        temp_x = temp_s_argmax/grid_num
        temp_y = temp_s_argmax%grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]
    
    ## max expected score
    temp_index += 1
    e_score = prob_grid_normalscore.dot(np.arange(61)) + prob_grid_bullscore.dot(np.array([fb.score_SB, fb.score_DB]))
    max_e_score = np.max(e_score)
    temp_argmax = np.argmax(e_score)
    temp_x = temp_argmax/grid_num
    temp_y = temp_argmax%grid_num
    aiming_grid[temp_index] = [temp_x, temp_y]        
    print('max_e_score={}, max_e_score_index={}'.format(max_e_score, aiming_grid[temp_index]))

    ##[0, 340]
    aiming_grid_num = temp_index + 1
    aiming_grid = aiming_grid[:aiming_grid_num,:]
    aiming_grid = np.maximum(aiming_grid, 0)
    aiming_grid = np.minimum(aiming_grid, grid_num-1)

    ## return probability
    prob_grid_normalscore_new = np.zeros((aiming_grid_num, 61))
    prob_grid_singlescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_doublescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_triplescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_bullscore_new = np.zeros((aiming_grid_num, 2))    
    for temp_index in range(aiming_grid_num):
        prob_grid_normalscore_new[temp_index,:] = prob_grid_normalscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_singlescore_new[temp_index,:] = prob_grid_singlescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_doublescore_new[temp_index,:] = prob_grid_doublescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_triplescore_new[temp_index,:] = prob_grid_triplescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_bullscore_new[temp_index,:] = prob_grid_bullscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
    
    return [aiming_grid, prob_grid_normalscore_new, prob_grid_singlescore_new, prob_grid_doublescore_new, prob_grid_triplescore_new, prob_grid_bullscore_new]


## save action set v2
def save_aiming_grid_v2(playerID_list):    
    grid_version_result = 'v2'
    print('generate and save action set grid_version={}'.format(grid_version_result))
    for playerID in playerID_list:
        name_pa = 'player{}'.format(playerID)    
        [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = get_aiming_grid_v2(name_pa, data_parameter_dir=fb.data_parameter_dir, grid_version='full')
        
        postfix = ''
        info = 'SB={} DB={} R1={} postfix={} skillmodel=full grid_version={}'.format(fb.score_SB, fb.score_DB, fb.R1, postfix, grid_version_result)
        
        result_dic = {}
        result_dic['info'] = info
        result_dic['aiming_grid'] = aiming_grid
        result_dic['prob_grid_normalscore'] = prob_grid_normalscore
        result_dic['prob_grid_singlescore'] = prob_grid_singlescore
        result_dic['prob_grid_doublescore'] = prob_grid_doublescore
        result_dic['prob_grid_triplescore'] = prob_grid_triplescore
        result_dic['prob_grid_bullscore'] = prob_grid_bullscore
        
        result_dir = fb.data_parameter_dir + '/grid_{}'.format(grid_version_result)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)        
        result_filename = result_dir + '/{}_gaussin_prob_grid_{}.pkl'.format(name_pa, grid_version_result)
        ft.dump_pickle(result_filename, result_dic, printflag=True)
    
    print()
    return


## save action set v2
def save_aiming_grid_circleboard(playerID_list):    
    grid_version_result = 'circleboard'
    print('generate and save action set grid_version={}'.format(grid_version_result))
    
    grid_num_full = 341*341
    aiming_index = np.arange(0,grid_num_full, dtype=np.int32)
    aiming_grid = np.zeros((grid_num_full,2), dtype=np.int32)
    aiming_grid[:,0] = aiming_index/341
    aiming_grid[:,1] = aiming_index%341
    distance = ((aiming_grid - 170)**2).sum(axis=1)
    index = np.where(distance <= 170**2)[0]   ## indexes of aiming locations which are inside the circleboard 
    
    for playerID in playerID_list:
        name_pa = 'player{}'.format(playerID)    
        [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = load_prob_grid(name_pa, data_parameter_dir=fb.data_parameter_dir, grid_version='full')
        
        postfix = ''
        info = 'SB={} DB={} R1={} postfix={} skillmodel=full grid_version={}'.format(fb.score_SB, fb.score_DB, fb.R1, postfix, grid_version_result)
        
        aiming_grid1 = aiming_grid[index].copy()    
        prob_grid1_normalscore = prob_grid_normalscore.reshape((grid_num_full, 61))[index]
        prob_grid1_singlescore = prob_grid_singlescore.reshape((grid_num_full, 20))[index]
        prob_grid1_doublescore = prob_grid_doublescore.reshape((grid_num_full, 20))[index]
        prob_grid1_triplescore = prob_grid_triplescore.reshape((grid_num_full, 20))[index]
        prob_grid1_bullscore = prob_grid_bullscore.reshape((grid_num_full, 2))[index]
            
        result_dic = {}
        result_dic['info'] = info
        result_dic['aiming_grid'] = aiming_grid1
        result_dic['prob_grid_normalscore'] = prob_grid1_normalscore
        result_dic['prob_grid_singlescore'] = prob_grid1_singlescore
        result_dic['prob_grid_doublescore'] = prob_grid1_doublescore
        result_dic['prob_grid_triplescore'] = prob_grid1_triplescore
        result_dic['prob_grid_bullscore'] = prob_grid1_bullscore
        
        result_dir = fb.data_parameter_dir + '/grid_{}'.format(grid_version_result)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)        
        result_filename = result_dir + '/{}_gaussin_prob_grid_{}.pkl'.format(name_pa, grid_version_result)
        ft.dump_pickle(result_filename, result_dic, printflag=True)
    
    print()
    return
