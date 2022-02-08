import os
import sys
import time

import function_board as fb
import function_tool as ft
import function_get_aiming_grid
import function_evaluate_policy as fep

import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)

import torch
torch.set_printoptions(precision=3)
torch.set_printoptions(linewidth=300)
torch.set_printoptions(threshold=300)

import function_solve_dp


#%%
game_begin_score_502 = 501+1
#game_begin_score_502 = 80
## end game state value
value_win_pa = 1 ## A win
value_win_pb = 0 ## A lose

#%%
## Player A plays with Player B
## value_pa[score_state_pa, score_state_pb] gives Player A's winning probability at the state (score_state_pa, score_state_pb, t=A, i=3, u=0), i.e., the beginning of a turn where Player A's score is score_state_pa, Player B's score is score_state_pb, and Player A throws in this turn.  
## value_pb[score_state_pa, score_state_pb] gives Player A's (not B!) winning probability at the state (score_state_pa, score_state_pb, t=B, i=3, u=0), i.e., the beginning of a turn where Player A's score is score_state_pa, Player B's score is score_state_pb, and Player B throws in this turn.  

def zsg_policy_evaluation(value_pa, value_pb, score_state_pa, score_state_pb, prob_turn_transit_pa, prob_turn_transit_pb):
    """
    Compute the game value (in terms of Player A's winning probability) for a specific turn (score_state_pa, score_state_pb) given Player A and B's policies. 
    Args: 
        value_pa, value_pb: game values for [s_a < score_state_pa, s_b < score_state_pb] are already solved. 
        score_state_pa, score_state_pb: scores for Player A and B at the beginning of this turn
        prob_turn_transit_pa, prob_turn_transit_pb: state transition probability associated with the given policies. 
    
    Returns: 
        value_state_pa: Player A's winning probability when A throws in this turn
        value_state_pb: Player A's winning probability when B throws in this turn
    """
    
    score_max_pa = min(score_state_pa-2, 3*fb.maxhitscore)
    score_max_pb = min(score_state_pb-2, 3*fb.maxhitscore)

    prob_score_pa = prob_turn_transit_pa['score']
    prob_score_pb = prob_turn_transit_pb['score']
    prob_finish_pa = prob_turn_transit_pa['finish']
    prob_finish_pb = prob_turn_transit_pb['finish']
    prob_zeroscore_pa = prob_turn_transit_pa['bust'] + prob_score_pa[0]
    prob_zeroscore_pb = prob_turn_transit_pb['bust'] + prob_score_pb[0]

    constant_pa = prob_finish_pa*value_win_pa
    constant_pa += np.dot(prob_score_pa[1:], value_pb[score_state_pa-1:score_state_pa-score_max_pa-1:-1, score_state_pb])        
    constant_pb = prob_finish_pb*value_win_pb #0
    constant_pb += np.dot(prob_score_pb[1:], value_pa[score_state_pa, score_state_pb-1:score_state_pb-score_max_pb-1:-1])
    value_state_pa = (constant_pa+constant_pb*prob_zeroscore_pa)/(1-prob_zeroscore_pa*prob_zeroscore_pb)
    value_state_pb = constant_pb + value_state_pa*prob_zeroscore_pb
    
    return [value_state_pa, value_state_pb]


def zsg_policy_improvement(param):
    """
    Do a policy improvement. Solve the Bellman equation. Find the best aiming location for each state in a turn using the last updated state values. 
    Args: 
        a dict param containing necessary informations. 
    
    Returns:     
        max_action_diff, max_value_relerror: relative errors after this policy iteration step 
    """
    
    #### input value ####
    prob_normalscore_tensor = param['prob_normalscore_tensor']
    prob_doublescore_dic = param['prob_doublescore_dic']
    prob_DB = param['prob_DB']
    prob_bust_dic = param['prob_bust_dic']
    
    state_len_vector = param['state_len_vector']
    score_state = param['score_state']    
    state_action = param['state_action']
    state_value = param['state_value']         
    state_action_update = param['state_action_update']
    state_value_update = param['state_value_update']
    action_diff = param['action_diff']
    value_relerror = param['value_relerror']
    
    flag_max = param['flag_max'] ## maximize or minimize 
    next_turn_value = param['next_turn_value']
    game_end_value = param['game_end_value']    
    if 'round_index' in param:
        round_index = param['round_index']
    else:
        round_index = 0

    #### policy improvement ####
    for rt in [1,2,3]:
        this_throw_state_len = state_len_vector[rt]
        ## state which can not bust.  score_state-score_gained>=62 
        state_notbust_len =  max(min(score_state-61, this_throw_state_len),0)
        if (state_notbust_len > 0):
            if (rt==1 and round_index==0):
                ## combine all non-bust states together 
                state_notbust_update_index = state_notbust_len                    
                next_state_value_array = np.zeros((61, state_notbust_len))                    
                for score_gained in range(state_notbust_len):
                    ## skip infeasible state
                    if not fb.state_feasible_array[rt, score_gained]:
                        continue
                    score_remain = score_state - score_gained
                    score_max = 60 ## always 60 here
                    score_max_plus1 = score_max + 1
                    next_state_value_array[:,score_gained] = next_turn_value[score_remain:score_remain-score_max_plus1:-1]
            elif (rt==2 and (round_index==0 or score_state<182)):
                ## combine all non-bust states together 
                state_notbust_update_index = state_notbust_len
                next_state_value_array = np.zeros((61, state_notbust_len))                    
                for score_gained in range(state_notbust_len):
                    ## skip infeasible state
                    if not fb.state_feasible_array[rt, score_gained]:
                        continue
                    score_remain = score_state - score_gained
                    score_max = 60 ## always 60 here
                    score_max_plus1 = score_max + 1
                    next_state_value_array[:,score_gained] = state_value_update[rt-1][score_gained:score_gained+score_max_plus1]
            else: ##(rt==1 and round_index>0) or (rt==2 and round_index>0 and score_state>=182) or (rt==3)
                ## only update state of score_gained = 0
                state_notbust_update_index = 1
                next_state_value_array = np.zeros(61)
                score_gained = 0
                score_remain = score_state - score_gained
                score_max = 60 ## always 60 here
                score_max_plus1 = score_max + 1                    
                ## make a copy
                if (rt > 1):
                    next_state_value_array[:] = state_value_update[rt-1][score_gained:score_gained+score_max_plus1]
                ## transit to next turn when rt=1
                else:
                    next_state_value_array[:] = next_turn_value[score_remain:score_remain-score_max_plus1:-1]

            ## matrix product to compute all together
            next_state_value_tensor = torch.from_numpy(next_state_value_array)
            win_prob_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)
            
            ## searching
            if flag_max:
                temp1 = win_prob_tensor.max(axis=0)
            else:
                temp1 = win_prob_tensor.min(axis=0)                    
            state_action_update[rt][0:state_notbust_update_index] = temp1.indices.numpy()
            state_value_update[rt][0:state_notbust_update_index] =  temp1.values.numpy()                
        
        ## state which possibly bust.  score_state-score_gained<62 
        if (state_notbust_len < this_throw_state_len):
            ## combine all bust states together 
            state_bust_len = this_throw_state_len - state_notbust_len
            next_state_value_array = np.zeros((61, state_bust_len))
            for score_gained in range(state_notbust_len, this_throw_state_len):
                ## skip infeasible state
                if not fb.state_feasible_array[rt, score_gained]:
                    continue
                score_remain = score_state - score_gained
                #score_max = min(score_remain-2, 60)
                score_max = score_remain-2 ## less than 60 here
                score_max_plus1 = score_max + 1
                score_gained_index = score_gained - state_notbust_len ## index off set
                if (rt > 1):
                    next_state_value_array[0:score_max_plus1,score_gained_index] = state_value_update[rt-1][score_gained:score_gained+score_max_plus1]
                ## transit to next turn when rt=1
                else:
                    next_state_value_array[0:score_max_plus1,score_gained_index] = next_turn_value[score_remain:score_remain-score_max_plus1:-1]
            
            ## matrix product to compute all together
            next_state_value_tensor = torch.from_numpy(next_state_value_array)
            win_prob_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)

            ## consider bust/finishing for each bust state seperately 
            win_prob_array = win_prob_tensor.numpy()                
            for score_gained in range(state_notbust_len, this_throw_state_len):
                ## skip infeasible state
                if not fb.state_feasible_array[rt, score_gained]:
                    continue
                score_remain = score_state - score_gained
                #score_max = min(score_remain-2, 60)
                score_max = score_remain-2 ## less than 60 here
                score_max_plus1 = score_max + 1
                score_gained_index = score_gained - state_notbust_len

                ## transit to the end of game
                if (score_remain == fb.score_DB):                        
                    win_prob_array[:,score_gained_index] += prob_DB*game_end_value
                elif (score_remain <= 40 and score_remain%2==0):
                    win_prob_array[:,score_gained_index] += prob_doublescore_dic[score_remain]*game_end_value
                else:
                    pass    
                ## transit to bust
                win_prob_array[:,score_gained_index] += prob_bust_dic[score_max]*next_turn_value[score_state]

            ## searching
            if flag_max:
                temp1 = win_prob_tensor.max(axis=0)
            else:
                temp1 = win_prob_tensor.min(axis=0)
            state_action_update[rt][state_notbust_len:this_throw_state_len] = temp1.indices.numpy()
            state_value_update[rt][state_notbust_len:this_throw_state_len] =  temp1.values.numpy()                

        #### finish rt=1,2,3. check improvement
        action_diff[rt][:] = np.abs(state_action_update[rt] - state_action[rt])                                
        value_relerror[rt] = np.abs((state_value_update[rt] - state_value[rt])/state_value_update[rt]).max()
        state_action[rt][:] = state_action_update[rt][:]
        state_value[rt][:] = state_value_update[rt][:]

    max_action_diff = max([action_diff[1].max(), action_diff[2].max(), action_diff[3].max()])
    max_value_relerror = value_relerror.max()

    return [max_action_diff, max_value_relerror]

def zsg_policy_improvement_gpu(param):
    
    #### input value ####
    prob_normalscore_tensor = param['prob_normalscore_tensor']
    prob_bust_tensor = param['prob_bust_tensor']
    prob_finish_tensor = param['prob_finish_tensor']
    gpu_device = param['gpu_device']

    state_len_vector = param['state_len_vector']
    score_state = param['score_state']    
    state_action = param['state_action']
    state_value = param['state_value']         
    state_action_update = param['state_action_update']
    state_value_update = param['state_value_update']
    action_diff = param['action_diff']
    value_relerror = param['value_relerror']
    
    flag_max = param['flag_max'] ## maximize or minimize 
    next_turn_value = param['next_turn_value']
    game_end_value = param['game_end_value']    
    if 'round_index' in param:
        round_index = param['round_index']
    else:
        round_index = 0
    
    #### policy improvement ####
    for rt in [1,2,3]:
        this_throw_state_len = state_len_vector[rt]
        ## state which can not bust.  score_state-score_gained>=62 
        state_notbust_len =  max(min(score_state-61, this_throw_state_len),0)
        if (state_notbust_len > 0):
            if (rt==1 and round_index==0):
                ## combine all non-bust states together 
                state_notbust_update_index = state_notbust_len                    
                next_state_value_array = np.zeros((61, state_notbust_len))                    
                for score_gained in range(state_notbust_len):
                    ## skip infeasible state
                    if not fb.state_feasible_array[rt, score_gained]:
                        continue
                    score_remain = score_state - score_gained
                    score_max = 60 ## always 60 here
                    score_max_plus1 = score_max + 1
                    next_state_value_array[:,score_gained] = next_turn_value[score_remain:score_remain-score_max_plus1:-1]
            elif (rt==2 and (round_index==0 or score_state<182)):
                ## combine all non-bust states together 
                state_notbust_update_index = state_notbust_len
                next_state_value_array = np.zeros((61, state_notbust_len))                    
                for score_gained in range(state_notbust_len):
                    ## skip infeasible state
                    if not fb.state_feasible_array[rt, score_gained]:
                        continue
                    score_remain = score_state - score_gained
                    score_max = 60 ## always 60 here
                    score_max_plus1 = score_max + 1
                    next_state_value_array[:,score_gained] = state_value_update[rt-1][score_gained:score_gained+score_max_plus1]
            else: ##(rt==1 and round_index>0) or (rt==2 and round_index>0 and score_state>=182) or (rt==3)
                ## only update state of score_gained = 0
                state_notbust_update_index = 1
                next_state_value_array = np.zeros(61)
                score_gained = 0
                score_remain = score_state - score_gained
                score_max = 60 ## always 60 here
                score_max_plus1 = score_max + 1                    
                ## make a copy
                if (rt > 1):
                    next_state_value_array[:] = state_value_update[rt-1][score_gained:score_gained+score_max_plus1]
                ## transit to next turn when rt=1
                else:
                    next_state_value_array[:] = next_turn_value[score_remain:score_remain-score_max_plus1:-1]

            ## matrix product to compute all together
            next_state_value_tensor = torch.from_numpy(next_state_value_array).cuda(gpu_device)
            win_prob_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)
            
            ## searching
            if flag_max:
                temp1 = win_prob_tensor.max(axis=0)
            else:
                temp1 = win_prob_tensor.min(axis=0)                    
            state_action_update[rt][0:state_notbust_update_index] = temp1.indices.cpu().numpy()
            state_value_update[rt][0:state_notbust_update_index] =  temp1.values.cpu().numpy()                
        
        ## state which possibly bust.  score_state-score_gained<62 
        if (state_notbust_len < this_throw_state_len):
            ## combine all bust states together 
            state_bust_len = this_throw_state_len - state_notbust_len
            next_state_value_array = np.zeros((61, 60))
            ## data position in next_state_value_array 
            score_gained_first_bust_state = state_notbust_len
            score_gained_last_bust_state = this_throw_state_len - 1 
            score_max_first_bust_state = score_state - score_gained_first_bust_state - 2
            score_max_last_bust_state = score_state - score_gained_last_bust_state - 2
                        
            ## loop score_gained backward and use score_max (from small to large) as index (consistant to prob_bust_tensor and prob_finish_tensor)
            for score_gained in range(this_throw_state_len-1, state_notbust_len-1, -1):
                ## skip infeasible state
                if not fb.state_feasible_array[rt, score_gained]:
                    continue
                score_remain = score_state - score_gained
                #score_max = min(score_remain-2, 60)
                score_max = score_remain-2 ## less than 60 here
                score_max_plus1 = score_max + 1
                if (rt > 1):
                    next_state_value_array[0:score_max_plus1,score_max] = state_value_update[rt-1][score_gained:score_gained+score_max_plus1]
                ## transit to next turn when rt=1
                else:
                    next_state_value_array[0:score_max_plus1,score_max] = next_turn_value[score_remain:score_remain-score_max_plus1:-1]
            
            ## matrix product to compute all together
            next_state_value_tensor = torch.from_numpy(next_state_value_array).cuda(gpu_device)
            win_prob_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)

            ## consider bust/finishing for all bust states together
            if (game_end_value != 0):
                win_prob_tensor += prob_finish_tensor*game_end_value
            ## transit to bust
            win_prob_tensor += prob_bust_tensor*next_turn_value[score_state]
            
            ## searching
            if flag_max:
                temp1 = win_prob_tensor.max(axis=0)
            else:
                temp1 = win_prob_tensor.min(axis=0)
            state_action_update[rt][state_notbust_len:this_throw_state_len] = temp1.indices.cpu().numpy()[range(score_max_first_bust_state, score_max_last_bust_state-1, -1)]
            state_value_update[rt][state_notbust_len:this_throw_state_len] =  temp1.values.cpu().numpy()[range(score_max_first_bust_state, score_max_last_bust_state-1, -1)]

        #### finish rt=1,2,3. check improvement
        action_diff[rt][:] = np.abs(state_action_update[rt] - state_action[rt])                                
        value_relerror[rt] = np.abs((state_value_update[rt] - state_value[rt])/state_value_update[rt]).max()
        state_action[rt][:] = state_action_update[rt][:]
        state_value[rt][:] = state_value_update[rt][:]

    max_action_diff = max([action_diff[1].max(), action_diff[2].max(), action_diff[3].max()])
    max_value_relerror = value_relerror.max()

    return [max_action_diff, max_value_relerror]


#%%
## fix player B's Naive Strategy (NS) and optimize player A        
def solve_zsg_optA_fixNS(name_pa, name_pb, data_parameter_dir=fb.data_parameter_dir, grid_version_pa=fb.grid_version, grid_version_pb=fb.grid_version, dp_policy_folder=None, result_dir=None, postfix='', gpu_device=None):
    info = 'A_{}_B_{}_optA: SB={} DB={} max_score={} R1={} grid_version_pa={} grid_version_pb={}'.format(name_pa, name_pb, fb.score_SB, fb.score_DB, fb.maxhitscore, fb.R1, grid_version_pa, grid_version_pb)
    print(info)
    ##
    if result_dir is not None:
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_filename = result_dir + '/zsg_A_{}_B_{}{}_optA.pkl'.format(name_pa, name_pb, postfix)
        result_value_filename = result_dir + '/zsg_value_A_{}_B_{}{}_optA.pkl'.format(name_pa, name_pb, postfix)
    
    #player A: pa throw first
    #player B: pb throw after player A, policy is fixed as ns
    print('player A is {} and player B is {}'.format(name_pa, name_pb))
    print('optimize player A policy and player B policy is fixed')
    [aiming_grid_pa, prob_grid_normalscore_pa, prob_grid_singlescore_pa, prob_grid_doublescore_pa, prob_grid_triplescore_pa, prob_grid_bullscore_pa] = function_get_aiming_grid.load_aiming_grid(name_pa, data_parameter_dir=data_parameter_dir, grid_version=grid_version_pa)
    [aiming_grid_pb, prob_grid_normalscore_pb, prob_grid_singlescore_pb, prob_grid_doublescore_pb, prob_grid_triplescore_pb, prob_grid_bullscore_pb] = function_get_aiming_grid.load_aiming_grid(name_pb, data_parameter_dir=data_parameter_dir, grid_version=grid_version_pb)
    
    ## use single player game as the fixed policy    
    dp_policy_dict_pa = None
    dp_policy_dict_pb = None
    if dp_policy_folder is not None:
        dp_policy_filename_pa = dp_policy_folder + '/singlegame_{}_turn_{}.pkl'.format(name_pa, grid_version_pa)
        if (os.path.isfile(dp_policy_filename_pa) == True):
            dp_policy_dict_pa = ft.load_pickle(dp_policy_filename_pa)
            print('load player A policy {}'.format(dp_policy_filename_pa))
        dp_policy_filename_pb = dp_policy_folder + '/singlegame_{}_turn_{}.pkl'.format(name_pb, grid_version_pb)
        if (os.path.isfile(dp_policy_filename_pb) == True):
            dp_policy_dict_pb = ft.load_pickle(dp_policy_filename_pb)
            print('load player B policy {}'.format(dp_policy_filename_pb))    
    if dp_policy_dict_pa is None:
        print('solve player A NS policy')
        dp_policy_dict_pa = function_solve_dp.solve_dp_turn(aiming_grid_pa, prob_grid_normalscore_pa, prob_grid_doublescore_pa, prob_grid_bullscore_pa)
    if dp_policy_dict_pb is None:
        print('solve player B NS policy')
        dp_policy_dict_pb = function_solve_dp.solve_dp_turn(aiming_grid_pb, prob_grid_normalscore_pb, prob_grid_doublescore_pb, prob_grid_bullscore_pb)

    #### data for player A ####
    num_aiming_location_pa = aiming_grid_pa.shape[0]
    prob_normalscore_pa = prob_grid_normalscore_pa
    prob_doublescore_dic_pa = {}
    for doublescore_index in range(20):
        doublescore = 2*(doublescore_index+1)
        prob_doublescore_dic_pa[doublescore] = np.array(prob_grid_doublescore_pa[:,doublescore_index])
    prob_DB_pa = np.array(prob_grid_bullscore_pa[:,1])
    
    ## the probability of not bust for each action given score_max=i (score_remain=i+2)
    prob_bust_dic_pa = {}
    prob_notbust_dic_pa = {}
    for score_max in range(60):    
        ## transit to next throw or turn
        prob_notbust_pa = prob_grid_normalscore_pa[:,0:score_max+1].sum(axis=1)
        ## transit to the end of game
        score_remain = score_max + 2
        if (score_remain == fb.score_DB):
            prob_notbust_pa += prob_DB_pa
        elif (score_remain <= 40 and score_remain%2==0):
            prob_notbust_pa += prob_doublescore_dic_pa[score_remain]
        ##
        prob_notbust_pa = np.minimum(np.maximum(prob_notbust_pa, 0),1)
        prob_notbust_dic_pa[score_max] = prob_notbust_pa
        prob_bust_dic_pa[score_max] = 1 - prob_notbust_dic_pa[score_max]

    param_pa = {}    
    if (gpu_device is None):
        function_policy_improvement = zsg_policy_improvement
        prob_normalscore_tensor_pa = torch.from_numpy(prob_normalscore_pa)
        param_pa['prob_normalscore_tensor'] = prob_normalscore_tensor_pa
        param_pa['prob_doublescore_dic'] = prob_doublescore_dic_pa
        param_pa['prob_DB'] = prob_DB_pa
        param_pa['prob_bust_dic'] = prob_bust_dic_pa
        
    else:
        print('using gpu_device={}'.format(gpu_device))
        function_policy_improvement = zsg_policy_improvement_gpu
        prob_normalscore_tensor_pa = torch.from_numpy(prob_normalscore_pa).cuda(gpu_device)
        prob_bust_array_pa = np.zeros((num_aiming_location_pa, 60))
        prob_finish_array_pa = np.zeros((num_aiming_location_pa, 60))        
        for score_max in range(60):
            prob_bust_array_pa[:,score_max] = prob_bust_dic_pa[score_max]
            ## transit to the end of game
            score_remain = score_max + 2
            if (score_remain == fb.score_DB):
                prob_finish_array_pa[:,score_max] += prob_DB_pa
            elif (score_remain <= 40 and score_remain%2==0):
                prob_finish_array_pa[:,score_max] += prob_doublescore_dic_pa[score_remain]            
        prob_bust_tensor_pa = torch.from_numpy(prob_bust_array_pa).cuda(gpu_device)
        prob_finish_tensor_pa = torch.from_numpy(prob_finish_array_pa).cuda(gpu_device)
        param_pa['prob_normalscore_tensor'] = prob_normalscore_tensor_pa
        param_pa['prob_bust_tensor'] = prob_bust_tensor_pa
        param_pa['prob_finish_tensor'] = prob_finish_tensor_pa
        param_pa['gpu_device'] = gpu_device

    #### 
    iteration_round_limit = 20
    iteration_relerror_limit = 10**-9
        
    value_pa = np.zeros((502,502))  # player A's winning probability when A throws at state [score_A, score_B]
    value_pb = np.zeros((502,502))  # player A's winning probability when B throws at state [score_A, score_B]
    value_win_pa = 1.0
    value_win_pb = 0.0
    num_iteration_record_pa = np.zeros((502,502), dtype=np.int8)
    
    state_len_vector_pa = np.zeros(4, dtype=np.int32)
    state_value_default  = [None]  ## expected # of turns for each state in the turn
    action_diff_pa  = [None]
    value_relerror_pa = np.zeros(4)
    for rt in [1,2,3]:
        ## for rt=3: possible score_gained = 0
        ## for rt=2: possible score_gained = 0,1,...,60
        ## for rt=1: possible score_gained = 0,1,...,120
        this_throw_state_len = fb.maxhitscore*(3-rt) + 1
        state_value_default.append(np.ones(this_throw_state_len)*fb.largenumber)
        action_diff_pa.append(np.ones(this_throw_state_len))
    
    ## for player A. (player B is fixed)
    ## first key: score_state_pa=2,...,501; second key: score_state_pb=2,...,501; thrid key: throws=3,2,1
    optimal_value_dic = {} 
    optimal_action_index_dic = {}
    prob_turn_transit_dic_pa = {}
    for score in range(2,502):
        optimal_value_dic[score] = {}
        optimal_action_index_dic[score] = {}
        prob_turn_transit_dic_pa[score] = {}
    
    #### algorithm start ####
    t_policy_improvement = 0
    t_policy_evaluation = 0
    t_other = 0
    t1 = time.time()
    for score_state_pb in range(2, game_begin_score_502):
        t_scoreloop_begin = time.time()
        score_state_list = []    
        ## fix player B score, loop through player A
        for score_state_pa in range(2, game_begin_score_502):
            score_state_list.append([score_state_pa, score_state_pb])
        
        ########     solve all states in turn [score_A, score_B]    ########
        for [score_state_pa, score_state_pb] in score_state_list:
            #print('##### score_state [score_pa, score_pb] = {} ####'.format([score_state_pa, score_state_pb]))
                
            ## initialize player A initial policy:
            for rt in [1,2,3]:        
                this_throw_state_len_pa = min(score_state_pa-2, fb.maxhitscore*(3-rt)) + 1
                state_len_vector_pa[rt] = this_throw_state_len_pa
            state_value_pa = ft.copy_numberarray_container(state_value_default)
            if score_state_pb > 2:
                state_action_pa = ft.copy_numberarray_container(optimal_action_index_dic[score_state_pa][score_state_pb-1])
                prob_turn_transit_pa = prob_turn_transit_dic_pa[score_state_pa][score_state_pb-1]
            else:
                state_action_pa = ft.copy_numberarray_container(dp_policy_dict_pa['optimal_action_index_dic'][score_state_pa])
                prob_turn_transit_pa = dp_policy_dict_pa['prob_scorestate_transit'][score_state_pa]
            state_value_update_pa = ft.copy_numberarray_container(state_value_pa)
            state_action_update_pa = ft.copy_numberarray_container(state_action_pa)

            ## player B, turn score transit probability is fixed
            prob_turn_transit_pb = dp_policy_dict_pb['prob_scorestate_transit'][score_state_pb]
                    
            ## assemble variables
            ## player A
            param_pa['state_len_vector'] = state_len_vector_pa
            param_pa['score_state'] = score_state_pa    
            param_pa['state_action'] = state_action_pa
            param_pa['state_value'] = state_value_pa
            param_pa['state_action_update'] = state_action_update_pa
            param_pa['state_value_update'] = state_value_update_pa    
            param_pa['action_diff'] = action_diff_pa
            param_pa['value_relerror'] = value_relerror_pa        
            ## maximize player A's win_prob
            param_pa['flag_max'] = True
            param_pa['next_turn_value'] = value_pb[:,score_state_pb] ## player B throws in next turn
            param_pa['game_end_value'] = value_win_pa
            
            ## policy iteration
            for round_index in range(iteration_round_limit):            
                
                #### policy evaluation ####
                tpe1 = time.time()
                ## evaluate current policy, player A winning probability at (score_pa, score_pb, i=3, u=0)
                ## value_pa: player A throws first, value_pb: player A throws second 
                ## player A, turn score transit probability                
                ## use the initial prob_turn_transit_pa value for round_index=0
                if (round_index >0):
                    prob_turn_transit_pa = fep.solve_turn_transit_probability_fast(score_state_pa, state_action_pa, prob_grid_normalscore_pa, prob_grid_doublescore_pa, prob_grid_bullscore_pa, prob_bust_dic_pa)
                [value_state_pa, value_state_pb] = zsg_policy_evaluation(value_pa, value_pb, score_state_pa, score_state_pb, prob_turn_transit_pa, prob_turn_transit_pb)
                value_pa[score_state_pa, score_state_pb] = value_state_pa
                value_pb[score_state_pa, score_state_pb] = value_state_pb
                tpe2 = time.time()
                t_policy_evaluation += (tpe2-tpe1)                 
                #print('evaluate rt3 value= {}'.format([value_pa[score_state_pa, score_state_pb], value_pb[score_state_pa, score_state_pb]]))
        
                #### policy improvement for player A ####
                tpi1 = time.time()
                param_pa['round_index'] = round_index
                [max_action_diff, max_value_relerror] = function_policy_improvement(param_pa)
                tpi2 = time.time()
                t_policy_improvement += (tpi2 - tpi1)                
                if (max_action_diff < 1):
                    break
                if (max_value_relerror < iteration_relerror_limit):
                    break
    
            optimal_action_index_dic[score_state_pa][score_state_pb] = state_action_pa
            optimal_value_dic[score_state_pa][score_state_pb] = state_value_pa
            prob_turn_transit_dic_pa[score_state_pa][score_state_pb] = prob_turn_transit_pa
            num_iteration_record_pa[score_state_pa, score_state_pb] = round_index + 1
            #### done:V(sa, sb,i=3/2/1,u)     

        #if (score_state_pb%20==0 or score_state_pb==2):
        #    print('#### score_state_pb={}, time={}'.format(score_state_pb, time.time()-t_scoreloop_begin))

    
    ## computation is done
    t2 = time.time()
    print('solve_zsg_opt_{}_fix_{} in {} seconds'.format(name_pa, name_pb, t2-t1))
    print('t_policy_evaluation  = {} seconds'.format(t_policy_evaluation))
    print('t_policy_improvement = {} seconds'.format(t_policy_improvement))
    print('t_other = {} seconds'.format(t_other))    
    #print('value_pa {} '.format(value_pa))
    #print('value_pb {} '.format(value_pb))
    
    result_dic = {'optimal_action_index_dic':optimal_action_index_dic, 'value_pa':value_pa, 'value_pb':value_pb,'optimal_value_dic':optimal_value_dic, 'info':info}    
    if (result_dir is not None):
        ft.dump_pickle(result_filename, result_dic)
        print('save {}'.format(result_filename))
        ft.dump_pickle(result_value_filename, {'value_pa':value_pa, 'value_pb':value_pb, 'info':info})
        print('save {}'.format(result_value_filename))
        return 'save'
    else:
        return result_dic


#%%
## fix player A's Naive Strategy (NS) and optimize player B
def solve_zsg_optB_fixNS(name_pa, name_pb, data_parameter_dir=fb.data_parameter_dir, grid_version_pa=fb.grid_version, grid_version_pb=fb.grid_version, dp_policy_folder=None, result_dir=None, postfix='', gpu_device=None):
    info = 'A_{}_B_{}_optB: SB={} DB={} max_score={} R1={} grid_version_pa={} grid_version_pb={}'.format(name_pa, name_pb, fb.score_SB, fb.score_DB, fb.maxhitscore, fb.R1, grid_version_pa, grid_version_pb)
    print(info)
    ##
    if result_dir is not None:    
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_filename = result_dir + '/zsg_A_{}_B_{}{}_optB.pkl'.format(name_pa, name_pb, postfix)
        result_value_filename = result_dir + '/zsg_value_A_{}_B_{}{}_optB.pkl'.format(name_pa, name_pb, postfix)
    
    ## need to reset the result key name: Player A is name_pb and Player B is name_pa
    ## game values are represented in terms of Player A's winning probability
    temp_result_dic = solve_zsg_optA_fixNS(name_pb, name_pa, data_parameter_dir=data_parameter_dir, grid_version_pa=grid_version_pb, grid_version_pb=grid_version_pa, dp_policy_folder=dp_policy_folder, gpu_device=gpu_device)
    result_dic = {'info':info}
    value_pa = 1-temp_result_dic['value_pb'].T    
    value_pb = 1-temp_result_dic['value_pa'].T
    value_pa[:2,:] = 0
    value_pa[:,:2] = 0
    value_pb[:2,:] = 0
    value_pb[:,:2] = 0
    
    result_dic['value_pa'] = value_pa
    result_dic['value_pb'] = value_pb    
    result_dic['optimal_action_index_dic'] = {}
    result_dic['optimal_value_dic'] = {}    
    for score_state_pa in range(2, game_begin_score_502):
    #for score_state_pa in range(2, 101):
        result_dic['optimal_action_index_dic'][score_state_pa] = {}
        result_dic['optimal_value_dic'][score_state_pa] = {}
        for score_state_pb in range(2, game_begin_score_502):
        #for score_state_pb in range(2, 101):
            result_dic['optimal_action_index_dic'][score_state_pa][score_state_pb] = temp_result_dic['optimal_action_index_dic'][score_state_pb][score_state_pa]
            result_dic['optimal_value_dic'][score_state_pa][score_state_pb] = temp_result_dic['optimal_value_dic'][score_state_pb][score_state_pa]
    
    if (result_dir is not None):
        ft.dump_pickle(result_filename, result_dic)
        print('save {}'.format(result_filename))
        ft.dump_pickle(result_value_filename, {'value_pa':value_pa, 'value_pb':value_pb, 'info':info})
        print('save {}'.format(result_value_filename))
        return 'save'
    else:
        return result_dic



#%%
## optimize player A and B alternatively until achieving optimal
def solve_zsg_optboth(name_pa, name_pb, data_parameter_dir=fb.data_parameter_dir, grid_version_pa=fb.grid_version, grid_version_pb=fb.grid_version, dp_policy_folder=None, result_dir=None, postfix='', gpu_device=None):
    info = 'A_{}_B_{}_optA: SB={} DB={} max_score={} R1={} grid_version_pa={} grid_version_pb={}'.format(name_pa, name_pb, fb.score_SB, fb.score_DB, fb.maxhitscore, fb.R1, grid_version_pa, grid_version_pb)
    print(info)
    ##
    if result_dir is not None:
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_filename = result_dir + '/zsg_A_{}_B_{}{}_optboth.pkl'.format(name_pa, name_pb, postfix)
        result_value_filename = result_dir + '/zsg_value_A_{}_B_{}{}_optboth.pkl'.format(name_pa, name_pb, postfix)
    
    #player A: pa throw first
    #player B: pb throw after player A, policy is fixed as ns
    print('player A is {} and player B is {}'.format(name_pa, name_pb))
    print('optimize both players')
    [aiming_grid_pa, prob_grid_normalscore_pa, prob_grid_singlescore_pa, prob_grid_doublescore_pa, prob_grid_triplescore_pa, prob_grid_bullscore_pa] = function_get_aiming_grid.load_aiming_grid(name_pa, data_parameter_dir=data_parameter_dir, grid_version=grid_version_pa)
    [aiming_grid_pb, prob_grid_normalscore_pb, prob_grid_singlescore_pb, prob_grid_doublescore_pb, prob_grid_triplescore_pb, prob_grid_bullscore_pb] = function_get_aiming_grid.load_aiming_grid(name_pb, data_parameter_dir=data_parameter_dir, grid_version=grid_version_pb)
    
    ## use single player game as the fixed policy    
    dp_policy_dict_pa = None
    dp_policy_dict_pb = None
    if dp_policy_folder is not None:
        dp_policy_filename_pa = dp_policy_folder + '/singlegame_{}_turn_{}.pkl'.format(name_pa, grid_version_pa)
        if (os.path.isfile(dp_policy_filename_pa) == True):
            dp_policy_dict_pa = ft.load_pickle(dp_policy_filename_pa)
            print('load player A policy {}'.format(dp_policy_filename_pa))
        dp_policy_filename_pb = dp_policy_folder + '/singlegame_{}_turn_{}.pkl'.format(name_pb, grid_version_pb)
        if (os.path.isfile(dp_policy_filename_pb) == True):
            dp_policy_dict_pb = ft.load_pickle(dp_policy_filename_pb)
            print('load player B policy {}'.format(dp_policy_filename_pb))    
    if dp_policy_dict_pa is None:
        print('solve player A NS policy')
        dp_policy_dict_pa = function_solve_dp.solve_dp_turn(aiming_grid_pa, prob_grid_normalscore_pa, prob_grid_doublescore_pa, prob_grid_bullscore_pa)
    if dp_policy_dict_pb is None:
        print('solve player B NS policy')
        dp_policy_dict_pb = function_solve_dp.solve_dp_turn(aiming_grid_pb, prob_grid_normalscore_pb, prob_grid_doublescore_pb, prob_grid_bullscore_pb)
        
    #### data for player A ####
    num_aiming_location_pa = aiming_grid_pa.shape[0]
    prob_normalscore_pa = prob_grid_normalscore_pa
    prob_doublescore_dic_pa = {}
    for doublescore_index in range(20):
        doublescore = 2*(doublescore_index+1)
        prob_doublescore_dic_pa[doublescore] = np.array(prob_grid_doublescore_pa[:,doublescore_index])
    prob_DB_pa = np.array(prob_grid_bullscore_pa[:,1])
    
    ## the probability of not bust for each action given score_max=i (score_remain=i+2)
    prob_bust_dic_pa = {}
    prob_notbust_dic_pa = {}
    for score_max in range(60):    
        ## transit to next throw or turn
        prob_notbust_pa = prob_grid_normalscore_pa[:,0:score_max+1].sum(axis=1)
        ## transit to the end of game
        score_remain = score_max + 2
        if (score_remain == fb.score_DB):
            prob_notbust_pa += prob_DB_pa
        elif (score_remain <= 40 and score_remain%2==0):
            prob_notbust_pa += prob_doublescore_dic_pa[score_remain]
        ##
        prob_notbust_pa = np.minimum(np.maximum(prob_notbust_pa, 0),1)
        prob_notbust_dic_pa[score_max] = prob_notbust_pa
        prob_bust_dic_pa[score_max] = 1 - prob_notbust_dic_pa[score_max]
    
    param_pa = {}    
    if (gpu_device is None):
        function_policy_improvement = zsg_policy_improvement
        prob_normalscore_tensor_pa = torch.from_numpy(prob_normalscore_pa)
        param_pa['prob_normalscore_tensor'] = prob_normalscore_tensor_pa
        param_pa['prob_doublescore_dic'] = prob_doublescore_dic_pa
        param_pa['prob_DB'] = prob_DB_pa
        param_pa['prob_bust_dic'] = prob_bust_dic_pa
        
    else:
        print('using gpu_device={}'.format(gpu_device))
        function_policy_improvement = zsg_policy_improvement_gpu
        prob_normalscore_tensor_pa = torch.from_numpy(prob_normalscore_pa).cuda(gpu_device)
        prob_bust_array_pa = np.zeros((num_aiming_location_pa, 60))
        prob_finish_array_pa = np.zeros((num_aiming_location_pa, 60))        
        for score_max in range(60):
            prob_bust_array_pa[:,score_max] = prob_bust_dic_pa[score_max]
            ## transit to the end of game
            score_remain = score_max + 2
            if (score_remain == fb.score_DB):
                prob_finish_array_pa[:,score_max] += prob_DB_pa
            elif (score_remain <= 40 and score_remain%2==0):
                prob_finish_array_pa[:,score_max] += prob_doublescore_dic_pa[score_remain]            
        prob_bust_tensor_pa = torch.from_numpy(prob_bust_array_pa).cuda(gpu_device)
        prob_finish_tensor_pa = torch.from_numpy(prob_finish_array_pa).cuda(gpu_device)
        param_pa['prob_normalscore_tensor'] = prob_normalscore_tensor_pa
        param_pa['prob_bust_tensor'] = prob_bust_tensor_pa
        param_pa['prob_finish_tensor'] = prob_finish_tensor_pa
        param_pa['gpu_device'] = gpu_device
    
    #### data for player B ####
    num_aiming_location_pb = aiming_grid_pb.shape[0]
    prob_normalscore_pb = prob_grid_normalscore_pb
    prob_doublescore_dic_pb = {}
    for doublescore_index in range(20):
        doublescore = 2*(doublescore_index+1)
        prob_doublescore_dic_pb[doublescore] = np.array(prob_grid_doublescore_pb[:,doublescore_index])
    prob_DB_pb = np.array(prob_grid_bullscore_pb[:,1])
    
    ## the probability of not bust for each action given score_max=i (score_remain=i+2)
    prob_bust_dic_pb = {}
    prob_notbust_dic_pb = {}
    for score_max in range(60):    
        ## transit to next throw or turn
        prob_notbust_pb = prob_grid_normalscore_pb[:,0:score_max+1].sum(axis=1)
        ## transit to the end of game
        score_remain = score_max + 2
        if (score_remain == fb.score_DB):
            prob_notbust_pb += prob_DB_pb
        elif (score_remain <= 40 and score_remain%2==0):
            prob_notbust_pb += prob_doublescore_dic_pb[score_remain]
        ##
        prob_notbust_pb = np.minimum(np.maximum(prob_notbust_pb, 0),1)
        prob_notbust_dic_pb[score_max] = prob_notbust_pb
        prob_bust_dic_pb[score_max] = 1 - prob_notbust_dic_pb[score_max]
    
    param_pb = {}    
    if (gpu_device is None):
        function_policy_improvement = zsg_policy_improvement
        prob_normalscore_tensor_pb = torch.from_numpy(prob_normalscore_pb)
        param_pb['prob_normalscore_tensor'] = prob_normalscore_tensor_pb
        param_pb['prob_doublescore_dic'] = prob_doublescore_dic_pb
        param_pb['prob_DB'] = prob_DB_pb
        param_pb['prob_bust_dic'] = prob_bust_dic_pb
        
    else:
        #print('using gpu_device={}'.format(gpu_device))
        function_policy_improvement = zsg_policy_improvement_gpu
        prob_normalscore_tensor_pb = torch.from_numpy(prob_normalscore_pb).cuda(gpu_device)
        prob_bust_array_pb = np.zeros((num_aiming_location_pb, 60))
        prob_finish_array_pb = np.zeros((num_aiming_location_pb, 60))        
        for score_max in range(60):
            prob_bust_array_pb[:,score_max] = prob_bust_dic_pb[score_max]
            ## transit to the end of game
            score_remain = score_max + 2
            if (score_remain == fb.score_DB):
                prob_finish_array_pb[:,score_max] += prob_DB_pb
            elif (score_remain <= 40 and score_remain%2==0):
                prob_finish_array_pb[:,score_max] += prob_doublescore_dic_pb[score_remain]            
        prob_bust_tensor_pb = torch.from_numpy(prob_bust_array_pb).cuda(gpu_device)
        prob_finish_tensor_pb = torch.from_numpy(prob_finish_array_pb).cuda(gpu_device)
        param_pb['prob_normalscore_tensor'] = prob_normalscore_tensor_pb
        param_pb['prob_bust_tensor'] = prob_bust_tensor_pb
        param_pb['prob_finish_tensor'] = prob_finish_tensor_pb
        param_pb['gpu_device'] = gpu_device
        
    #### 
    iteration_round_limit_zsgtwoplayers = 5
    iteration_relerror_limit_zsgtwoplayers = 10**-9
    iteration_round_zsgtwoplayers = np.zeros((502,502), dtype=np.int8)
    
    iteration_round_limit_singleplayer_policy = 20
    iteration_relerror_limit_singleplayer_policy = 10**-9
    
    value_pa = np.zeros((502,502))  # player A's winning probability when A throws at state [score_A, score_B]
    value_pb = np.zeros((502,502))  # player A's winning probability when B throws at state [score_A, score_B]
    value_win_pa = 1.0
    value_win_pb = 0.0
    num_iteration_record_pa = np.zeros((502,502), dtype=np.int8)
    num_iteration_record_pb = np.zeros((502,502), dtype=np.int8)
    ## values when optimizing A
    value_pa_optA = value_pa.copy()
    value_pb_optA = value_pb.copy()
    ## values when optimizing B
    value_pa_optB = value_pa.copy()
    value_pb_optB = value_pb.copy()    
    
    state_len_vector_pa = np.zeros(4, dtype=np.int32)
    state_value_default  = [None]  
    action_diff_pa  = [None]
    value_relerror_pa = np.zeros(4)
    for rt in [1,2,3]:
        ## for rt=3: possible score_gained = 0
        ## for rt=2: possible score_gained = 0,1,...,60
        ## for rt=1: possible score_gained = 0,1,...,120
        this_throw_state_len = fb.maxhitscore*(3-rt) + 1
        state_value_default.append(np.ones(this_throw_state_len)*fb.largenumber)
        action_diff_pa.append(np.ones(this_throw_state_len))    
    state_len_vector_pb = np.zeros(4, dtype=np.int32)
    action_diff_pb = ft.copy_numberarray_container(action_diff_pa)
    value_relerror_pb = np.zeros(4)
    
    optimal_value_dic_pa = {}
    optimal_action_index_dic_pa = {}
    prob_turn_transit_dic_pa = {}
    optimal_value_dic_pb = {} 
    optimal_action_index_dic_pb = {}
    prob_turn_transit_dic_pb = {}
    for score in range(2,502):
        optimal_value_dic_pa[score] = {}
        optimal_action_index_dic_pa[score] = {}
        prob_turn_transit_dic_pa[score] = {}
        optimal_value_dic_pb[score] = {}
        optimal_action_index_dic_pb[score] = {}
        prob_turn_transit_dic_pb[score] = {}
    
    #### algorithm start ####
    t_policy_improvement = 0
    t_policy_evaluation = 0
    t_other = 0
    t1 = time.time()
    for score_state_pb in range(2, game_begin_score_502):
        t_scoreloop_begin = time.time()
        score_state_list = []
        ## fix player B score, loop through player A
        for score_state_pa in range(2, game_begin_score_502):
            score_state_list.append([score_state_pa, score_state_pb])
    
        ########     solve all states in turn [score_A, score_B]    ########
        for [score_state_pa, score_state_pb] in score_state_list:
            #print('##### score_state [score_pa, score_pb] = {} ####'.format([score_state_pa, score_state_pb]))
    
            ## initialize the starting policy:
            ## player A
            for rt in [1,2,3]:        
                this_throw_state_len_pa = min(score_state_pa-2, fb.maxhitscore*(3-rt)) + 1
                state_len_vector_pa[rt] = this_throw_state_len_pa
            state_value_pa = ft.copy_numberarray_container(state_value_default)
            if score_state_pb > 2:
                state_action_pa = ft.copy_numberarray_container(optimal_action_index_dic_pa[score_state_pa][score_state_pb-1])
                prob_turn_transit_pa = prob_turn_transit_dic_pa[score_state_pa][score_state_pb-1]
            else:
                state_action_pa = ft.copy_numberarray_container(dp_policy_dict_pa['optimal_action_index_dic'][score_state_pa])
                prob_turn_transit_pa = dp_policy_dict_pa['prob_scorestate_transit'][score_state_pa]
            state_value_update_pa = ft.copy_numberarray_container(state_value_pa)
            state_action_update_pa = ft.copy_numberarray_container(state_action_pa)
    
            ## player B
            for rt in [1,2,3]:        
                this_throw_state_len_pb = min(score_state_pb-2, fb.maxhitscore*(3-rt)) + 1
                state_len_vector_pb[rt] = this_throw_state_len_pb
            state_value_pb = ft.copy_numberarray_container(state_value_default)
            if score_state_pa > 2:
                state_action_pb = ft.copy_numberarray_container(optimal_action_index_dic_pb[score_state_pa-1][score_state_pb])
                prob_turn_transit_pb = prob_turn_transit_dic_pb[score_state_pa-1][score_state_pb]
            else:
                state_action_pb = ft.copy_numberarray_container(dp_policy_dict_pb['optimal_action_index_dic'][score_state_pb])
                prob_turn_transit_pb = dp_policy_dict_pb['prob_scorestate_transit'][score_state_pb]
            state_value_update_pb = ft.copy_numberarray_container(state_value_pb)
            state_action_update_pb = ft.copy_numberarray_container(state_action_pb)            
            
            ## assemble variables
            ## player A
            param_pa['score_state'] = score_state_pa    
            param_pa['state_len_vector'] = state_len_vector_pa        
            param_pa['state_action'] = state_action_pa
            param_pa['state_value'] = state_value_pa
            param_pa['state_action_update'] = state_action_update_pa
            param_pa['state_value_update'] = state_value_update_pa
            param_pa['action_diff'] = action_diff_pa
            param_pa['value_relerror'] = value_relerror_pa    
            ## maximize player A's win_prob
            param_pa['flag_max'] = True
            param_pa['next_turn_value'] = value_pb[:,score_state_pb] ## player B throws in next turn
            param_pa['game_end_value'] = value_win_pa ## end game state A win
            
            ## player B
            param_pb['score_state'] = score_state_pb    
            param_pb['state_len_vector'] = state_len_vector_pb
            param_pb['state_action'] = state_action_pb
            param_pb['state_value'] = state_value_pb
            param_pb['state_action_update'] = state_action_update_pb
            param_pb['state_value_update'] = state_value_update_pb
            param_pb['action_diff'] = action_diff_pb
            param_pb['value_relerror'] = value_relerror_pb    
            ## maximize player B's win_prob = minimize player A's win_prob
            param_pb['flag_max'] = False
            param_pb['next_turn_value'] = value_pa[score_state_pa,:] ## player A throws in next turn
            param_pb['game_end_value'] = value_win_pb ## end game state B win     
            
            ## optimize A and B iteratively
            for round_index_zsgtwoplayers in range(iteration_round_limit_zsgtwoplayers):
                ## print('## optimize two players round = {} ##'.format(round_index_zsgtwoplayers))
                ## iterate at least once for each player
                
                #### optimize A policy ####
                value_pa_state_old = value_pa[score_state_pa,score_state_pb] ## starting value 0
                value_pb_state_old = value_pb[score_state_pa,score_state_pb] ## starting value 0                
                for round_index in range(iteration_round_limit_singleplayer_policy):                    
                    
                    ## policy evaluation
                    tpe1 = time.time()                
                    ## use the initial prob_turn_transit_pa value for round_index=0
                    if (round_index >0):
                        prob_turn_transit_pa = fep.solve_turn_transit_probability_fast(score_state_pa, state_action_pa, prob_grid_normalscore_pa, prob_grid_doublescore_pa, prob_grid_bullscore_pa, prob_bust_dic_pa)
                    ## player B is fixed, use stored value
                    [value_state_pa, value_state_pb] = zsg_policy_evaluation(value_pa, value_pb, score_state_pa, score_state_pb, prob_turn_transit_pa, prob_turn_transit_pb)
                    value_pa[score_state_pa, score_state_pb] = value_state_pa
                    value_pb[score_state_pa, score_state_pb] = value_state_pb
                    tpe2 = time.time()
                    t_policy_evaluation += (tpe2-tpe1)
    
                    #### policy improvement for player A ####
                    tpi1 = time.time()
                    param_pa['round_index'] = round_index
                    [max_action_diff_pa, max_value_relerror_pa] = function_policy_improvement(param_pa)
                    tpi2 = time.time()
                    t_policy_improvement += (tpi2 - tpi1)
                    if (max_action_diff_pa < 1):
                        break    
                    if (max_value_relerror_pa < iteration_relerror_limit_singleplayer_policy):
                        break
        
                optimal_action_index_dic_pa[score_state_pa][score_state_pb] = state_action_pa
                optimal_value_dic_pa[score_state_pa][score_state_pb] = state_value_pa
                prob_turn_transit_dic_pa[score_state_pa][score_state_pb] = prob_turn_transit_pa
                num_iteration_record_pa[score_state_pa, score_state_pb] = round_index + 1
                #### done optimize player A
                
                ## check optimality
                value_pa_optA[score_state_pa,score_state_pb] = value_pa[score_state_pa,score_state_pb]
                value_pb_optA[score_state_pa,score_state_pb] = value_pb[score_state_pa,score_state_pb]
                max_zsgvalue_relerror = max([np.abs(value_pa_state_old-value_pa[score_state_pa,score_state_pb]), np.abs(value_pb_state_old-value_pb[score_state_pa,score_state_pb])])
                #print('A:max_zsgvalue_relerror={}'.format(max_zsgvalue_relerror))      
                if (max_zsgvalue_relerror < iteration_relerror_limit_zsgtwoplayers):
                    break
    
    
                #### optimize B policy ####
                value_pa_state_old = value_pa[score_state_pa,score_state_pb] ## starting value 0
                value_pb_state_old = value_pb[score_state_pa,score_state_pb] ## starting value 0                
                for round_index in range(iteration_round_limit_singleplayer_policy):                    
                    
                    ## policy evaluation
                    tpe1 = time.time()
                    ## player A is fixed, only need to compute once
                    if (round_index > 0):
                        prob_turn_transit_pb = fep.solve_turn_transit_probability_fast(score_state_pb, state_action_pb, prob_grid_normalscore_pb, prob_grid_doublescore_pb, prob_grid_bullscore_pb, prob_bust_dic_pb)
                    [value_state_pa, value_state_pb] = zsg_policy_evaluation(value_pa, value_pb, score_state_pa, score_state_pb, prob_turn_transit_pa, prob_turn_transit_pb)
                    value_pa[score_state_pa, score_state_pb] = value_state_pa
                    value_pb[score_state_pa, score_state_pb] = value_state_pb
                    tpe2 = time.time()
                    t_policy_evaluation += (tpe2-tpe1)
    
                    #### policy improvement for player B ####
                    tpi1 = time.time()
                    param_pb['round_index'] = round_index
                    [max_action_diff_pb, max_value_relerror_pb] = function_policy_improvement(param_pb)
                    tpi2 = time.time()
                    t_policy_improvement += (tpi2 - tpi1)
                    if (max_action_diff_pb < 1):
                        break    
                    if (max_value_relerror_pb < iteration_relerror_limit_singleplayer_policy):
                        break
        
                optimal_action_index_dic_pb[score_state_pa][score_state_pb] = state_action_pb
                optimal_value_dic_pb[score_state_pa][score_state_pb] = state_value_pb
                prob_turn_transit_dic_pb[score_state_pa][score_state_pb] = prob_turn_transit_pb
                num_iteration_record_pb[score_state_pa, score_state_pb] = round_index + 1
                #### done optimize player B
        
                ## check optimality
                value_pa_optB[score_state_pa,score_state_pb] = value_pa[score_state_pa,score_state_pb]
                value_pb_optB[score_state_pa,score_state_pb] = value_pb[score_state_pa,score_state_pb]
                max_zsgvalue_relerror = max([np.abs(value_pa_state_old-value_pa[score_state_pa,score_state_pb]), np.abs(value_pb_state_old-value_pb[score_state_pa,score_state_pb])])
                #print('B:max_zsgvalue_relerror={}'.format(max_zsgvalue_relerror))
                if (max_zsgvalue_relerror < iteration_relerror_limit_zsgtwoplayers):
                    break
            
            #### done optimize A and B iteratively
            value_pa[score_state_pa,score_state_pb] = 0.5*(value_pa_optA[score_state_pa,score_state_pb]+value_pa_optB[score_state_pa,score_state_pb])
            value_pb[score_state_pa,score_state_pb] = 0.5*(value_pb_optA[score_state_pa,score_state_pb]+value_pb_optB[score_state_pa,score_state_pb])
            iteration_round_zsgtwoplayers[score_state_pa,score_state_pb] = round_index_zsgtwoplayers + 1
            #print('optimize A and B iteratively in time={} seconds'.format(time.time()-t_opt_twoplayers_begin))
        
        #### finish a column        
        #if (score_state_pb%20==0 or score_state_pb==2):
        #    print('#### score_state_pb={}, time={}'.format(score_state_pb, time.time()-t_scoreloop_begin))
    
    ## computation is done
    t2 = time.time()
    print('solve_zsg_opt_{}_fix_{} in {} seconds'.format(name_pa, name_pb, t2-t1))
    print('t_policy_evaluation  = {} seconds'.format(t_policy_evaluation))
    print('t_policy_improvement = {} seconds'.format(t_policy_improvement))
    print('t_other = {} seconds'.format(t_other))    
    #print('value_pa {} '.format(value_pa))
    #print('value_pb {} '.format(value_pb))
        
    
    result_dic = {'info':info, 'optimal_action_index_dic_pa':optimal_action_index_dic_pa, 'optimal_action_index_dic_pb':optimal_action_index_dic_pb, 'value_pa':value_pa, 'value_pb':value_pb, 'value_pa_optA':value_pa_optA, 'value_pa_optB':value_pa_optB, 'value_pb_optA':value_pb_optA, 'value_pb_optB':value_pb_optB,  'optimal_value_dic_pa':optimal_value_dic_pa, 'optimal_value_dic_pb':optimal_value_dic_pb, 'iteration_round_zsgtwoplayers':iteration_round_zsgtwoplayers, 'num_iteration_record_pa':num_iteration_record_pa, 'num_iteration_record_pb':num_iteration_record_pb}
    if (result_dir is not None):
        ft.dump_pickle(result_filename, result_dic)
        print('save {}'.format(result_filename))
        ft.dump_pickle(result_value_filename, {'info':info, 'value_pa':value_pa, 'value_pb':value_pb, 'iteration_round_zsgtwoplayers':iteration_round_zsgtwoplayers})
        print('save {}'.format(result_value_filename))
        return 'save'
    else:
        return result_dic
