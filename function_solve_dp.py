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
torch.set_printoptions(precision=4)
torch.set_printoptions(linewidth=300)
torch.set_printoptions(threshold=300)


#%%
## single player game without the turn feature
def solve_dp_noturn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore=None, prob_grid_bullscore=None, prob_grid_doublescore_dic=None):
    """
    Solve the single player game without the turn feature. Find the optimal policy to minimize the expected number of throws for reaching zero score. 
    Args: 
        the action set and the hitting probability associated with the skill model
    
    Returns: 
        optimal_value[score_state]: the expected number of throws for reaching zero from score_state=2,...,501.
        optimal_action_index[score_state]: the index of the aiming location used for score_state=2,...,501.
    """

    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore_1tosmax_dic = {}
    prob_normalscore_1tosmaxsum_dic = {}
    for score_max in range(0,61):
        score_max_plus1 = score_max + 1 
        prob_normalscore_1tosmax_dic[score_max] = np.array(prob_grid_normalscore[:,1:score_max_plus1])
        prob_normalscore_1tosmaxsum_dic[score_max] = prob_normalscore_1tosmax_dic[score_max].sum(axis=1)
    if prob_grid_doublescore_dic is None:
        prob_doublescore_dic = {}
        for doublescore_index in range(20):
            doublescore = 2*(doublescore_index+1)
            prob_doublescore_dic[doublescore] = np.array(prob_grid_doublescore[:,doublescore_index])
    else:
        prob_doublescore_dic = prob_grid_doublescore_dic
    prob_DB = np.array(prob_grid_bullscore[:,1])

    ## possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros(502)
    #optimal_value[1] = np.nan
    optimal_action_index = np.zeros(502, np.int32)
    optimal_action_index[0] = -1
    optimal_action_index[1] = -1
        
    for score_state in range(2,502):            
        ## use matrix operation to search all aiming locations
        
        ## transit to less score state    
        ## s1 = min(score_state-2, 60)
        ## p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s1]*v[score_state-s1]
        score_max = min(score_state-2, 60)
        score_max_plus1 = score_max + 1 
        ## transit to next state
        num_tothrow = 1.0 + prob_normalscore_1tosmax_dic[score_max].dot(optimal_value[score_state-1:score_state-score_max-1:-1])
        ## probability of transition to state other than s itself
        prob_otherstate = prob_normalscore_1tosmaxsum_dic[score_max]
        
        ## transit to the end of game
        if (score_state == fb.score_DB): ## hit double bull
            prob_otherstate += prob_DB
        elif (score_state <= 40 and score_state%2==0): ## hit double
            prob_otherstate += prob_doublescore_dic[score_state]
        else: ## game does not end
            pass
        
        ## expected number of throw for all aiming locations
        prob_otherstate = np.maximum(prob_otherstate, 0)
        num_tothrow = num_tothrow / prob_otherstate
                            
        ## searching
        optimal_value[score_state] = num_tothrow.min()
        optimal_action_index[score_state] = num_tothrow.argmin()

    return [optimal_value, optimal_action_index]


#%%
## single player game with the turn feature
def solve_singlegame(name_pa, data_parameter_dir=fb.data_parameter_dir, grid_version=fb.grid_version, result_dir=None, postfix='', gpu_device=None):
    """
    Solve the single player game with the turn feature. Find the optimal policy to minimize the expected number of turns for reaching zero score. 
    Args: 
        name_pa: player ID
        data_parameter_dir=fb.data_parameter_dir
        grid_version: the action set and the hitting probability associated with the skill model.
            use 'v2' for the small action set of 984 aiming locations.
            use 'circleboard'  for the action set of the 1mm gird on the entire dartboard, 90,785 aiming locations.         
        result_dir: folder to store the result 
        postfix='':
        gpu_device: None for CPU computation, otherwise use the gpu device ID defined in the system (default 0).
    Returns: 
        a dict or save it.
    """
    
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = function_get_aiming_grid.load_aiming_grid(name_pa, data_parameter_dir=data_parameter_dir, grid_version=grid_version)
    
    if gpu_device is None:
        print('runing solve_dp_turn')
        result_dic = solve_dp_turn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    else:
        print('runing gpusolve_dp_turn')
        result_dic = solve_dp_turn_gpu(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore, gpu_device=gpu_device)
    
    if (result_dir is not None):
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)        
        result_filename = result_dir + '/singlegame_{}_turn_{}{}.pkl'.format(name_pa, grid_version, postfix)
        ft.dump_pickle(result_filename, result_dic, printflag=True)
    else:
        return result_dic


def solve_dp_turn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore):
    """
    Solve the single player game with the turn feature.
    Args: 
        the action set and the hitting probability associated with the skill model
    
    Returns: 
        optimal values and the corresponding aiming locations for each state (s,i,u)
    """
    
    ## aiming_grid
    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore = prob_grid_normalscore
    prob_doublescore_dic = {}
    for doublescore_index in range(20):
        doublescore = 2*(doublescore_index+1)
        prob_doublescore_dic[doublescore] = np.array(prob_grid_doublescore[:,doublescore_index])
    prob_DB = np.array(prob_grid_bullscore[:,1])
    
    ## the probability of not bust for each action given score_max=i (score_remain=i+2)
    prob_bust_dic = {}
    prob_notbust_dic = {}
    for score_max in range(60):    
        ## transit to next throw or turn
        prob_notbust = prob_grid_normalscore[:,0:score_max+1].sum(axis=1)
        ## transit to the end of game
        score_remain = score_max + 2
        if (score_remain == fb.score_DB):
            prob_notbust += prob_DB
        elif (score_remain <= 40 and score_remain%2==0):
            prob_notbust += prob_doublescore_dic[score_remain]
        ##
        prob_notbust = np.minimum(np.maximum(prob_notbust, 0),1)
        prob_notbust_dic[score_max] = prob_notbust
        prob_bust_dic[score_max] = 1 - prob_notbust_dic[score_max]
    
    prob_normalscore_tensor = torch.from_numpy(prob_normalscore)
    
    iteration_round_limit = 20
    iteration_relerror_limit = 10**-9
    
    #### state space example of (SB=25 DB=50) ####
    ## rt: the number of remaining throws in a turn
    ## state_infeasible_rt2 = [23, 29, 31, 35, 37, 41, 43, 44, 46, 47, 49, 52, 53, 55, 56, 58, 59]
    ## state_infeasible_rt1 = [103, 106, 109, 112, 113, 115, 116, 118, 119]    
        
    optimal_value_rt3 = np.zeros(502) #vector: optimal value for the beginning state of each turn (rt=3)
    optimal_value_dic = {} ## first key: score=0,2,...,501, second key: remaining throws=3,2,1
    optimal_action_index_dic = {}
    num_iteration_record = np.zeros(502, dtype=np.int32)
    
    state_len_vector = np.zeros(4, dtype=np.int32)
    state_value  = [None]  ## optimal value (expected # of turns to finish the game) for each state in the current playing turn
    state_action = [None]  ## aimming locations for for each state in the current playing turn
    action_diff  = [None]
    value_relerror = np.zeros(4)
    for rt in [1,2,3]:
        ## for rt=3: possible score_gained = 0
        ## for rt=2: possible score_gained = 0,1,...,60
        ## for rt=1: possible score_gained = 0,1,...,120
        this_throw_state_len = fb.maxhitscore*(3-rt) + 1
        state_value.append(np.ones(this_throw_state_len)*fb.largenumber)
        state_action.append(np.ones(this_throw_state_len, np.int32)*fb.infeasible_marker)
        action_diff.append(np.ones(this_throw_state_len))
    state_value_update = ft.copy_numberarray_container(state_value)
    state_action_update = ft.copy_numberarray_container(state_action)
    
    ## use no_turn policy as the initial policy
    [noturn_optimal_value, noturn_optimal_action_index] = solve_dp_noturn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    
    t1 = time.time()
    for score_state in range(2, 502):
        #print('#### solve_dp_turn score_state={} ####'.format(score_state))    
        
        ## initialization 
        for rt in [1,2,3]:
            ## for rt=3: score_gained = 0
            ## for rt=2: score_gained = 0,1,...,min(s-2,60)
            ## for rt=1: score_gained = 0,1,...,min(s-2,120)
            this_throw_state_len = min(score_state-2, fb.maxhitscore*(3-rt)) + 1
            state_len_vector[rt] = this_throw_state_len
                    
            ## initialize the starting policy: 
            ## use no_turn action in (s, i, u=0)
            ## use turn action (s-1, i, u-1) in (s, i, u!=0) if (s-1, i, u-1) is feasible state
            state_action[rt][0] = noturn_optimal_action_index[score_state]            
            for score_gained in range(1,this_throw_state_len):                
                if fb.state_feasible_array[rt, score_gained]:  ## if True
                    if fb.state_feasible_array[rt, score_gained-1]:
                        state_action[rt][score_gained] = optimal_action_index_dic[score_state-1][rt][score_gained-1]
                    else:                        
                        state_action[rt][score_gained] = noturn_optimal_action_index[score_state-score_gained]
                else:
                    state_action[rt][score_gained] = fb.infeasible_marker
    
        ## policy iteration
        for round_index in range(iteration_round_limit):
    
            ## policy evaluation
            rt = 3
            score_gained = 0
            score_max_turn = min(score_state-2, 3*fb.maxhitscore)
            prob_turn_transit = fep.solve_turn_transit_probability_fast(score_state, state_action, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore, prob_bust_dic)        
            prob_turn_zeroscore = prob_turn_transit['bust'] + prob_turn_transit['score'][0]
            new_value_rt3 = (1 + np.dot(prob_turn_transit['score'][1:], optimal_value_rt3[score_state-1:score_state-score_max_turn-1:-1])) / (1-prob_turn_zeroscore)
            state_value_update[rt][score_gained] = new_value_rt3
            optimal_value_rt3[score_state] = new_value_rt3
            #print('evaluate rt3 value= {}'.format(new_value_rt3)
    
            ## policy improvement
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
                            next_state_value_array[:,score_gained] = optimal_value_rt3[score_remain:score_remain-score_max_plus1:-1]
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
                            next_state_value_array[:] = optimal_value_rt3[score_remain:score_remain-score_max_plus1:-1]
    
                    ## matrix product to compute all together
                    next_state_value_tensor = torch.from_numpy(next_state_value_array)
                    ## transit to next throw in the same turn when rt=3,2
                    if (rt > 1):                    
                        num_turns_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)
                    ## transit to next turn when rt=1
                    else:
                        num_turns_tensor = 1 + prob_normalscore_tensor.matmul(next_state_value_tensor)
    
                    ## searching
                    temp1 = num_turns_tensor.min(axis=0)                
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
                            next_state_value_array[0:score_max_plus1,score_gained_index] = optimal_value_rt3[score_remain:score_remain-score_max_plus1:-1]
                    
                    next_state_value_tensor = torch.from_numpy(next_state_value_array)
                    ## transit to next throw in the same turn when rt=3,2
                    if (rt > 1):                    
                        num_turns_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)
                    ## transit to next turn when rt=1
                    else:
                        num_turns_tensor = 1 + prob_normalscore_tensor.matmul(next_state_value_tensor)                                                               
    
                    ## consider bust/finishing for each bust state seperately 
                    num_turns_array = num_turns_tensor.numpy()                
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
                        if (rt > 1):
                            if (score_remain == fb.score_DB):                        
                                num_turns_array[:,score_gained_index] += prob_DB
                            elif (score_remain <= 40 and score_remain%2==0):
                                num_turns_array[:,score_gained_index] += prob_doublescore_dic[score_remain]
                            else:
                                pass
    
                        ## transit to bust
                        if (rt==3):
                            num_turns_array[:,score_gained_index] += prob_bust_dic[score_max]
                            ## solve an equation other than using the policy evaluation value (s,i=3,u=0)
                            num_turns_array[:,score_gained_index] = num_turns_array[:,score_gained_index] / prob_notbust_dic[score_max] 
                        elif (rt==2):
                            num_turns_array[:,score_gained_index] += prob_bust_dic[score_max]*(1+new_value_rt3)
                        else:
                            num_turns_array[:,score_gained_index] += prob_bust_dic[score_max]*(new_value_rt3)  ## 1 turn is already counted before
    
                    ## searching
                    temp1 = num_turns_tensor.min(axis=0)
                    state_action_update[rt][state_notbust_len:this_throw_state_len] = temp1.indices.numpy()
                    state_value_update[rt][state_notbust_len:this_throw_state_len] =  temp1.values.numpy()                
    
                #### finish rt=1,2,3. check improvement
                action_diff[rt][:] = np.abs(state_action_update[rt] - state_action[rt])                                
                value_relerror[rt] = np.abs((state_value_update[rt] - state_value[rt])/state_value_update[rt]).max()
                state_action[rt][:] = state_action_update[rt][:]
                state_value[rt][:] = state_value_update[rt][:]
    
            max_action_diff = max([action_diff[1].max(), action_diff[2].max(), action_diff[3].max()])
            max_value_relerror = value_relerror.max()            
            
            if (max_action_diff < 1):
            #if max_value_relerror < iteration_relerror_limit:
                num_iteration_record[score_state] = round_index + 1
                break
    
        for rt in [1,2,3]:
            state_value_update[rt][fb.state_infeasible[rt]] = fb.largenumber
            state_action_update[rt][fb.state_infeasible[rt]] = fb.infeasible_marker
        optimal_action_index_dic[score_state] = ft.copy_numberarray_container(state_action_update)
        optimal_value_dic[score_state] = ft.copy_numberarray_container(state_value_update, new_dtype=fb.result_float_dytpe)
        optimal_value_rt3[score_state] = state_value[3][0]
        ## done:V(s,i=3/2/1,u)
    
    ##
    prob_scorestate_transit = {}    
    prob_scorestate_transit =  fep.solve_policy_transit_probability(optimal_action_index_dic, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    t2 = time.time()
    print('solve dp_turn_policyiter in {} seconds'.format(t2-t1))

    print(optimal_value_rt3)
    result_dic = {'optimal_value_dic':optimal_value_dic, 'optimal_action_index_dic':optimal_action_index_dic, 'optimal_value_rt3':optimal_value_rt3, 'prob_scorestate_transit':prob_scorestate_transit}

    return result_dic


def solve_dp_turn_gpu(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore, gpu_device=0):
    """
    Solve the single player game with the turn feature using GPU computation.
    Args: 
        the action set and the hitting probability associated with the skill model
    
    Returns: 
        optimal values and the corresponding aiming locations for each state (s,i,u)
    """
    
    ## aiming_grid
    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore = prob_grid_normalscore
    prob_doublescore_dic = {}
    for doublescore_index in range(20):
        doublescore = 2*(doublescore_index+1)
        prob_doublescore_dic[doublescore] = np.array(prob_grid_doublescore[:,doublescore_index])
    prob_DB = np.array(prob_grid_bullscore[:,1])
    
    ## the probability of not bust for each action given score_max=i (score_remain=i+2)
    prob_bust_dic = {}
    prob_notbust_dic = {}
    prob_bust_array = np.zeros((num_aiming_location, 60))
    prob_notbust_array = np.zeros((num_aiming_location, 60))
    prob_finish_array = np.zeros((num_aiming_location, 60))
    for score_max in range(60):    
        ## transit to next throw or turn
        prob_notbust = prob_grid_normalscore[:,0:score_max+1].sum(axis=1)
        ## transit to the end of game
        score_remain = score_max + 2
        if (score_remain == fb.score_DB):
            prob_notbust += prob_DB
            prob_finish_array[:,score_max] += prob_DB
        elif (score_remain <= 40 and score_remain%2==0):
            prob_notbust += prob_doublescore_dic[score_remain]
            prob_finish_array[:,score_max] += prob_doublescore_dic[score_remain]
        
        prob_notbust = np.minimum(np.maximum(prob_notbust, 0),1) ##
        prob_notbust_dic[score_max] = prob_notbust #a new copy
        prob_bust_dic[score_max] = 1 - prob_notbust
        
        prob_notbust_array[:,score_max] = prob_notbust
        prob_bust_array[:,score_max] = 1 - prob_notbust
    print('using gpu_device={}'.format(gpu_device))
    prob_normalscore_tensor = torch.from_numpy(prob_normalscore).cuda(gpu_device)
    prob_bust_tensor = torch.from_numpy(prob_bust_array).cuda(gpu_device)
    prob_notbust_tensor = torch.from_numpy(prob_notbust_array).cuda(gpu_device)
    prob_finish_tensor = torch.from_numpy(prob_finish_array).cuda(gpu_device)
   
    iteration_round_limit = 20
    iteration_relerror_limit = 10**-9
    
    ####################
    ## state space example of (SB=25 DB=50)
    ## rt: remaining number of throws in a turn
    ## state_infeasible_rt2 = [23, 29, 31, 35, 37, 41, 43, 44, 46, 47, 49, 52, 53, 55, 56, 58, 59]
    ## state_infeasible_rt1 = [103, 106, 109, 112, 113, 115, 116, 118, 119]    
        
    optimal_value_rt3 = np.zeros(502) #vector: optimal value for the beginning state of each turn (rt=3)
    optimal_value_dic = {} ## first key: score=0,2,...,501, second key: remaining throws=3,2,1
    optimal_action_index_dic = {}
    num_iteration_record = np.zeros(502, dtype=np.int32)
    
    state_len_vector = np.zeros(4, dtype=np.int32)
    state_value  = [None]  ## expected # of turns for each state in the turn
    state_action = [None]  ## aimming locations for for each state in the turn
    action_diff  = [None]
    value_relerror = np.zeros(4)
    for rt in [1,2,3]:
        ## for rt=3: possible score_gained = 0
        ## for rt=2: possible score_gained = 0,1,...,60
        ## for rt=1: possible score_gained = 0,1,...,120
        this_throw_state_len = fb.maxhitscore*(3-rt) + 1
        state_value.append(np.ones(this_throw_state_len)*fb.largenumber)
        state_action.append(np.ones(this_throw_state_len, np.int32)*fb.infeasible_marker)
        action_diff.append(np.ones(this_throw_state_len))
    state_value_update = ft.copy_numberarray_container(state_value)
    state_action_update = ft.copy_numberarray_container(state_action)
    
    ## using no_turn policy as the initial policy
    [noturn_optimal_value, noturn_optimal_action_index] = solve_dp_noturn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    
    t1 = time.time()
    for score_state in range(2, 502):
        #print('#### solve_dp_turn score_state={} ####'.format(score_state))    
        
        ## initialization 
        for rt in [1,2,3]:
            ## for rt=3: score_gained = 0
            ## for rt=2: score_gained = 0,1,...,min(s-2,60)
            ## for rt=1: score_gained = 0,1,...,min(s-2,120)
            this_throw_state_len = min(score_state-2, fb.maxhitscore*(3-rt)) + 1
            state_len_vector[rt] = this_throw_state_len
                    
            ## initialize the starting policy: 
            ## use no_turn action in (s, i, u=0)
            ## use turn action (s-1, i, u-1) in (s, i, u!=0) if (s-1, i, u-1) is feasible state
            state_action[rt][0] = noturn_optimal_action_index[score_state]            
            for score_gained in range(1,this_throw_state_len):                
                if fb.state_feasible_array[rt, score_gained]:  ## if True
                    if fb.state_feasible_array[rt, score_gained-1]:
                        state_action[rt][score_gained] = optimal_action_index_dic[score_state-1][rt][score_gained-1]
                    else:                        
                        state_action[rt][score_gained] = noturn_optimal_action_index[score_state-score_gained]
                else:
                    state_action[rt][score_gained] = fb.infeasible_marker
    
        ## policy iteration
        for round_index in range(iteration_round_limit):
    
            ## policy evaluation
            rt = 3
            score_gained = 0
            score_max_turn = min(score_state-2, 3*fb.maxhitscore)
            prob_turn_transit = fep.solve_turn_transit_probability_fast(score_state, state_action, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore, prob_bust_dic)        
            prob_turn_zeroscore = prob_turn_transit['bust'] + prob_turn_transit['score'][0]
            new_value_rt3 = (1 + np.dot(prob_turn_transit['score'][1:], optimal_value_rt3[score_state-1:score_state-score_max_turn-1:-1])) / (1-prob_turn_zeroscore)
            state_value_update[rt][score_gained] = new_value_rt3
            optimal_value_rt3[score_state] = new_value_rt3
            #print('evaluate rt3 value= {}'.format(new_value_rt3)
    
            ## policy improvement
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
                            next_state_value_array[:,score_gained] = optimal_value_rt3[score_remain:score_remain-score_max_plus1:-1]
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
                            next_state_value_array[:] = optimal_value_rt3[score_remain:score_remain-score_max_plus1:-1]
    
                    ## matrix product to compute all together
                    next_state_value_tensor = torch.from_numpy(next_state_value_array).cuda(gpu_device)
                    ## transit to next throw in the same turn when rt=3,2
                    if (rt > 1):                    
                        num_turns_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)
                    ## transit to next turn when rt=1
                    else:
                        num_turns_tensor = 1 + prob_normalscore_tensor.matmul(next_state_value_tensor)
    
                    ## searching
                    temp1 = num_turns_tensor.min(axis=0)                
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
                            next_state_value_array[0:score_max_plus1,score_max] = optimal_value_rt3[score_remain:score_remain-score_max_plus1:-1]
    
                    next_state_value_tensor = torch.from_numpy(next_state_value_array).cuda(gpu_device)
                    ## transit to next throw in the same turn when rt=3,2
                    if (rt > 1):                    
                        num_turns_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)
                    ## transit to next turn when rt=1
                    else:
                        num_turns_tensor = 1 + prob_normalscore_tensor.matmul(next_state_value_tensor)
    
                    ## consider bust/finishing for all bust states together
                    if (rt == 1):
                        num_turns_tensor += prob_bust_tensor*new_value_rt3
                    elif (rt == 2):
                        num_turns_tensor += (prob_finish_tensor + prob_bust_tensor*(1+new_value_rt3))
                    else: ##(rt == 3):
                        num_turns_tensor += (prob_finish_tensor + prob_bust_tensor)
                        num_turns_tensor = num_turns_tensor/prob_notbust_tensor
                                            
                    ## searching
                    temp1 = num_turns_tensor.min(axis=0)
                    ## reverse the order to fill
                    state_action_update[rt][state_notbust_len:this_throw_state_len] = temp1.indices.cpu().numpy()[range(score_max_first_bust_state, score_max_last_bust_state-1, -1)]
                    state_value_update[rt][state_notbust_len:this_throw_state_len] =  temp1.values.cpu().numpy()[range(score_max_first_bust_state, score_max_last_bust_state-1, -1)]
    
                #### finish rt=1,2,3. check improvement
                action_diff[rt][:] = np.abs(state_action_update[rt] - state_action[rt])                                
                value_relerror[rt] = np.abs((state_value_update[rt] - state_value[rt])/state_value_update[rt]).max()
                state_action[rt][:] = state_action_update[rt][:]
                state_value[rt][:] = state_value_update[rt][:]
    
            max_action_diff = max([action_diff[1].max(), action_diff[2].max(), action_diff[3].max()])
            max_value_relerror = value_relerror.max()            
            
            if (max_action_diff < 1):
            #if max_value_relerror < iteration_relerror_limit:
                num_iteration_record[score_state] = round_index + 1
                break
    
        for rt in [1,2,3]:
            state_value_update[rt][fb.state_infeasible[rt]] = fb.largenumber
            state_action_update[rt][fb.state_infeasible[rt]] = fb.infeasible_marker
        optimal_action_index_dic[score_state] = ft.copy_numberarray_container(state_action_update)
        optimal_value_dic[score_state] = ft.copy_numberarray_container(state_value_update, new_dtype=fb.result_float_dytpe)
        optimal_value_rt3[score_state] = state_value[3][0]
        ## done:V(s,i=3/2/1,u)
    
    ##
    prob_scorestate_transit = {}    
    prob_scorestate_transit =  fep.solve_policy_transit_probability(optimal_action_index_dic, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    t2 = time.time()
    print('solve dp_turn_policyiter in {} seconds'.format(t2-t1))

    print(optimal_value_rt3)
    result_dic = {'optimal_value_dic':optimal_value_dic, 'optimal_action_index_dic':optimal_action_index_dic, 'optimal_value_rt3':optimal_value_rt3, 'prob_scorestate_transit':prob_scorestate_transit}

    return result_dic
