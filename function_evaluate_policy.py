import os
import sys
import numpy as np
import time

import function_board as fb
import function_tool as ft

#%%
state_infeasible = fb.state_infeasible

def solve_turn_transit_probability(score_state, state_action, prob_normalscore, prob_doublescore, prob_bullscore):
    """
    Solve the state transition probability after a turn playing with a specified aiming policy
    
    Args: 
        score_state: score at the beginning of the turn, e.g., 2,3,...,501 
        state_action: a dict of aiming locations (actions in the policy) for each state (s,i,u) in this turn
        prob_normalscore, prob_doublescore, prob_bullscore: the skill model 
    
    Returns: A dict
        result_dict['finish']: probability of finishing the game (reach zero by making a double)
        result_dict['bust']: probability of busting the game (transit to next turn of (s=score_state,i=3,u=0))
        result_dict['score']: probability of achieving a cumulative score_gained in this turn (transit to next turn of (s=score_state-score_gained,i=3,u=0))
    """    
    
    ##
    result_dict = {}
    prob_finish = 0 ## probability of finishing the game
    prob_bust = 0   ## probability of busting the game
    ## initialize for (s, rt=3, score_gained=0)
    next_throw_state_len = 1
    prob_transit_next_throw_state = np.ones(next_throw_state_len)
    
    for rt in [3,2,1]:
        prob_this_throw_state = prob_transit_next_throw_state
        this_throw_state_len = next_throw_state_len
        next_throw_state_len = min(score_state-2, fb.maxhitscore*(4-rt)) + 1
        prob_transit_next_throw_state = np.zeros(next_throw_state_len)  ## probability vector of total score_gained after this throw
        
        for score_gained in range(this_throw_state_len):
            ## skip infeasible state
            if not fb.state_feasible_array[rt, score_gained]:
                continue   

            ## aimming location of the policy at this state
            aiming_location_index = state_action[rt][score_gained]
            prob_this_state = prob_this_throw_state[score_gained]
            
            #largest possible normal socre to make in the next throw without busting
            score_remain = score_state - score_gained
            score_max = min(score_remain-2, 60)
            score_max_plus1 = score_max + 1
        
            ## transit to next throw or turn with normal scores
            prob_transit_next_throw_state[score_gained:score_gained+score_max_plus1] += prob_normalscore[aiming_location_index, 0:score_max_plus1]*prob_this_state
            ## game can not bust or end when score_max = 60, i.e.,  prob_notbust = 1
            if (score_max < 60):
                prob_notbust_this_state = prob_normalscore[aiming_location_index, 0:score_max+1].sum()
                ## transit to the end of game
                if (score_remain == fb.score_DB):
                    prob_finish += prob_bullscore[aiming_location_index, 1]*prob_this_state
                    prob_notbust_this_state += prob_bullscore[aiming_location_index, 1]
                elif (score_remain <= 40 and score_remain%2==0):
                    doublescore_index = (score_remain//2) - 1
                    prob_finish += prob_doublescore[aiming_location_index, doublescore_index]*prob_this_state
                    prob_notbust_this_state += prob_doublescore[aiming_location_index, doublescore_index]
                else:
                    pass

                ## transit to bust
                prob_bust += (max(1 - prob_notbust_this_state,0))*prob_this_state
            
    result_dict['finish'] = prob_finish
    result_dict['bust'] = prob_bust
    result_dict['score'] = prob_transit_next_throw_state

    return result_dict


def solve_turn_transit_probability_fast(score_state, state_action, prob_normalscore, prob_doublescore, prob_bullscore, prob_bust_dic):
    """
    A fast way of implementing solve_turn_transit_probability by using pre-stored prob_bust_dic
    """     
    
    result_dict = {}
    prob_finish = 0 ## probability of finishing the game
    prob_bust_total = 0   ## probability of busting the game
    ## initialize for (s, rt=3, score_gained=0)
    next_throw_state_len = 1
    prob_transit_next_throw_state = np.ones(next_throw_state_len)
    
    for rt in [3,2,1]:
        prob_this_throw_state = prob_transit_next_throw_state
        this_throw_state_len = next_throw_state_len
        next_throw_state_len = min(score_state-2, fb.maxhitscore*(4-rt)) + 1
        prob_transit_next_throw_state = np.zeros(next_throw_state_len)  ## probability vector of total score_gained after this throw
        
        prob_normalscore_transit = prob_normalscore[state_action[rt][0:this_throw_state_len]]*prob_this_throw_state.reshape((this_throw_state_len,1))
        
        for score_gained in range(this_throw_state_len):  # loop through score already gained
            ## skip infeasible state
            if not fb.state_feasible_array[rt, score_gained]:
                continue   

            ## aimming location of the policy at this state
            aiming_location_index = state_action[rt][score_gained]
            prob_this_state = prob_this_throw_state[score_gained]
            
            #largest possible normal socre to make in the next throw without busting
            score_remain = score_state - score_gained
            score_max = min(score_remain-2, 60)
            score_max_plus1 = score_max + 1
        
            ## transit to next throw or turn with normal scores            
            prob_transit_next_throw_state[score_gained:score_gained+score_max_plus1] += prob_normalscore_transit[score_gained, 0:score_max_plus1]
            ## game can not bust or end when score_max = 60, i.e.,  prob_notbust = 1
            if (score_max < 60):
                ## transit to the end of game
                if (score_remain == fb.score_DB):
                    prob_finish += prob_bullscore[aiming_location_index, 1]*prob_this_state
                elif (score_remain <= 40 and score_remain%2==0):
                    doublescore_index = (score_remain//2) - 1
                    prob_finish += prob_doublescore[aiming_location_index, doublescore_index]*prob_this_state
                else:
                    pass

                #transit to bust
                prob_bust_total += prob_bust_dic[score_max][aiming_location_index]*prob_this_state
            
    result_dict['finish'] = prob_finish
    result_dict['bust'] = prob_bust_total
    result_dict['score'] = prob_transit_next_throw_state

    return result_dict  


def solve_policy_transit_probability(policy_action_index_dic, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore):
    """
    For each turn, solve the state transition probability for a specified aiming policy
    
    Args: 
        policy_action_index_dic: a dict of aiming locations (actions in the policy) for each state (s,i,u) of each turn s=2,...,501
        prob_normalscore, prob_doublescore, prob_bullscore: the skill model 
    
    Returns: A dict
    """  
    
    prob_policy_transit_dict = {}
    t1 = time.time()
    for score_state in range(2,502):
        prob_policy_transit_dict[score_state] = solve_turn_transit_probability(score_state, policy_action_index_dic[score_state], prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)

    t2 = time.time()
    print('solve prob_policy_transit in {} seconds'.format(t2-t1))
    
    return prob_policy_transit_dict
