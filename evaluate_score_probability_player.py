import os
import sys
import numpy as np
import time

import scipy.io as sio
from scipy.stats import multivariate_normal

import function_board as fb
import function_tool as ft

#%%
def evaluate_score_probability(playerID_list):    
    """
    Players' fitted skill models are contained in ALL_Model_Fits.mat.
    A dart throw landing follows a bivariate Gaussian distribution with the mean(center) as the aiming location and the covariance matrix given in the fitted model. 
    This function conducts a numerical integration to evaluate the hitting probability of each score (each score segment in the dartboard) associated with each aiming location on the 1mm-grid.
    
    Args: 
        A list of playerID to evaluate, e.g., [1,2] for player1 (Anderson) and player2 (Aspinall).
    
    Returns: 
        A dict of four numpy arrays: prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore.
        prob_grid_singlescore[xi,yi,si] is of size 341*341*20.  (prob_grid_doublescore and prob_grid_triplescore have the similar structure.)
        xi and yi are the x-axis and y-axis indexes (starting from 0) of the square 1mm grid enclosing the circle dartboard.
        si = 0,1,...,19 for score S1,S2,...,S20
        prob_grid_bullscore[xi,yi,si] is of size 341*341*2, where si=0 represents SB and si=1 represents DB 
        For example, when aiming at the center of the dartboard,
        prob_grid_singlescore[xi=170,yi=170,si=9] is the probability of hitting S10 
        prob_grid_doublescore[xi=170,yi=170,si=0] is the probability of hitting D1 
        prob_grid_triplescore[xi=170,yi=170,si=7] is the probability of hitting T8 
        prob_grid_bullscore[xi=170,yi=170,si=0] is the probability of hitting SB
        prob_grid_bullscore[xi=170,yi=170,si=1] is the probability of hitting DB
        
        Results are stored in the folder ./data_parameter/player_gaussin_fit/grid_full
                
    """

    result_dir = fb.data_parameter_dir + '/grid_full'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    player_parameter = sio.loadmat('./ALL_Model_Fits.mat')
    
    ## 1mm-width grid of 341*341 aiming locations (a sqaure enclosing the circle dart board)
    [xindex, yindex, xgrid, ygrid, grid_num] = fb.get_1mm_grid()
    
    ## 0.5mm-width grid of 681*681 locations for evaluating the PDF of the fitted Gaussian distribution
    f_density_grid_pixel_per_mm = 2
    f_density_grid_pixel_per_mm = 0.1
    f_density_grid_num = int(2*fb.R*f_density_grid_pixel_per_mm) + 1
    f_density_grid_width = 1.0/f_density_grid_pixel_per_mm
    f_density_constant = f_density_grid_width*f_density_grid_width
    print('f_density_grid_num={} f_density_grid_width={}'.format(f_density_grid_num, f_density_grid_width))
    
    ## f_density_grid x coordinate left to right increasing
    f_density_xindex = range(f_density_grid_num)
    f_density_xgrid =  np.arange(f_density_grid_num) * f_density_grid_width - fb.R
    ## y coordinate top to bottom increasing
    f_density_yindex = f_density_xindex[:]
    f_density_ygrid = f_density_xgrid[:]
    
    # build f_density_grid, x is the horizon axis (column index) and y is the vertical axis (row index). Hence, y is at first
    y, x = np.mgrid[-fb.R:fb.R+0.1*f_density_grid_width:f_density_grid_width, -fb.R:fb.R+0.1*f_density_grid_width:f_density_grid_width]
    pos = np.dstack((x, y))
    
    ## score information on the f_density_grid
    singlescore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    doublescore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    triplescore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    bullscore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    for xi in f_density_xindex:
        for yi in f_density_yindex:
            singlescore_grid[yi,xi] = fb.get_score_singleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            doublescore_grid[yi,xi] = fb.get_score_doubleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            triplescore_grid[yi,xi] = fb.get_score_tripleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            bullscore_grid[yi,xi] = fb.get_score_bullonly(f_density_xgrid[xi], f_density_ygrid[yi])
    singlescore_coordinate_dic = {}
    doublescore_coordinate_dic = {}
    triplescore_coordinate_dic = {}
    bullscore_coordinate_dic = {}
    
    ## coordinate for each score
    for si in range(20):
        singlescore_coordinate_dic[si] = np.where(singlescore_grid==fb.singlescorelist[si])
        doublescore_coordinate_dic[si] = np.where(doublescore_grid==fb.doublescorelist[si])
        triplescore_coordinate_dic[si] = np.where(triplescore_grid==fb.triplescorelist[si])
    bullscore_coordinate_dic[0] = np.where(bullscore_grid==fb.bullscorelist[0])
    bullscore_coordinate_dic[1] = np.where(bullscore_grid==fb.bullscorelist[1])
    
    ## 
    for playerID in playerID_list:
        name_pa = 'player{}'.format(playerID)
        result_filename = result_dir + '/' + '{}_gaussin_prob_grid.pkl'.format(name_pa)
        print('\ncomputing {}'.format(result_filename))
        player_index = playerID - 1
        
        ## new result grid    
        prob_grid_singlescore = np.zeros((grid_num, grid_num, fb.singlescorelist_len))
        prob_grid_doublescore = np.zeros((grid_num, grid_num, fb.doublescorelist_len))
        prob_grid_triplescore = np.zeros((grid_num, grid_num, fb.triplescorelist_len))
        prob_grid_bullscore = np.zeros((grid_num, grid_num, fb.bullscorelist_len))
    
        #### conduct a numerical integration to evaluate the hitting probability for each score associated with the given aiming location
        time1 = time.time()
        for xi in xindex:
            ##print(xi)
            for yi in yindex:
                ## select the proper Gaussian distribution according to the area to which the aiming location belongs
                mu = [xgrid[xi], ygrid[yi]]
                score, multiplier = fb.get_score_and_multiplier(mu)
                if (score==60 and multiplier==3): ##triple 20
                    covariance_matrix = player_parameter['ModelFit_T20'][0, player_index][2]
                elif (score==57 and multiplier==3): ##triple 19
                    covariance_matrix = player_parameter['ModelFit_T19'][0, player_index][2]
                elif (score==54 and multiplier==3): ##triple 18
                    covariance_matrix = player_parameter['ModelFit_T18'][0, player_index][2]
                elif (score==51 and multiplier==3): ##triple 17
                    covariance_matrix = player_parameter['ModelFit_T17'][0, player_index][2]
                elif (score==50 and multiplier==2): ##double bull
                    covariance_matrix = player_parameter['ModelFit_B50'][0, player_index][2]
                else:
                    covariance_matrix = player_parameter['ModelFit_All_Doubles'][0, player_index][2]
                        
                ## f_density_grid is the PDF of the fitted Gaussian distribution
                rv = multivariate_normal(mu, covariance_matrix)
                f_density_grid = rv.pdf(pos)
            
                ## check score and integrate density
                for si in range(20):
                    prob_grid_singlescore[xi,yi,si] = f_density_grid[singlescore_coordinate_dic[si]].sum()*f_density_constant
                    prob_grid_doublescore[xi,yi,si] = f_density_grid[doublescore_coordinate_dic[si]].sum()*f_density_constant
                    prob_grid_triplescore[xi,yi,si] = f_density_grid[triplescore_coordinate_dic[si]].sum()*f_density_constant
                prob_grid_bullscore[xi,yi,0] = f_density_grid[bullscore_coordinate_dic[0]].sum()*f_density_constant
                prob_grid_bullscore[xi,yi,1] = f_density_grid[bullscore_coordinate_dic[1]].sum()*f_density_constant
                
        result_dic = {'prob_grid_singlescore':prob_grid_singlescore, 'prob_grid_doublescore':prob_grid_doublescore,'prob_grid_triplescore':prob_grid_triplescore, 'prob_grid_bullscore':prob_grid_bullscore}    
        time2 = time.time()
        print('computation is done in {} seconds'.format(time2-time1))    
        ft.dump_pickle(result_filename, result_dic, printflag=True)
    
    return
