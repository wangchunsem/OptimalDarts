1: Computation environment:
Python 3.7.4 + numpy 1.16.5 + pytorch 1.7.1

2: Code Files
2.1 All_Model_fits.mat: Provide the fitted skill model (the covariance matrix of the conditional Gaussian Model (2) in the paper) for each players. 
2.2 function_tool.py: Provide some file operation functions, i.e., reading and saving.
2.3 function_board.py: Provide the layout of the dart board and the 1mm grid of aiming locations. 
2.4 evaluate_score_probability_player.py: Conduct a numerical integration to evaluate the hitting probability of each score (each score segment in the dartboard and the corresponding the numerical score) associated with each aiming location on the 1mm-grid.
2.5 function_get_aiming_grid.py: Provide functions to generate action set. We used a small action set of 984 aiming points for the numerical experiment in the early version of the paper. We used an action set of 90,785 aiming points (all points in the 1mm grid over the dartboard) for the numerical experiment in the current version of the paper. 
2.6 function_solve_dp.py: Solve the single player dart game, i.e., the Non-Strategic DP described in Section 4.1 of the paper. 
2.7 function_evaluate_policy.py: Solve the state transition (turn to turn) probability associated with a specified aiming policy in the single player game. 
2.8 function_solve_zsg_gpu.py: Solve the ZSG of the dart game described in Section 4.2 of the paper. Algorithm 1 and Algorithm 3 are implemented.

3: The data structures and the basic logic of the algorithms are provided in the comments in the code files.
An example to solve the ZSG game between Player 1 Anderson and 2 Aspinall is in the runexample.py. 

