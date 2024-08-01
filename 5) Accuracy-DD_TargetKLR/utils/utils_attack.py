import math
import copy
import ot
from scipy.special import softmax

from .utils_op import *

""" Compute Attacker Blackbox/Whitebox Reward using Wasserstein Distance Rate """
def Attack_Cost_Compute_W(env, init_T, agent_Q, target, cost_matrix=0, distance_type=0, whitebox=1, sinkhorn=0):
    
    policy = Get_Policy(agent_Q, env)
    #agent_Q = np.expand_dims(agent_Q, axis=0)
    #target = DicQ_To_MatrixQ(target, env)
    
    target_map = np.repeat(np.sum(target, axis=1).reshape(1,16), env.nA, axis = 1)
    target_policy = (target+0.001)/(1+0.001*env.nA)
    no_of_agents = 1 #agent_Q.shape[0]
    #policy = agent_Q.reshape(agent_Q.shape[0]*env.nS, env.nA) #Flatten agent_Q from (no_of_agents, env.nS, env.nA) to (no_of_agents x env.nS, env.nA)
    #if(whitebox == 1):
        #policy = Get_Policy(policy, env)
    T =  env.T.copy()

    # compute P*
    init_T_matrix = np.transpose(init_T,(1,0,2)).reshape(env.nS*env.nA,env.nS) # s1,a1;s1,a2;s1,a3;s1,a4;s2,a1;
    init_T_matrix = np.repeat(init_T_matrix, env.nA, axis=1) #Dim 1 - 1st state, all actions; 2nd state, all actions ... #Dim 2 - 1st state, all actions; 2nd state, all actions ...
    init_T_matrix = np.tile(init_T_matrix, (no_of_agents,1))
    policy_matrix = np.repeat(policy.reshape(no_of_agents,env.nS*env.nA), env.nS*env.nA, axis=0) #np.tile(policy.reshape(no_of_agents,env.nS*env.nA), (env.nS*env.nA,1)) #policy.reshape(1,env.nS*env.nA)
    target_policy_matrix =  np.repeat(target_policy.reshape(1,env.nS*env.nA), no_of_agents*env.nS*env.nA, axis=0) #np.tile(target_policy.reshape(1,env.nS*env.nA), (env.nS*env.nA,1)) #target_policy.reshape(1,env.nS*env.nA)
    target_map = np.tile(target_map, (no_of_agents*env.nS*env.nA,1))
    P_star = target_map*(init_T_matrix*target_policy_matrix) + (1-target_map)*(init_T_matrix*policy_matrix)

    # compute P_u(s',a'|s,a) with updated policy
    T_matrix = np.transpose(T,(1,0,2)).reshape(env.nS*env.nA,env.nS)
    T_matrix = np.repeat(T_matrix, env.nA, axis=1)
    T_matrix = np.tile(T_matrix, (no_of_agents,1))
    if(distance_type == 0): #Complete
        P = T_matrix * policy_matrix
    elif(distance_type == 1): #Grid
        P = target_map*(T_matrix*target_policy_matrix) + (1-target_map)*(T_matrix*policy_matrix)
    elif(distance_type == 2): #Behavior
        P = init_T_matrix * policy_matrix

    initial_distribution = np.zeros((1,64)) #1x64
    initial_distribution[0,0],initial_distribution[0,1],initial_distribution[0,2],initial_distribution[0,3] = 0.25,0.25,0.25,0.25

    nth_step_prob_star = [1]*no_of_agents
    nth_step_prob = [1]*no_of_agents
    nth_step_path_star = [1]*no_of_agents
    nth_step_path = [1]*no_of_agents
    #Nth Step Probabilities
    for k in range(no_of_agents):
        nth_step_prob_star[k] = np.matmul(P_star[k*64:(k+1)*64],P_star[k*64:(k+1)*64])
        nth_step_prob[k] = np.matmul(P[k*64:(k+1)*64],P[k*64:(k+1)*64])
        for i in range(8):
            nth_step_prob_star[k] = np.matmul(nth_step_prob_star[k],P_star[k*64:(k+1)*64])
            nth_step_prob[k] = np.matmul(nth_step_prob[k],P[k*64:(k+1)*64])
        nth_step_path_star[k] = np.squeeze(np.matmul(initial_distribution, nth_step_prob_star[k]))
        nth_step_path[k] = np.squeeze(np.matmul(initial_distribution, nth_step_prob[k]))

    #Normalize Nth Step Probabilities
    nth_step_path_star = np.around(nth_step_path_star,10)
    nth_step_path = np.around(nth_step_path,10)
    nth_step_path_star = nth_step_path_star/np.expand_dims(np.sum(nth_step_path_star, axis=1), axis=1)
    nth_step_path = nth_step_path/np.expand_dims(np.sum(nth_step_path, axis=1), axis=1)

    #Wasserstein Cost Computation
    if(sinkhorn == 0):
        transport_matrix_WDR = [1]*no_of_agents
        cost = [1]*no_of_agents
        for k in range(no_of_agents):
            transport_matrix_WDR[k] = ot.emd(nth_step_path_star[k], nth_step_path[k], M=cost_matrix)
            cost[k] = np.sum(cost_matrix*transport_matrix_WDR[k])
    else:
        transport_matrix_WDR_sinkhorn = [1]*no_of_agents
        cost = [1]*no_of_agents
        for k in range(no_of_agents):
            transport_matrix_WDR_sinkhorn[k] = ot.bregman.sinkhorn(nth_step_path_star[k], nth_step_path[k], M=cost_matrix, reg=100.0)
            cost[k] = np.sum(cost_matrix*transport_matrix_WDR_sinkhorn[k])

    # Done Computation
    #deviation = Attack_Error(policy, np.tile(target, (no_of_agents,1)))
    #if (deviation == 0):
        #done = 1
    #else:
        #done = 0

    return np.mean(cost) #, done, deviation


""" Function to compute Attck_Cost_t based on KLR """
def Attack_Cost_Compute_K(env, init_T, agent_Q, target, cost_matrix=0, distance_type=0, whitebox=1, sinkhorn=0):
    
    policy = Get_Policy(agent_Q, env)
    #agent_Q = np.expand_dims(agent_Q, axis=0)
    #target = DicQ_To_MatrixQ(target, env)
    
    target_map = np.repeat(np.sum(target, axis=1).reshape(1,16), env.nA, axis = 1)
    target_policy = (target+0.001)/(1+0.001*env.nA)
    no_of_agents = 1 #agent_Q.shape[0]
    #policy = agent_Q.reshape(agent_Q.shape[0]*env.nS, env.nA) #Flatten agent_Q from (no_of_agents, env.nS, env.nA) to (no_of_agents x env.nS, env.nA)
    #if(whitebox == 1):
        #policy = Get_Policy(policy, env)
    T =  env.T.copy()

    # compute P*
    init_T_matrix = np.transpose(init_T,(1,0,2)).reshape(env.nS*env.nA,env.nS)
    init_T_matrix = np.repeat(init_T_matrix, env.nA, axis=1)
    init_T_matrix = np.tile(init_T_matrix, (no_of_agents,1))
    policy_matrix = np.repeat(policy.reshape(no_of_agents,env.nS*env.nA), env.nS*env.nA, axis=0) #np.tile(policy.reshape(no_of_agents,env.nS*env.nA), (env.nS*env.nA,1)) #policy.reshape(1,env.nS*env.nA)
    target_policy_matrix =  np.repeat(target_policy.reshape(1,env.nS*env.nA), no_of_agents*env.nS*env.nA, axis=0) #np.tile(target_policy.reshape(1,env.nS*env.nA), (env.nS*env.nA,1)) #target_policy.reshape(1,env.nS*env.nA)
    target_map = np.tile(target_map, (no_of_agents*env.nS*env.nA,1))
    P_star = target_map*(init_T_matrix*target_policy_matrix) + (1-target_map)*(init_T_matrix*policy_matrix)

    # compute P_u(s',a'|s,a) with updated policy
    T_matrix = np.transpose(T,(1,0,2)).reshape(env.nS*env.nA,env.nS)
    T_matrix = np.repeat(T_matrix, env.nA, axis=1)
    T_matrix = np.tile(T_matrix, (no_of_agents,1))        
    if(distance_type == 0): #Complete
        P = T_matrix * policy_matrix
    elif(distance_type == 1): #Grid
        P = target_map*(T_matrix*target_policy_matrix) + (1-target_map)*(T_matrix*policy_matrix)
    elif(distance_type == 2): #Behavior
        P = init_T_matrix * policy_matrix    

    # COST
    DKL_matrix = np.sum(P*np.log(P/P_star), axis=1).reshape(no_of_agents*env.nS, env.nA) # compute D_t^KL
    sxa = env.nS*env.nA

    w_matrix, v_matrix = np.linalg.eig(np.transpose(P.reshape(no_of_agents,sxa,sxa), (0,2,1))) # stationary distribution
    j_stationary_matrix = np.argmin(abs(w_matrix - 1.0), axis=1).astype(int)
    q_stationary_matrix = np.transpose(v_matrix, (0,2,1))[np.arange(no_of_agents),j_stationary_matrix].real
    q_stationary_matrix /= np.sum(q_stationary_matrix, axis=1).reshape(no_of_agents,1)
    cost = np.sum(q_stationary_matrix.reshape(no_of_agents*env.nS, env.nA) * DKL_matrix)

    # DONE
    #deviation = Attack_Error(policy, np.tile(target, (no_of_agents,1)))
    #if (deviation == 0):
        #done = 1
    #else:
        #done = 0

    return cost/no_of_agents #, done, deviation

"""def Attack_Cost_Compute_K(env, init_T, agent_Q, target_policy, cost_matrix=0):
    
    #target_policy = DicQ_To_MatrixQ(target, env)
    
    for i in range(len(target_policy)):
        target_policy[i] = (target_policy[i]+0.001)/(1+0.001*env.nA)

    policy = Get_Policy(agent_Q, env)
    T =  env.T.copy()

    # compute P*
    P_star = np.zeros((env.nS*env.nA, env.nS*env.nA))

    for cur_s in range(env.nS):
        for cur_a in range(env.nA):
            row_index = cur_s*env.nA + cur_a
            for next_s in range(env.nS):
                if next_s in target:
                    for next_a in range(env.nA):
                        col_index = next_s*env.nA + next_a
                        P_star[row_index][col_index] = init_T[cur_a][cur_s][next_s]*target_policy[next_s][next_a]
                else:
                    for next_a in range(env.nA):
                        col_index = next_s*env.nA + next_a
                        P_star[row_index][col_index] = init_T[cur_a][cur_s][next_s]*policy[next_s][next_a]


    # compute P_u(s',a'|s,a) with updated policy
    P = np.zeros((env.nS*env.nA, env.nS*env.nA))
    for cur_s in range(env.nS):
        for cur_a in range(env.nA):
            row_index = cur_s*env.nA + cur_a
            for next_s in range(env.nS):
                for next_a in range(env.nA):
                    col_index = next_s*env.nA + next_a
                    P[row_index][col_index] = T[cur_a][cur_s][next_s]*policy[next_s][next_a]


    # compute D_t^KL
    DKL = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in range(env.nA):
            row_index = s*env.nA + a
            for col_index in range(env.nS*env.nA):
                DKL[s][a] += P[row_index][col_index]*math.log(P[row_index][col_index]/P_star[row_index][col_index])
    # stationary distribution          
    w, v = np.linalg.eig(P.T)
    j_stationary = np.argmin(abs(w - 1.0))
    q_stationary = v[:,j_stationary].real
    q_stationary /= q_stationary.sum()

    # compute cost
    cost = 0
    for s in range(env.nS):
        for a in range(env.nA):
            index = s*env.nA + a
            cost += q_stationary[index] * DKL[s][a]

    return cost"""



"""
Function to measure whether attack done
"""
def Attack_Done_Identify(env, target, Q):
    
    #target = DicQ_To_MatrixQ(target, env)
    target_map = np.sum(target, axis=1)
    total = np.sum(target_map)
    index_target = np.argmax(target, axis=1)
    accuracy_elementwise = target_map*(index_target == np.argmax(Q, axis=1))
    sum_accuracy_elementwise = np.sum(accuracy_elementwise)
    accuracy = sum_accuracy_elementwise/total

    Q_softmax = softmax(Q, axis=1)
    accuracy_softmax_complete_elementwise = target_map*( Q_softmax[np.arange(16),index_target] )
    accuracy_softmax_complete = np.sum( accuracy_softmax_complete_elementwise )/total
    
    accuracy_softmax_elementwise = accuracy_elementwise * accuracy_softmax_complete_elementwise
    accuracy_softmax = np.where( (sum_accuracy_elementwise!=0), np.sum(accuracy_softmax_elementwise)/sum_accuracy_elementwise, 0.0).item()
    
    if accuracy == 1.0:
        done = 1
    else:
        done = 0
    
    return done, accuracy, accuracy_softmax, accuracy_softmax_complete

def Attack_Effort(curA, env):
    prevA = curA.copy() #New
    curA_updated = env.altitude.copy().reshape((16, 1))
    action_clipped = curA_updated - prevA #Size: 16x1
    effort = np.mean(np.abs(action_clipped))
    
    return effort, curA_updated



def Attack_Done_Identify_Old(env, target, Q):
    error = 0
    amount = len(target)

    for s in range(env.nS):
        if s in target:
            target_a_index = np.argmax(target[s])
            learner_a_index = np.argmax(Q[s])

            if target_a_index != learner_a_index:
                error += 1

    if error == 0:
        done = 1
    else:
        done = 0

    accuracy_rate = (amount-error)/amount

    return done, accuracy_rate




if __name__ == "__main__":
    
    # Env
    from envs.env3D_4x4 import GridWorld_3D_env
    env = GridWorld_3D_env()
    
    # Victim
    from victim.victim_Q import VictimAgent_Q
    victim_args = {
        "env": env, 
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    victim = VictimAgent_Q(**victim_args)
    victim.train(2)
    
    """ target policy """
    target = defaultdict(lambda: np.zeros(env.action_space.n))

    target[0] = np.array([0, 0, 1, 0])
    target[4] = np.array([0, 0, 1, 0])
    target[8] = np.array([0, 0, 1, 0])
    target[12] = np.array([0, 1, 0, 0])
    target[13] = np.array([0, 1, 0, 0])

    print(f"target policy: {target}")
    
    ''' Evaluation '''
    cost = Attack_Cost_Compute(env, victim.init_T, victim.Q, target)
    print(cost)
    
    done, rate = Attack_Done_Identify(env, target, victim.Q)
    print(done, rate)