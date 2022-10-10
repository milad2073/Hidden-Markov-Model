from time import sleep
import numpy as np 
import sys

N = 5 # Number of states
M = 27 # number of all possible observations


### wrong way of creating A, because the distribution of each row is not normal. see the following link 
# https://stackoverflow.com/questions/8064629/random-numbers-that-add-to-100-matlab/8068956#8068956
# tmpA = np.random.uniform(size=[N,N])
# # tmpA = np.eye(N)
# A = np.zeros_like(tmpA)
# for i in range(tmpA.shape[0]):
#     for j in range(tmpA.shape[1]):
#         A[i,j] = tmpA[i,j]/np.sum(tmpA[i,:])
# A = np.round(A,3)


tmpA =np.sort(np.random.rand(N,N-1))
tmpA = np.insert(tmpA,0,np.zeros(N),1)
tmpA = np.append(tmpA,np.ones([N,1]),1)
A = np.diff(tmpA)
assert np.all(A.sum(axis=1) == np.ones(N)), 'sum of each row of A should be one'


tmpB =np.sort(np.random.rand(N,M-1))
tmpB = np.insert(tmpB,0,np.zeros(N),1)
tmpB = np.append(tmpB,np.ones([N,1]),1)
B = np.diff(tmpB)
assert np.all(B.sum(axis=1) == np.ones(N)), 'sum of each row of B should be one'


tmpP =np.sort(np.random.rand(1,N - 1))
tmpP = np.insert(tmpP,0,np.zeros(1),1)
tmpP = np.append(tmpP,np.ones([1,1]),1)
P = np.diff(tmpP).squeeze()
assert P.sum() == 1 , 'sum of P should be one'


def decodeOutput(val):
    if val < 0 or val>26:
        return '*'
    elif val == 26:
        return ' '
    else:
        return chr(val+97)

def encodeOutput(char):
    if len(char)!=1:
        raise 'invalid character'

    char = char.lower()
    if char == ' ':
        return 26

    val = ord(char) - 97
    if val < 0 or val>26:
        raise 'invalid character'
    
    return val
        
def calc_alpha(A_inp,B_inp,P_inp,O):
    T = O.shape[0]
    alpha = np.zeros([N,T])

    for j in range(N):
        alpha[j,0] = P_inp[j]*B_inp[j,O[0]]

    for t in range(1,T):
        for j in range(N):
            alpha[j,t] = 0
            for i in range(N):
                alpha[j,t] += alpha[i,t-1]*A_inp[i,j]
            alpha[j,t] *= B_inp[j,O[t]]

    return alpha

def calc_beta(A_inp,B_inp,O):
    T = O.shape[0]
    beta = np.zeros([N,T])

    beta[:,-1] = 1

    for t in range(T-2,-1,-1):
        for i in range(N):
            beta[i,t] = 0
            for j in range(N):
                beta[i,t] += beta[j,t+1]*A_inp[i,j]*B_inp[j,O[t+1]]

    return beta

# calculate the probability of O given A,B, and P
def forward(O):
    T = O.shape[0]
    alpha = calc_alpha(A,B,P,O)
    
    res = 0
    for i in range(N):
        res += alpha[i,T-1]

    # beta = calc_beta(A,B,P,O)
    # res2 = 0
    # for j in range(N):
    #     res2 += P[j]*B[j,O[0]]*beta[j,0]

    return res

# forward(np.array([1,2,3]))

# returns the most probable sequence of states (and its probability) given an observation O and A, B, and P 
def decode(O):
    T = O.shape[0]
    v =  np.zeros([N,T])
    bt = np.zeros([N,T])

    for j in range(N):
        v[j,0] = P[j]*B[j,O[0]]
    
    for t in range(1,T):
        for j in range(N):
            v[j,t] = 0
            for i in range(N):
                if v[i,t-1]*A[i,j] > v[j,t]:
                    v[j,t] = v[i,t-1]*A[i,j]
                    bt[j,t] = i
    
    maximum_p = 0
    ind_maximum_p = -1
    for i in range(N):
        if v[i,T-1] > maximum_p:
            maximum_p = v[i,T-1]
            ind_maximum_p = i
    
    sequence = np.zeros([T])
    
    sequence[T-1] = ind_maximum_p
    for t in range(T-2,-1,-1):
        sequence[t] = bt[int(sequence[t+1]),t+1]
    return maximum_p, sequence


# Baum-Welch algorithm
# learn A and B given these parameters:
# 1- initial values for A and B
# 2- P
# 3- an observation sequence O
def forward_backward(init_A, init_B,init_P, O):

    T = O.shape[0]
    A_hat = init_A.copy()
    B_hat = init_B.copy()
    P_hat = init_P.copy()

    converged = False
    iter = 0
    while not converged and iter<20:
        iter += 1
        print(f'iteration #{iter}. ')
        alpha = calc_alpha(A_hat,B_hat,P_hat,O)
        beta  = calc_beta(A_hat,B_hat,O)

        prob_o = 0
        for i in range(N):
            prob_o += alpha[i,T-1]
        

        gamma =  np.zeros([N,T])
        xi = np.zeros([N,N,T])

        for t in range(T):
            for j in range(N):
                gamma[j,t] = alpha[j,t]*beta[j,t]/prob_o
                for i in range(N):
                    if t < T-1:
                        xi[i,j,t] = alpha[i,t]*B_hat[j,O[t+1]]*beta[j,t+1]/prob_o
        

        A_hat = np.zeros_like(A_hat)
        B_hat = np.zeros_like(B_hat)

        for i in range(N):
            for j in range(N):
                sm = 0
                for k in range(N):
                    sm += xi[i,k,:].sum()
                A_hat[i,j] = xi[i,j,:].sum()/sm
        
        for j in range(N):
            sm = gamma[j,:].sum()
            for vk in range(M):
                tmp = 0
                for t in range(T):
                    if O[t]==vk:
                        tmp += gamma[j,t]
                B_hat[j,vk] = tmp/sm
                
    return A_hat,B_hat,P_hat




observations = [encodeOutput(c) for c in 'miladm']
observations = np.array(observations)

A, B, P = forward_backward(A, B, P, observations)
print(A)
print(B)
print(P)

assert np.all(A.sum(axis=1) == np.ones(N)), 'sum of each row of A should be one'
assert np.all(B.sum(axis=1) == np.ones(N)), 'sum of each row of B should be one'
assert P.sum() == 1 , 'sum of P should be one'


states = []
observations = []

for i in range(50):
    if i==0:
        current_state = np.random.choice(list(range(N)),p=P) 
    else:
        current_state = np.random.choice(list(range(N)),p=A[current_state,:]) 
    states.append(current_state)
    output = np.random.choice(list(range(M)),p=B[current_state,:]) 
    observations.append(output)
    char = decodeOutput(output)
    print(char ,end='')
    sys.stdout.flush()
    sleep(0.1)
observations = np.array(observations)
print('')
print(states)
print(observations)
prob = forward(observations)
print(f'probability = {prob}')
prob_max,path = decode(observations)
print(f'max prob = {prob_max}')
print(f'path = {path}')



