from array import array
from time import sleep
import numpy as np 
import sys

N = 5 # Number of states
M = 27 # number of all possible observations

tmpA = np.random.uniform(size=[N,N])
# tmpA = np.eye(N)
A = np.zeros_like(tmpA)
for i in range(tmpA.shape[0]):
    for j in range(tmpA.shape[1]):
        A[i,j] = tmpA[i,j]/np.sum(tmpA[i,:])

# np.fill_diagonal(A,0)
tmpB = np.random.uniform(size=[N,M])
B = np.zeros_like(tmpB)
for i in range(tmpB.shape[0]):
    for j in range(tmpB.shape[1]):
        B[i,j] = tmpB[i,j]/np.sum(tmpB[i,:])

P = np.random.uniform(size=[N,])
P = np.array([p/np.sum(P) for p in P])

def get_index(values, distribution):
    if type(values)==list:
        values = np.array(values)
    if type(values)==float:
        values = np.array([values])
    cdf = [np.sum(distribution[0:i]) for i in range(1,distribution.shape[0]+1)]
    res = np.ones_like(values)
    for i,v in enumerate(values):
        tmp = cdf - v
        tmp[tmp<0] = 1
        res[i] = np.argmin(tmp)
    return res.astype(int)


def showNewRes(res):
    if res < 0 or res>26:
        print('*',end='')
    elif res == 26:
        print(' ',end='')
    else:
        print(chr(res+97),end='-')
    sys.stdout.flush()




def forward(O):
    T = O.shape[0]
    a = np.zeros([N,T])

    for j in range(N):
        a[j,0] = P[j]*B[j,O[0]]

    for t in range(1,T):
        for j in range(N):
            a[j,t] = 0
            for i in range(N):
                a[j,t] += a[i,t-1]*A[i,j]
            a[j,t] *= B[j,O[t]]
    
    res = 0
    for i in range(N):
        res += a[i,T-1]

    return res
    

states = []
observations = []

for i in range(1):
    if i==0:
        current_state = get_index(np.random.uniform(),P)[0]
    else:
        current_state = get_index(np.random.uniform(),A[current_state,:])[0]
    states.append(current_state)
    output = get_index(np.random.uniform(),B[current_state,:])[0]
    observations.append(output)
    showNewRes(output)
    sleep(0.1)
observations = np.array(observations)
print('')
print(states)
print(observations)
prob = forward(observations)
print(f'probability = {prob}')



