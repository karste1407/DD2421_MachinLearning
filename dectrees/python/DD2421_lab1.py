import monkdata as m
import numpy as np
import random
import math
from dtree import entropy,averageGain,buildTree,check

S1 = m.monk1
S2 = m.monk2
S3 = m.monk3

S1_test = m.monk1test
S2_test = m.monk2test
S3_test = m.monk3test

#ASSIGNMENT 1

E1 = entropy(S1)
E2 = entropy(S2)
E3 = entropy(S3)

print('Entropy: ')
print('-------------')
print('MONK1:   ',E1)
print('MONK2:   ',E2)
print('MONK3:   ',E3)

#ASSIGNMENT 2
#Coin toss with normal vs rigged coin (1 = Heads, 0 = tails)
print('')

r_arr = np.random.rand(100,1)
coin_toss = r_arr[r_arr > 0.5]
rigged_toss = r_arr[r_arr > 0.1]

nHeads = np.array([sum(coin_toss)/len(coin_toss), sum(rigged_toss)/len(rigged_toss)])
nTails = 1 - nHeads

E_ct = -nTails[0]*np.log2(nTails[0]) -nHeads[0]*np.log2(nHeads[0])
E_rt = -nTails[1]*np.log2(nTails[1]) -nHeads[1]*np.log2(nHeads[1])

print('Entropy: ')
print('-------------')
print('Coin Toss 50/50:   ',E_ct)
print('Coin Toss 90/10:   ',E_rt)

#ASSIGNMENT 3
print('')
avg_gains = np.zeros((len(m.attributes),3))
for idx,attr in enumerate(m.attributes):
    avg1 = averageGain(S1,attr)
    avg2 = averageGain(S2,attr)
    avg3 = averageGain(S3,attr)
    avg_gains[idx,:] = ([np.round(avg1,3),np.round(avg2,3),np.round(avg3,3)])

print('Dataset| a1,    a2,   a3,   a4,   a5,   a6')
print('MONK1: |', avg_gains[:,0])
print('MONK2: |', avg_gains[:,1])
print('MONK3: |', avg_gains[:,2])

#ASSIGNMENT 5
print('')

tree1 = buildTree(S1,m.attributes)
tree2 = buildTree(S2,m.attributes)
tree3 = buildTree(S3,m.attributes)

scores_training = [check(tree1,S1), check(tree2,S2), check(tree3,S3)]
scores_test = [check(tree1,S1_test), check(tree2,S2_test), check(tree3,S3_test)]

print('Accuracy Training Set,   Accuracy Test Set')
print('MONK1: ',str(scores_training[0]) + '             ' + str(scores_test[0]))
print('MONK2: ',str(scores_training[1]) + '             ' + str(scores_test[1]))
print('MONK3: ',str(scores_training[2]) + '             ' + str(scores_test[2]))


## ASSIGNMENT 7
print('')

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

split_rates = [0.3,0.4,0.5,0.6,0.7,0.8]
cross_validate = 10
acc = np.zeros((cross_validate,3))
acc_holder = np.zeros((len(split_rates),3))

for j,rate in enumerate(split_rates):
    for i in range(cross_validate):
        S1_train, S1_test = partition(S1, rate)
        S2_train, S2_test = partition(S2, rate)
        S3_train, S3_test = partition(S3, rate)

        t1 = buildTree(S1_train,m.attributes)
        t2 = buildTree(S2_train,m.attributes)
        t3 = buildTree(S3_train,m.attributes)
        score = [check(t1,S1_test), check(t2,S2_test), check(t3,S3_test)]
        acc[i,:] = score
    acc_cv = np.mean(acc,axis=0)
    acc_holder[j,:] = acc_cv

print(acc_holder)
