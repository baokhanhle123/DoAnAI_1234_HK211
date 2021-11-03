import random
import numpy as np
import time
import tkinter as tk
def randominput(n):
    test=[]
    for x in range(n*n):            # tao chuoi tu 0 den n^2-1
        test.append(x)

    random.shuffle(test)    

    return np.array(test).reshape(n,n)   # chuyen thanh ma tran 5*5



def make_goal_state(state):
    test=[]
    n=len(state[0])
    for x in range(n*n - 1):           
        test.append(x + 1)
    test.append(0)

    return np.array(test).reshape(n,n)

def neighbor_state(state):
    queue = []
    n = len(state[0])
    for i in range(n):
        for j in range(n):
            if(state[i][j] == 0):
                if(i - 1 >= 0):
                    n1_state = np.array(state)
                    s = state[i - 1][j]
                    n1_state[i - 1][j] = 0
                    n1_state[i][j] = s
                    queue.append(n1_state)
                if(i + 1 < n):
                    n1_state = np.array(state)
                    s = state[i + 1][j]
                    n1_state[i + 1][j] = 0
                    n1_state[i][j] = s
                    queue.append(n1_state)
                if(j - 1 >= 0):
                    n1_state = np.array(state)
                    s = state[i][j - 1]
                    n1_state[i][j - 1] = 0
                    n1_state[i][j] = s
                    queue.append(n1_state)
                if(j + 1 < n):
                    n1_state = np.array(state)
                    s = state[i][j + 1]
                    n1_state[i][j + 1] = 0
                    n1_state[i][j] = s
                    queue.append(n1_state)
                return queue

def index(m,n):
    if m==0:
        return n-1,n-1
    x=(m-1)//n
    y=(m-1)%n
    return x,y

def heuristic(gameState):
    n=len(gameState[0])
    sum=0
    for x1 in range(n):
        for y1 in range(n):
            x2 ,y2 = index(gameState[x1][y1],n)
            if (x1==x2) & (y1==y2):
                sum=sum+1
    return n*n-sum - 1

def heuristic2(gameState):
    n=len(gameState[0])
    sum=0
    for x1 in range(n):
        for y1 in range(n):
            if(gameState[x1][y1] != 0):
                x2 ,y2 = index(gameState[x1][y1],n)
                sum +=abs(x1-x2)+abs(y1-y2)
    return sum 
def Custominput(test,n):
    return np.array(test).reshape(n,n)

def hillclimbing(state):
    queue = []
    goal_state = make_goal_state(state)
    original_state = neighbor_state(state)
    #bool = []
    #for i in range(len(original_state)):
        #bool.append(0)
    state1 = np.array(state)
    queue.append(state1)
    while True:
        if (state == goal_state).all():
            break
        else:
            neighbor = neighbor_state(state)
            array = []
            for i in range (len(neighbor)):
                array.append(heuristic(neighbor[i]) + heuristic2(neighbor[i]))
            if (min(array) <= heuristic(state) + heuristic2(neighbor[i])):
                state = neighbor[array.index(min(array))]
                #bool[array.index(min(array))] = 1
                tmp = np.array(state)
                queue.append(tmp)
            else:
                break

    return queue

test2=[6,5,3,2,10,9,1,0,12,13,15,4,14,8,7,11]
state=Custominput(test2,4)
queue = hillclimbing(state)
def indexshow(m,n):
    x=m//n
    y=m%n
    return x,y

top = tk.Tk()
top.title('xep hinhf')

def show(gameState):
    n=len(gameState[0])
    b=gameState.reshape(n*n)
    showgamestate=[]
    for x in range(n*n):
        if b[x]!=0:
            value=b[x]
        else:
            value=""
        a0=tk.Button(text=value,font=("Helvetica",20,),height=3,width=7)
        showgamestate.append(a0)
    for x in range(n*n):
        x1, y1=indexshow(x,n)
        showgamestate[x].grid(row=x1, column=y1)

for i in range(len(queue)):
    show(queue[i])
    top.mainloop()
