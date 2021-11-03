import random
import numpy as np
import heapq
import time
import tkinter as tk



class PriorityQueue:                    #hang doi uu tien hien thuc bang haepq
    def __init__(self):
        self._data = []
        self._index = 0
        self.count=0
    def push(self, item, priority):
        heapq.heappush(self._data, (priority, self._index, item))
        self._index += 1
        self.count +=1
    def pop(self):
        self.count -=1
        return heapq.heappop(self._data)[-1]

def Custominput(test,n):
    return np.array(test).reshape(n,n)

def PosOfPlayer(gameState):
    return tuple(np.argwhere((gameState == 0))[0])

def CheckLegalAction(action,PosPlayer,gameState):
    n =len(gameState[0])
    xPlayer, yPlayer = PosPlayer
    x1, y1 = xPlayer + action[0], yPlayer + action[1]
    if (((x1>=0) & (x1<n)) & ((y1>=0) & (y1<n))):
        return 1
    else:
        return 0

def LegalActon(posPlayer,gameState):
    allActions = [[-1,0,'u'],[1,0,'d'],[0,-1,'l'],[0,1,'r']]
    legalActions = []
    for action in allActions:
        
        if CheckLegalAction(action, posPlayer, gameState):
            legalActions.append(action)
        else: 
            continue
    return tuple(tuple(x) for x in legalActions) 

def UpdateState(PosPlayer, gameState, action):
    #print(gameState, PosPlayer)
    xPlayer, yPlayer = PosPlayer
    x1, y1 = xPlayer + action[0], yPlayer + action[1]

    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] 
    newPosPlayer = tuple(newPosPlayer)

    n = len(gameState[0])

    newgameState = np.zeros((n,n),dtype=int)
    for x in range(n):
        for y in range(n):
            newgameState[x][y]=gameState[x][y]
    newgameState[xPlayer][yPlayer] = gameState[x1][y1]
    newgameState[x1][y1] = 0
    #print(gameState)
    return newPosPlayer, newgameState


def Endstate(gameState):
    n=len(gameState[0])
    endstate=[]
    for x in range(1,n*n):            # tao chuoi tu 0 den n^2-1
        endstate.append(x)
    endstate.append(0)
    return np.array(endstate)

def CheckEnd(gameState,end):
    n=len(gameState[0])
    a = gameState.reshape(1,n*n)[0]
    return (a == end).all()

def check(node,array):
    n=len(array)
    for x in range(n):
        if (node==array[x]).all() :
            return 0
    return 1

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
            x2 ,y2 =index(gameState[x1][y1],n)
            if (x1==x2) & (y1==y2):
                if gameState[x1][y1]!=0:
                    sum=sum+1
    return n*n-sum

def astar(gameState):
    n=len(gameState[0])
    posPlayer=PosOfPlayer(gameState)
    StartState=(posPlayer,gameState)

    queueState = PriorityQueue()
    queueState.push([StartState],heuristic(gameState))

    actions = PriorityQueue()
    actions.push([0],heuristic(gameState))
    exploredSet = np.zeros((1,n,n),dtype=int)
    b=Endstate(gameState)
    count=0
    while   queueState:
        node =  queueState.pop()
        node_action = actions.pop()
        count+=1
        if CheckEnd(node[0][1],b):
            print(','.join(node_action[1:]).replace(',',''))
            print(len(node_action),count)
            break
        cost=len(node_action)
        if check(node[0][1],exploredSet):
            exploredSet=np.concatenate((exploredSet,[node[0][1]]))
            for action in LegalActon(node[0][0], node[0][1]):
                newPosPlayer, newState = UpdateState(node[0][0], node[0][1], action)
                heur=heuristic(newState)
                queueState.push([(newPosPlayer, newState)],heur*n*8/3+cost)
                actions.push(node_action + [action[-1]],heur*n*8/3+cost)
    return node_action

"""hien thi"""

def UpdateShowState(gameState, action):
    #print(gameState, PosPlayer)
    n = len(gameState[0])
    for x in range(n):
        for y in range(n):
            if gameState[x][y]==0:
                xPlayer=x 
                yPlayer=y

    if action=='u':
        x1, y1 = xPlayer - 1, yPlayer + 0
    elif action=='r':
        x1, y1 = xPlayer + 0, yPlayer + 1
    elif action=='d':
        x1, y1 = xPlayer + 1, yPlayer + 0
    elif action=='l':
        x1, y1 = xPlayer + 0, yPlayer - 1

    
    gameState[xPlayer][yPlayer] = gameState[x1][y1]
    gameState[x1][y1] = 0
    #print(gameState)


def indexshow(m,n):
    x=m//n
    y=m%n
    return x,y

top = tk.Tk()
top.title('xep hinh')



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
    

def task():
    if len(move)!=0:
        UpdateShowState(gameState1,move[0])
        del move[0]
        show(gameState1) 
        time.sleep(1)
        top.after(100,task)






test1=[1,6,2,14,7,10,4,3,5,0,13,8,9,15,11,12]
#gameState1=Custominput(test1,4)


test1=[2,4,8,3,6,1,7,5,0]
gameState1=Custominput(test1,3)


show(gameState1)
print(gameState1)
time_start = time.time()
move=astar(gameState1)
time_end=time.time()
print('Runtime ',time_end-time_start,' second.')

del move[0]
top.after(100,task)
top.mainloop()
