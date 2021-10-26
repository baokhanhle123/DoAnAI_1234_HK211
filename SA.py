import copy
import math
import random
import numpy as np
import heapq
import time
import tkinter as tk

decreaseFactor = 0.99

'''
def randominput(n):
    test = []
    for x in range(n * n):  # tao chuoi tu 0 den n^2-1
        test.append(x)

    random.shuffle(test)

    return np.array(test).reshape(n, n)  # chuyen thanh ma tran 5*5
'''


def proper_shuffle(state, n):
    for i in range(n):
        m = random.choice(all_moves(state))
        # print all_moves(state), m
        # print_puzzle(state)
        do_move(state, m)


def Custominput(test, n):
    return np.array(test).reshape(n, n)


def PosOfPlayer(gameState):
    return tuple(np.argwhere((gameState == 0))[0])


def CheckLegalAction(action, PosPlayer, gameState):
    n = len(gameState[0])
    xPlayer, yPlayer = PosPlayer
    x1, y1 = xPlayer + action[0], yPlayer + action[1]
    if (((x1 >= 0) & (x1 < n)) & ((y1 >= 0) & (y1 < n))):
        return 1
    else:
        return 0


def LegalAction(posPlayer, gameState):
    allActions = [[-1, 0, 'u'], [1, 0, 'd'], [0, -1, 'l'], [0, 1, 'r']]
    legalActions = []
    for action in allActions:

        if CheckLegalAction(action, posPlayer, gameState):
            legalActions.append(action)
        else:
            continue
    return tuple(tuple(x) for x in legalActions)


def UpdateState(PosPlayer, gameState, action):
    # print(gameState, PosPlayer)
    xPlayer, yPlayer = PosPlayer
    x1, y1 = xPlayer + action[0], yPlayer + action[1]

    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]]
    newPosPlayer = tuple(newPosPlayer)

    n = len(gameState[0])

    newgameState = np.zeros((n, n), dtype=int)
    for x in range(n):
        for y in range(n):
            newgameState[x][y] = gameState[x][y]
    newgameState[xPlayer][yPlayer] = gameState[x1][y1]
    newgameState[x1][y1] = 0
    # print(gameState)
    return newPosPlayer, newgameState


def Endstate(gameState):
    n = len(gameState[0])
    endstate = []
    for x in range(1, n * n):  # tao chuoi tu 0 den n^2-1
        endstate.append(x)
    endstate.append(0)
    return np.array(endstate)


def CheckEnd(gameState, b):
    n = len(gameState[0])
    a = gameState.reshape(1, n * n)[0]
    return (a == b).all()


def check(node, array):
    n = len(array)
    for x in range(n):
        if (node == array[x]).all():
            return 0
    return 1


def index(m, n):
    if m == 0:
        return n - 1, n - 1
    x = (m - 1) // n
    y = (m - 1) % n
    return x, y


def heuristic(gameState):
    n = len(gameState[0])
    sum = 0
    for x1 in range(n):
        for y1 in range(n):
            x2, y2 = index(gameState[x1][y1], n)
            if (x1 == x2) & (y1 == y2):
                sum = sum + 1
    return n * n - sum


def heuristic2(gameState):
    n = len(gameState[0])
    sum = 0
    for x1 in range(n):
        for y1 in range(n):
            x2, y2 = index(gameState[x1][y1], n)
            sum += abs(x1 - x2) + abs(y1 - y2)
    return sum


def heuristic3(gameState):
    return heuristic(gameState) + heuristic2(gameState)


"""hien thi"""


def UpdateShowState(gameState, action):
    # print(gameState, PosPlayer)
    n = len(gameState[0])
    for x in range(n):
        for y in range(n):
            if gameState[x][y] == 0:
                xPlayer = x
                yPlayer = y

    if action == 'u':
        x1, y1 = xPlayer - 1, yPlayer + 0
    elif action == 'r':
        x1, y1 = xPlayer + 0, yPlayer + 1
    elif action == 'd':
        x1, y1 = xPlayer + 1, yPlayer + 0
    elif action == 'l':
        x1, y1 = xPlayer + 0, yPlayer - 1

    gameState[xPlayer][yPlayer] = gameState[x1][y1]
    gameState[x1][y1] = 0
    # print(gameState)


def indexshow(m, n):
    x = m // n
    y = m % n
    return x, y


top = tk.Tk()
top.title('xep hinhf')


def show(gameState):
    n = len(gameState[0])
    b = gameState.reshape(n * n)
    showgamestate = []
    for x in range(n * n):
        if b[x] != 0:
            value = b[x]
        else:
            value = ""
        a0 = tk.Button(text=value, font=("Helvetica", 20,), height=3, width=7)
        showgamestate.append(a0)
    for x in range(n * n):
        x1, y1 = indexshow(x, n)
        showgamestate[x].grid(row=x1, column=y1)


def nextState(gameState):
    n = len(gameState[0])
    posPlayer = PosOfPlayer(gameState)
    # StartState = (posPlayer, gameState)

    legalActions = LegalAction(posPlayer, gameState)
    randomAction = legalActions[random.randrange(len(legalActions))]
    action = randomAction[2];
    copyGameState = copy.deepcopy(gameState)
    UpdateShowState(copyGameState, action)
    return copyGameState


def solved(state):
    return heuristic(state) == 0

# The energy of a state
# energy(solution) = 0.
def energy(gameState):
    #energy = heuristic(gameState)
    #energy = heuristic2(gameState)
    energy = heuristic3(gameState)
    return energy


def all_moves(gameState):
    n = len(gameState[0])
    posPlayer = PosOfPlayer(gameState)
    # StartState = (posPlayer, gameState)

    legalActions = LegalAction(posPlayer, gameState)
    moves = []
    for action in legalActions:
        moves.append(action[2])
    return moves


def do_move(gameState, move):
    UpdateShowState(gameState, move)


def undo_move(gameState, move):
    if move == 'u':
        do_move(gameState, 'd')
    if move == 'd':
        do_move(gameState, 'u')
    if move == 'l':
        do_move(gameState, 'r')
    if move == 'r':
        do_move(gameState, 'l')


def sim_annealing(initial_state, max_moves, p=1):
    print("Starting from:")
    print(initial_state)
    # Note that I'm only renaming the initial state:
    state = initial_state
    temperature = max_moves
    oldE = energy(state)
    while not solved(state) and temperature > 0:
        moves = all_moves(state)
        accepted = False
        while not accepted:
            # Normally we'd generate a random number between 0 and the
            # size of the array all_moves -1 and then make m =
            # all_moves of that index. But the random module already
            # has a function choice:
            m = random.choice(moves)
            do_move(state, m)
            newE = energy(state)
            deltaE = newE - oldE
            if deltaE <= 0:
                accepted = True
            else:
                boltz = math.exp(-float(p * deltaE) / temperature)
                # A random float between 0 and 1:
                r = np.random.uniform(1, 0, 1)
                if r <= boltz:
                    accepted = True
            if not accepted:
                undo_move(state, m)
        oldE = newE
        temperature = temperature - 1
        print("Moving", m)
        print(state)


def sim_annealing2(initial_state, max_moves, p=1):
    print("Starting from:")
    print(initial_state)
    # Note that I'm only renaming the initial state:
    state = initial_state
    temperature = max_moves
    oldE = energy(state)
    while not solved(state) and temperature > 0:
        moves = all_moves(state)
        accepted = False
        while not accepted:
            # Normally we'd generate a random number between 0 and the
            # size of the array all_moves -1 and then make m =
            # all_moves of that index. But the random module already
            # has a function choice:
            m = random.choice(moves)
            do_move(state, m)
            newE = energy(state)
            deltaE = newE - oldE
            if deltaE <= 0:
                accepted = True
            else:
                boltz = math.exp(-float(p * deltaE) / temperature)
                # A random float between 0 and 1:
                r = np.random.uniform(1, 0, 1)
                if r <= boltz:
                    accepted = True
            if not accepted:
                undo_move(state, m)
        oldE = newE
        temperature = temperature - 1
        print("Moving", m)
        print(state)


def solvePuzzle(gameState, calculateE):
    solutionFound = False
    N = 10000
    T = 2.2
    oldGameState = gameState
    # sigma = CalculateInitialSigma()
    for i in range(0, N):
        # check

        # calculate
        Eold = calculateE(oldGameState)
        newGameState = nextState(oldGameState)
        Enew = calculateE(newGameState)
        deltaE = Enew - Eold

        probability = math.exp(deltaE / T)
        # accept next state?
        if deltaE < 0:
            oldGameState = newGameState
        elif probability > np.random.uniform(1, 0, 1):
            oldGameState = newGameState

        # decrease temperature
        if T >= 0.01:
            T *= decreaseFactor

        # print gameState
        print(oldGameState)
        # time.sleep(1)


# test1 = [1, 2, 3, 4, 5, 6, 7, 10, 12, 0, 8, 15, 14, 11, 9, 13]
# gameState1=Custominput(test1,4)


#test2 = [4, 6, 5, 7, 2, 0, 1, 3, 8]
test2 = [1, 2, 3, 4, 5, 6, 7, 8, 0]
gameState1 = Custominput(test2, 3)
proper_shuffle(gameState1, 40)
print(gameState1)
print(heuristic(gameState1))
# print(PosOfPlayer(gameState1))
# print(energy(gameState1))
show(gameState1)
sim_annealing2(gameState1, 100, 0.01)
# solvePuzzle(gameState1, energy)
top.mainloop()
