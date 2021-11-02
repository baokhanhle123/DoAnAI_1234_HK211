import copy
import math
import random
import numpy as np
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
    # energy = heuristic(gameState)
    # energy = heuristic2(gameState)
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


# local

def sim_annealing0(initial_state, max_moves, p=1):
    print("Starting from:")
    print(initial_state)
    # Note that I'm only renaming the initial state:
    state = initial_state
    while not solved(state):
        stuckCount = 0
        temperature = max_moves
        oldE = energy(state)
        accepted = False
        if oldE <= 0:
            accepted = True

        while not accepted:
            previousE = oldE
            for i in range(0, max_moves):
                if i == 99:
                    pass
                moves = all_moves(state)
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
                print(state)
                print(oldE)
                if oldE <= 0:
                    accepted = True
                    break

            temperature = temperature - 1
            if oldE <= 0:
                accepted = True
                break
            if oldE >= previousE:
                stuckCount += 1
            else:
                stuckCount = 0
            if stuckCount > 80:
                temperature += 2
            if solved(state):
                print(state)
                break

    print(state)


def sim_annealing_lucky(initial_state, max_moves, p=1):
    print("Starting from:")
    print(initial_state)
    # rename the initial state:
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


def sim_annealing(initial_state, intial_temperature, p=1):
    print("Starting from:")
    print(initial_state)
    # rename the initial state:
    state = initial_state
    temperature = intial_temperature
    while not solved(state) and temperature > 0:
        oldE = energy(state)
        moves = all_moves(state)
        accepted = False
        stuckCount = 0
        # Make a move
        while not accepted:
            # avoid stuck
            stuckCount += 1
            if stuckCount >= 5:
                temperature += 2
            # move randomly
            m = random.choice(moves)
            do_move(state, m)
            # accept move?
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
            # not accept + undo
            if not accepted:
                undo_move(state, m)

        temperature = temperature - 1
        if temperature <= 0:
            temperature += 2
        print("Moving", m)
        print(state)
        print(energy(state))


# make a real valid move on puzzle
def make_real_move(state, temperature, p):
    oldE = energy(state)  # Get old energy
    moves = all_moves(state)  # Explore all moves
    accepted = False
    stuckCount = 0
    # while not accepted the new move -> Loop until accept one!
    while not accepted:
        # avoid stuck
        stuckCount += 1
        if stuckCount >= 5:
            temperature += 2
        # move randomly
        m = random.choice(moves)
        do_move(state, m)
        # accept move?
        newE = energy(state)
        deltaE = newE - oldE
        if deltaE <= 0:  # accept
            accepted = True
        else:
            boltz = math.exp(-float(p * deltaE) / temperature)
            # A random float between 0 and 1:
            r = np.random.uniform(1, 0, 1)
            if r <= boltz:  # accept
                accepted = True
        # not accept + undo
        if not accepted:
            undo_move(state, m)

    return temperature, m


# Simulated annealing 1 with cooling function: temperature = temperature - 1
def sim_annealing1(initial_state, intial_temperature, p=1):
    list = []
    print("Starting from:")
    print(initial_state)
    # rename the initial state:
    state = initial_state
    temperature = intial_temperature

    # while puzzle is not solved and temperature > 0
    while not solved(state) and temperature > 0:
        # Make a real move
        temperature, m = make_real_move(state, temperature, p)
        list.append(m)

        # Decrease temperature
        temperature = temperature - 1
        # If not solved and temperature <= 0
        if temperature <= 0:
            temperature += 2
        # print("Moving", m)
        # print(state)
        # print(energy(state))

    return list


# Simulated annealing 2 with cooling function: temperature = temperature * 0.99
def sim_annealing2(initial_state, intial_temperature, p=1):
    print("Starting from:")
    print(initial_state)
    # rename the initial state:
    state = initial_state
    temperature = intial_temperature

    # while puzzle is not solved and temperature > 0
    while not solved(state) and temperature > 0:
        # Make a real move
        temperature, m = make_real_move(state, temperature, p)

        # Decrease temperature
        temperature = temperature * 0.99
        # If not solved and temperature <= 0
        if temperature <= 0:
            temperature += 2
        # print("Moving", m)
        # print(state)
        # print(energy(state))


def sim_annealing3(initial_state, intial_temperature, p=1):
    print("Starting from:")
    print(initial_state)
    # rename the initial state:
    state = initial_state
    maxTemperature = intial_temperature  # max temperature to decrease
    temperature = random.uniform(0, 1) * maxTemperature  # always < maxTemperature

    while not solved(state) and temperature > 0:
        oldE = energy(state)  # Get old energy
        moves = all_moves(state)  # Explore all moves
        accepted = False
        stuckCount = 0
        # Make a move
        while not accepted:
            # avoid stuck
            stuckCount += 1
            if stuckCount >= 5:
                temperature += 2
            # move randomly
            m = random.choice(moves)
            do_move(state, m)
            # accept move?
            newE = energy(state)
            deltaE = newE - oldE
            if deltaE <= 0:  # accept
                accepted = True
            else:
                boltz = math.exp(-float(p * deltaE) / temperature)
                # A random float between 0 and 1:
                r = np.random.uniform(1, 0, 1)
                if r <= boltz:  # accept
                    accepted = True
            # not accept + undo
            if not accepted:
                undo_move(state, m)

        # Decrease max temperature
        maxTemperature = maxTemperature - 1
        if maxTemperature <= 0:
            maxTemperature += 2
        # Calculate real temperature
        temperature = random.uniform(0, 1) * maxTemperature  # always < maxTemperature

        # print("Moving", m)
        # print(state)
        # print(energy(state))


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


def task():
    if len(move) != 0:
        UpdateShowState(gameState1, move[0])
        del move[0]
        show(gameState1)
        time.sleep(0.1)
        top.after(100, task)


# test1 = [1, 2, 3, 4, 5, 6, 7, 10, 12, 0, 8, 15, 14, 11, 9, 13]
# gameState1=Custominput(test1,4)

# test2 = [4, 6, 5, 7, 2, 0, 1, 3, 8]
test = [1, 2, 3, 4, 5, 6, 7, 8, 0]
gameState1 = Custominput(test, 3)
proper_shuffle(gameState1, 20)
gameState2 = copy.deepcopy(gameState1)
gameState3 = copy.deepcopy(gameState1)

print(gameState1)
# print(PosOfPlayer(gameState1))
print(energy(gameState1))
show(gameState1)

temperature = 100
p = 1

start = time.time()  # start time
sim_annealing1(gameState1, temperature, p)
end = time.time()  # end time
print(end - start)

start = time.time()
sim_annealing2(gameState2, temperature, p)
end = time.time()
print(end - start)

start = time.time()
sim_annealing3(gameState3, temperature, p)
end = time.time()
print(end - start)

top.mainloop()
