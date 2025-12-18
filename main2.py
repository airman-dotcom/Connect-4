import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, request,jsonify

import numpy as np

class Game:
    def __init__(self):
        self.board = [[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]]
        self.fulls = []
    def evaluate(self):
        row_check = [str(item) for sublist in self.board for item in sublist]
        row_check = "".join(row_check)
        col_check = ""
        diag_check1 = []
        diag_check2 = []
        for i in range(len(self.board[0])):
            for j in range(len(self.board)):
                col_check += str(self.board[j][i])
        for i in range(3):
            for j in range(4):
                diag_check1.append(str(self.board[i][j]) + str(self.board[i+1][j+1]) + str(self.board[i+2][j+2]) + str(self.board[i+3][j+3]))

        for i in [5,4,3]:
            for j in range(4):
                diag_check2.append(str(self.board[i][j]) + str(self.board[i-1][j+1]) + str(self.board[i-2][j+2]) + str(self.board[i-3][j+3]))
        
        if "1111" in row_check or "1111" in col_check or "1111" in diag_check1 or "1111" in diag_check2:
            return [True,1]
        if "2222" in row_check or "2222" in col_check or "2222" in diag_check1 or "2222" in diag_check2:
            return [True,2]
        return [False]
    
    def place(self, column, player):
        #player 1 = 1, player 2 = 2
        col = []
        for i in range(6):
            col.append(self.board[i][column])
        col.reverse()
        #print(col.index(0))
        if col.count(0) == 1:
            self.fulls.append(column)
        self.board[len(self.board)-1-col.index(0)][column] = player
    
    def output(self):
        return self.board

    def output_fulls(self):
        return self.fulls
        

class Player:
    def __init__(self, player_num):
        self.qtable = []
        self.states = {}
        self.player_num = player_num
        self.log = []
        self.random_rate = 0.9

    def remember(self,board):
        f = []
        g = ""
        for i in board:
            u = 0
            for j in range(len(i)):
                u+= 10**j * i[j]
            f.append(u)
        #for j in range(len(f)):
        #    g+= 10**j * f[j]
        f = [str(i) for i in f]
        g = "".join(f)
        self.board = g
    
    def forward(self, fulls):
        if self.board not in list(self.states.keys()):
            if len(self.qtable) == 0:
                self.states[self.board] = 0
            else:
                self.states[self.board] = len(self.qtable)
            self.qtable.append([])
            self.qtable[self.states[self.board]] = [np.random.random(),np.random.random(),np.random.random(),np.random.random(),np.random.random(),np.random.random(),np.random.random()]
        use = list(self.qtable[self.states[self.board]])
        for i in fulls:
            use[i] = -999
        
        move = use.index(max(use))
        if np.random.random() <= self.random_rate:
            move = np.random.randint(0,len(use))
            while use[move] == -999:
                move = np.random.randint(0,len(use))
        self.log.append([self.board,move])
        return move

    def evaluate(self, win):
        if win == 0:
            for i in self.log:
                ind = self.states[i[0]]
                self.qtable[ind][i[1]] -= 0.25
        if win == 1:
            for i in self.log:
                ind = self.states[i[0]]
                self.qtable[ind][i[1]] += 0.5
        self.log = []
        self.random_rate*=0.99
    def output(self):
        return [self.qtable,self.states]
    
p1 = Player(1)
p2 = Player(2)

def play():
    g = Game()
    for i in range(20):
        p1.remember(g.output())
        fulls = g.output_fulls()
        m1 = p1.forward(fulls)
        g.place(m1, 1)
        decision = g.evaluate()
        if decision[0] == True:
            if decision[1] == 1:
                print("Player " + str(decision[1]) + " has won!!!")
                p1.evaluate(1)
                p2.evaluate(0)
            else:
                p1.evaluate(0)
                p2.evaluate(1)
            return
        fulls = g.output_fulls()
        p2.remember(g.output())
        m2 = p2.forward(fulls)
        g.place(m2, 2)
        decision = g.evaluate()
        if decision[0] == True:
            print("Player " + str(decision[1]) + " has won!!!")
            if decision[1] == 1:
                p1.evaluate(1)
                p2.evaluate(0)
            else:
                p1.evaluate(0)
                p2.evaluate(1)
            return
        for row in g.output():
            print(*row, sep='\t')
        print()
def round():
    g = Game()
    for i in range(21):
            p1.remember(g.output())
            fulls = g.output_fulls()
            m1 = p1.forward(fulls)
            g.place(m1, 1)
            decision = g.evaluate()
            if decision[0] == True:
                if decision[1] == 1:
                    #print("Player " + str(decision[1]) + " has won!!!")
                    p1.evaluate(1)
                    p2.evaluate(0)
                else:
                    p1.evaluate(0)
                    p2.evaluate(1)
                return
            fulls = g.output_fulls()
            p2.remember(g.output())
            m2 = p2.forward(fulls)
            g.place(m2, 2)
            decision = g.evaluate()
            if decision[0] == True:
                #print("Player " + str(decision[1]) + " has won!!!")
                if decision[1] == 1:
                    p1.evaluate(1)
                    p2.evaluate(0)
                else:
                    p1.evaluate(0)
                    p2.evaluate(1)
                return
    p1.evaluate(0)
    p2.evaluate(0)
def train():
    for j in range(1000):
        print(j)
        round()
    #play()
    
train()
random_rate = 0.9
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def handle_data():
    if request.method == 'POST':
        return jsonify({"data": p2.output()}), 200

    elif request.method == 'GET':
        # Return some data for a GET request
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



#p = Player()


#train()