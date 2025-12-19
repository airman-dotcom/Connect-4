import numpy as np
from flask import Flask, render_template, request,jsonify

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
        for row in self.board:
            if "1111" in "".join(map(str, row)):
                return [True, 1]
            if "2222" in "".join(map(str, row)):
                return [True, 2]

        if "1111" in col_check or "1111" in diag_check1 or "1111" in diag_check2:
            return [True,1]
        if "2222" in col_check or "2222" in diag_check1 or "2222" in diag_check2:
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
    def __init__(self,player_number):
        self.player_number = player_number
        self.plays = ""
        self.probs = {}
    
    def remember(self, p_move):
        self.plays += str(p_move)

    def forward(self,fulls):
        if len(self.plays) == 0:
            self.probs = {"0":0.1428,"1":0.1428,"2":0.1428,"3":0.1428,"4":0.1428,"5":0.1428,"6":0.1428}
            move = np.random.randint(0,7)
            self.plays += str(move)
            return move
        if self.plays not in self.probs.keys():
            self.probs[self.plays] = {"0":0.1428,"1":0.1428,"2":0.1428,"3":0.1428,"4":0.1428,"5":0.1428,"6":0.1428}
            move = np.random.randint(0,7)
            while move in fulls:
                move = np.random.randint(0,7)
            self.plays += str(move)
            return move
        move = None
        probs = list(self.probs[self.plays].values())
        r = np.random.random()
        for i in range(len(probs)-1):
            if r <= probs[i]:
                move = 0
                break
            if r > probs[i] and r<=probs[i+1]:
                move = i
                break
        if move == None:
            move = 6

        while move in fulls:
            r = np.random.random()
            for i in range(len(probs)-1):
                if r <= probs[i]:
                    move = 0
                    break
                if r > probs[i] and r<=probs[i+1]:
                    move = i
                    break
            if move == None:
                move = 6
        
        self.plays += str(move)
        return move
    
    def evaluate(self,win):
        if self.player_number == 1:
            usables = [self.plays[:i] for i in range(3,len(self.plays)+1,2)]
        else:
            usables = [self.plays[:i] for i in range(2,len(self.plays),2)]
        if win == 1:
            for u in usables:
                ku = u[:-1]
                self.probs[ku][u[-1]] = min(1,self.probs[ku][u[-1]]+(0.2 + 0.2/6))
                for l in range(7):
                    self.probs[ku][str(l)]= max(0,self.probs[ku][str(l)]-0.2/6)
        else:
            for u in usables:
                ku = u[:-1]
                self.probs[ku][u[-1]] = max(0,self.probs[ku][u[-1]]-(0.2 + 0.2/6))
                for l in range(7):
                    self.probs[ku][str(l)] = min(1,self.probs[ku][str(l)]+0.2/6)
        self.plays = ""
    
    def reset(self):
        self.plays = ""
    
    def output(self):
        return self.probs
    

p1 = Player(1)
p2 = Player(2)

def play():
    g = Game()
    for i in range(21):
        fulls = g.output_fulls()
        m1 = p1.forward(fulls)
        g.place(m1,1)
        p2.remember(m1)
        decision = g.evaluate()
        if decision[0] == True:
            print("Player " + str(decision[1]) + " has won!!!")
            p1.evaluate(1)
            p2.evaluate(0)
            return
        fulls = g.output_fulls()
        m2 = p2.forward(fulls)
        g.place(m2,2)
        p1.remember(m2)
        decision = g.evaluate()
        if decision[0] == True:
            print("Player " + str(decision[1]) + " has won!!!")
            p1.evaluate(1)
            p2.evaluate(0)
            return
        for row in g.output():
            print(*row, sep='\t')
        print()
    p1.reset()
    p2.reset()

def round():
    g = Game()
    for i in range(21):
        fulls = g.output_fulls()
        m1 = p1.forward(fulls)
        g.place(m1,1)
        p2.remember(m1)
        decision = g.evaluate()
        if decision[0] == True:
            p1.evaluate(1)
            p2.evaluate(0)
            return
        fulls = g.output_fulls()
        m2 = p2.forward(fulls)
        g.place(m2,2)
        p1.remember(m2)
        decision = g.evaluate()
        if decision[0] == True:
            p1.evaluate(1)
            p2.evaluate(0)
            return
    p1.reset()
    p2.reset()

def train():
    for j in range(1000):
        print(j)
        round()
    #play()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def handle_data():
    if request.method == 'POST':
        return jsonify({"data": p2.output()}), 200

    elif request.method == 'GET':
        # Return some data for a GET request
        return render_template('index2.html')

if __name__ == '__main__':
    train()
    app.run(debug=True)
    