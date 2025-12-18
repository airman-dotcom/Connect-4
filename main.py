import networkx as nx
import os
import matplotlib.pyplot as plt
from pyvis.network import Network
from threading import Thread
import numpy as np
from tqdm import tqdm
G = nx.Graph() # Or nx.DiGraph() for a directed graph
num_neurons = 10
num_beginners = 1
num_evolutions = 1
num_children_rate = 0.6
max_iteration = 20
num_random_survivors = 2
learning_rate = 1
num_children_surviving = 10
begin_evolution_change = 32
mutation_rate = 0.9
#num_enders = num_beginners
num_connections = 2
min_signal_strength = 0.01
oper_lb = 1
oper_hb = 2
stop_range = 0.01
delta_loss = 0.2
gsum = []
symbols = [None,"+","-","*","/"]
x = []
y = []
#Test 1: optimal number of neurons set about 60
#Test 2: optimal number of children_surviving set about 10
#Test 3: optimal number of connections/node = 1
#Test 6: optimal upper and lower bound: (-86,86)
#Test 7: optimal number of nodes either 85 (1.7), or 106 (0.6)
#Test 8: optimal number of nodes either at 63 (2.7) or 68 (4.2)
#Test 9: ran to determine whether number of nodes from 60-75 would yield same results, did not
#Test 10: optimal mutation rate at 0.3 (1.6) and 0.6 (1.4)
#Test 11: optimal num_neurons at 106 (0.8 error)

class Game:
    def __init__(self):
        self.board = [[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]]
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
                diag_check1.append(str(self.board[i][j]) + str(self.board[i-1][j+1]) + str(self.board[i-2][j+2]) + str(self.board[i-3][j+3]))
        
        if "1111" in row_check or "1111" in col_check or "1111" in diag_check1 or "1111" in diag_check2:
            return [True,1]
        if "2222" in row_check or "2222" in col_check or "2222" in diag_check1 or "2222" in diag_check2:
            return [True,2]
        return [False]
    
    def place(self, column, player):
        #player 1 = 1, player 2 = 2
        col = []
        for i in range(6):
            col.append(self.board[i][column-1])
        col.reverse()
        #print(col.index(0))
        if 0 not in col:
            return "Impossible"
        self.board[len(self.board)-1-col.index(0)][column-1] = player
        return self.evaluate()
    
    def output(self):
        return self.board
        

class Player:
    def __init__(self, board, player_num):
        self.board = [item for sublist in board for item in sublist]
        self.player_num = player_num
        self.graph = nx.Graph()
        self.gsum = []
        if player_num == 1:
            self.board = [x if x == 0 or x == 1 else -1 for x in self.board]
        else:
            self.board = [x if x == 0 or x == 1 else -1 for x in self.board]

        opers = np.linspace(oper_lb,oper_hb,num_neurons-num_beginners).tolist()
        self.nodes = [str(i) for i in opers]
            # Add nodes
        for i in range(42):
            self.nodes.append("Initial"+str(i))
        self.graph.add_nodes_from(["Final"]+self.nodes)
        connections = []
        for i in self.nodes:
            for j in range(num_connections):
                c = self.nodes[np.random.randint(0,len(self.nodes))]
                connections.append((i,c))
        self.graph.add_edges_from(connections)
        for i in range(num_beginners):
            c = self.nodes[np.random.randint(0,len(self.nodes))]
            #d = nodes[np.random.randint(0,len(nodes))]
            self.graph.add_edge(c,"Final")
        for i in range(42):
            self.graph.nodes["Initial"+str(i)]['w'] = self.board[i]
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(self.graph)
        # Show the interactive graph
        net.show("interactive_graph.html")


    def fforward(self):
        for i in range(42):
            u = "Initial"+str(i)
            self.forward(float(self.board[i]),1,u,[u],u)


    def forward(self, signal, loss, node,traveled,output):
        global min_signal_strength,delta_loss
        if loss <= 0:
            return 0
        
        if node == "Final":
            #print(self.gsum)
            self.gsum.append(signal)
            return 0
        
        neighbors = [i for i in list(self.graph.neighbors(node))]# if i not in traveled]
        if "Initial" in node:
            node = self.graph.nodes[node]['w']
        for neigh in neighbors:
            u = traveled
            u.append(node)
            self.forward((signal+float(node))*loss,loss-delta_loss,neigh,u,output+","+str(neigh))

    def evolve(changes, self):
        for i in range(changes):
            edges = list(self.graph.edges)
            if np.random.rand() <= mutation_rate and edges:
                use = edges[np.random.randint(0,len(edges))]
                while "Intial" in use[0] or "Initial" in use[1] or use[0] == "Final" or use[1] == "Final":
                    use = edges[np.random.randint(0,len(edges))]
                self.graph.remove_edge(use[0],use[1])
            c = self.nodes[np.random.randint(0,len(self.nodes))]
            d = self.nodes[np.random.randint(0,len(self.nodes))]
            self.graph.add_edge(c,d)
    
    def remember(self,board):
        self.board = [item for sublist in board for item in sublist]
        if self.player_num == 1:
            self.board = [x if x == 0 or x == 1 else -1 for x in self.board]
        else:
            self.board = [x if x == 0 or x == 1 else -1 for x in self.board]


    def evaluate(self):
        l = self.gsum
        self.gsum = []
        return sum(l)


#p = Player()
g = Game()
p1 = Player(g.output(), 1)
p2 = Player(g.output(),2)
d = []
def sigmoid(x):
    return x*7/600

def train():
    global x,y,gsum
    stack = begin_evolution_change/learning_rate
    best_networks_results = []
    #pbar = tqdm(total=1)
    finished = False
    for i in range(20):
        #while not finished:
        p1.remember(g.output())
        p1.fforward()
        m1 = round(sigmoid(p1.evaluate()))
        if m1 > 6:
            m1 = 6
        g.place(m1+1,1)
        decision = g.evaluate()
        if decision[0] == True:
            print("Player " + str(decision[1]) + " has won!!!")
            for row in g.output():
                print(*row, sep='\t')
            return

        p2.remember(g.output())
        p2.fforward()
        m2 = round(sigmoid(p2.evaluate()))
        if m2 > 6:
            m2 = 6
        g.place(m2+1,2)
        decision = g.evaluate()
        if decision[0] == True:
            print("Player " + str(decision[1]) + " has won!!!")
            for row in g.output():
                print(*row, sep='\t')
            return
        
        #pbar.update(1)
    #pbar.close()

train()
#print(d)
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('My Histogram')
#plt.hist(d)
#plt.show()
    # Draw the graph
#nx.draw(G, with_labels=True, node_color='lightblue', node_size=1000, font_size=10)
#plt.title("Node Connections")
#plt.show()

"""

def fforward(tokens,network):
    for token in tokens:
        forward(network, float(token),1,"Initial",["Initial"],"Initial")

def forward(network, signal, loss, node,traveled,output):
    global gsum,min_signal_strength,delta_loss
    #print(abs(signal))
    if abs(signal) <= min_signal_strength or loss < 0:
        return 0
    if node == "Final":
        gsum.append(signal)
        return 0
    
    neighbors = [i for i in list(network.neighbors(node))]# if i not in traveled]
    if node == "Initial":
        node = "0"
    for neigh in neighbors:
        u = traveled
        u.append(node)
        forward(network,(signal+float(node))*loss,loss-delta_loss,neigh,u,output+","+str(neigh))

# run it
#print(forward(2, "Initial",G,0,0,[]))

def evolve(changes, network):
    
    for i in range(changes):
        edges = list(network.edges)
        if np.random.rand() <= mutation_rate and edges:
            use = edges[np.random.randint(0,len(edges))]
            while use[0] == "Initial" or use[1] == "Initial" or use[0] == "Final" or use[1] == "Final":
                use = edges[np.random.randint(0,len(edges))]
            network.remove_edge(use[0],use[1])
        c = nodes[np.random.randint(0,len(nodes))]
        d = nodes[np.random.randint(0,len(nodes))]
        network.add_edge(c,d)
    return network

def prepare(expr):
    almost = tokenize(expr)
    for i in range(len(almost)):
        if almost[i] in symbols:
            almost[i] = symbols.index(almost[i])
    return almost

def tokenize(expr):
    tokens = []
    num = ""
    for ch in expr:
        if ch.isdigit():
            num += ch
        else:
            if num:
                tokens.append(num)
                num = ""
            if ch.strip():
                tokens.append(ch)
    if num:
        tokens.append(num)
    return tokens

def train(expression,networks):
    global num_children,x,y,gsum
    stack = begin_evolution_change/learning_rate
    best_networks = networks
    best_networks_results = []
    pbar = tqdm(total=num_evolutions)
    for i in range(num_evolutions):
        stack = int(stack * learning_rate)
        children = []
        fav_child = best_networks[0]
        if len(best_networks_results)>0:
            #print("Past: " + str(min(best_networks_results)))
            fav_child = best_networks[best_networks_results.index(min(best_networks_results))]
        for j in best_networks:
            children += [evolve(stack, j.copy()) for k in range(int(num_neurons*num_children_rate))]
        children += [fav_child]
        #if i == 0:
        #    sim_num_children = int(sim_num_children/num_children_surviving)
        output = []
        kkk = 1
        #print("AAAH" + str(len(children)))
        for child in children:
            
            fforward(prepare(expression),child)
            output.append(sum(gsum))
            #print(sum(gsum), kkk)
            kkk+=1
            gsum=[]
        #print("YAYAYA")
        results = [abs(float(eval(expression))-i) for i in output]
        pbar.update(1)
        #print("Network output:", results)
        copy_results = list(results)
        copy_children = list(children)
        #if len(best_networks_results)>0 and fav_child in copy_children:
            #print([fav_child])
            #print(len(copy_children))
            #print(len(copy_results))
            #print(copy_results[copy_children.index(fav_child)])
        #print(min(copy_results))

        x.append(i+1)
        y.append(abs(min(copy_results)))

        if abs(min(copy_results)) < stop_range:
            #print(copy_results)
            pbar.close()
            #print(abs(min(copy_results)))
            g = copy_children[copy_results.index(min(copy_results))]
            #print(int(np.round(fforward("2", "Initial",g,0,0,[]))))
            #print([g])
            return [copy_children[copy_results.index(min(copy_results))]]
                
        j = 0
        surviving_children = []
        surviving_children_results = []
        while j < num_children_surviving and copy_results and copy_children:
            lowest_index = copy_results.index(min(copy_results))
            surviving_children_results.append(min(copy_results))
            surviving_children.append(copy_children[lowest_index])
            if len(copy_results) > 0:
                del copy_results[lowest_index]
            if len(copy_children) > 0:
                del copy_children[lowest_index] 
            j+=1
        for j in range(num_random_survivors):
            if len(copy_results) > 0 and len(copy_children) > 0:
                r = np.random.randint(0,len(copy_results))
                surviving_children.append(copy_children[r])
                surviving_children_results.append(copy_results[r])
                del copy_results[r]
                del copy_children[r]
            
        best_networks = surviving_children
        best_networks_results = surviving_children_results
    pbar.close()
    return best_networks
best_networks = train("2",[G])
def test(x,network):
    global gsum
    i = tokenize(x)
    for j in range(len(i)):
        if i[j] in symbols:
            i[j] = symbols.index(i[j])
    
    fforward(i,network)
    print(gsum)
    return sum(gsum)
print(test("2",best_networks[0]))
print("3: " + str(test("3",best_networks[0])))
net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
#net.toggle_physics(False)
net.from_nx(self.graph)
# Show the interactive graph
net.show("interactive_graph.html")
#print(y)
plt.plot(x,y)
plt.show()
"""
    # Add nodes and edges from the NetworkX graph
