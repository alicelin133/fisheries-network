import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

import time

class Simulation(object):
    """A network of territories of fishers, where each territory harvests fish,
    and fish move between territories. The network has the following properties:

    Network attributes:
        G: networkx object, each node corresponds to a territory, each edge
        corresponds to a connection b/w territories that fish can leak through
        n_fishers: the number of territories in the network (1 fisher per
        territory)
        t: integer, the current time step *t* of the simulation
        delta: float, extent to which territories are ecologically
        connected
        q: float, "catchability" of fish resource
        r: float, growth factor of fish population 
        K: float, carrying capacity of fish population
        price: float, price received per unit of fish
        cost: float, cost per unit of effort invested in fishing
        noise: float, switching strategies occurs with +/- *noise*
        num_feedback: int, number of cycles of harvest/regrowth/leakage per time step
    Node (territory) attributes:
        R: float, territory's resource level
        e: float, territory's effort level
        dR: float, for internal use in calculating leakage.
        pi: float, territory's payoff from current iteration in feedback loop
        payoffs: np.array of floats, territory's payoffs from each iteration of feedback loop
        U: float, territory's utility from current time step
    """

    def __init__(self, n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise, num_feedback):
        """Return a Simulation object with a complete graph on *n_fishers* nodes,
        a leakage factor of *delta*, a fish catchability factor of *q*, and
        individual nodes with *R_0[i]* resource level and *e_0[i]* effort level."""
        self.check_sim_attributes(n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise)
        self.G = nx.complete_graph(n_fishers)
        self.n_fishers = n_fishers
        self.t = 0 # intialize time step t to 0
        self.delta = delta
        self.q = q
        self.r = r
        self.K = K
        self.price = price
        self.cost = cost
        self.noise = noise
        self.num_feedback = num_feedback
        for i in range(self.G.number_of_nodes()):
            self.G.node[i]['R'] = R_0[i]
            self.G.node[i]['e'] = e_0[i]
            self.G.node[i]['dR'] = 0.0
            self.G.node[i]['pi'] = 0.0

    def check_sim_attributes(self, n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise):
        """Checks that values of data attributes given in parameters to 
        constructor are in the correct format."""
        if delta < 0 or delta > 1:
            raise ValueError("delta must be in [0,1]")
        if q < 0 or q > 1:
            raise ValueError("q must be in [0,1]")
        if np.shape(R_0)[0] != n_fishers:
            raise ValueError("R_0 length must match n_fishers")
        if np.shape(e_0)[0] != n_fishers:
            raise ValueError("e_0 length must match n_fishers")
        for i in range(np.shape(e_0)[0]):
            if e_0[i] < 0 or e_0[i] > 1:
                raise ValueError("entry {} in e_0 must be in [0,1]".format(i))
        
    def simulate(self, maxstep):
        """Runs a discrete simulation of the fisheries network for *maxstep*
        time steps."""
        # e_data[nood][t] = e of node nood at time step t
        self.e_data = np.zeros((self.n_fishers, maxstep))
        self.R_data = np.zeros((self.n_fishers, maxstep))
        self.pi_data = np.zeros((self.n_fishers, maxstep))
        for t in range(maxstep):
            for nood in self.G.nodes(data=False):
                self.e_data[nood][t] = self.G.node[nood]['e']
                self.R_data[nood][t] = self.G.node[nood]['R']
                self.pi_data[nood][t] = self.G.node[nood]['pi']
            self.update_resource()
            self.update_strategy()
            self.t += 1

    def update_resource(self):
        """Runs the harvest(), regrowth(), and leakage() methods in a loop
        self.num_feedback number of times. Also calculates the utility for
        each fisher based on payoffs from all self.num_feedback harvests
        combined."""
        for nood in self.G.nodes(data=False):
            self.G.node[nood]['payoffs'] = np.zeros(self.num_feedback)
        for i in range(self.num_feedback):
            self.harvest()
            self.regrowth()
            self.leakage()
            for nood in self.G.nodes(data=False):
                self.G.node[nood]['payoffs'][i] = self.G.node[nood]['pi']
        self.calculate_utility()
    
    def calculate_utility(self):
        """Calculates utility for each node after one update_resource loop by
        summing payoffs from each harvest in the loop."""
        for nood in self.G.nodes(data=False):
            self.G.node[nood]['U'] = np.sum(self.G.node[nood]['payoffs'])

    def harvest(self):
        """Updates resource R for each territory based on e, the territory's
        effort level, for a single time step. Also calculates payoff from
        harvest."""
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            e = self.G.node[nood]['e']
            harvest = self.q * R * e
            if harvest > R: # case 1: fisher wants more fish than he could take
                harvest = R
                self.G.node[nood]['R'] = 0
                print("Resource is {} for node {}".format(self.G.node[nood]['R'], nood))
            else: # case 2: fisher wants less fish than he could take
                self.G.node[nood]['R'] = R - harvest
            self.G.node[nood]['pi'] = self.price * harvest - self.cost * e

    def regrowth(self):
        """Updates resource R for each territory based on the fish population's
        logistic growth, occurs after the time step's harvest."""
        for i in self.G.nodes(data=False):
            R = self.G.node[i]['R']
            self.G.node[i]['R'] = R + self.r * R * (1 - R/self.K)
            
    def leakage(self):
        """Updates resource R for each territory based on resource leakage
        between territories. Each territory i gives (delta * R / self.G.degree(i))
        amount of resource to each of its adjacent territories."""
        # calculate how much fish go to each territory
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            deg = self.G.degree(nood)
            for neighbor in self.G[nood]:
                self.G.node[neighbor]['dR'] += R / (deg + 1)
        # update resource levels for each territory
        for nood in self.G.nodes(data=False):
            R = self.G.node[nood]['R']
            deg = self.G.degree(nood)
            dR = self.G.node[nood]['dR']
            R_new = R * (1 - self.delta * deg / (deg + 1)) + self.delta * dR
            self.G.node[nood]['R'] = R_new
            self.G.node[nood]['dR'] = 0 # RESET dR TO 0 FOR EACH NODE
    
    def update_strategy(self):
        """Selects one fisher randomly for possible strategy change. With
        small probability, new strategy is randomly chosen from [0,1].
        Otherwise, select a second fisher to compare payoff against. If 2nd
        fisher has lower payoff, do nothing. If 2nd fisher has higher payoff,
        1st fisher switches to effort level of 2nd fisher with prob.
        proportional to difference of payoffs, and with some noise. Repeat
        process self.n_fishers times."""
        for i in range(self.n_fishers): # initialize e_new attribute
            self.G.node[i]['e_new'] = self.G.node[i]['e']
        for i in range(1):
            fisher1 = np.random.randint(0,self.n_fishers)
            fisher2 = np.random.randint(0,self.n_fishers)
            U1 = self.G.node[fisher1]['U']
            # global mutation
            prob_mutation = 0
            rand = np.random.random()
            if rand < prob_mutation:
                # TODO: why multiply by self.r?????
                self.G.node[fisher1]['e_new'] = np.random.random() * self.r
            else:
                U2 = self.G.node[fisher2]['U']
                if U1 < U2:
                    if U1 != 0 or U2 != 0:
                        # Probability that fisher1 switches to fisher2's strategy
                        prob_switch = (U2 - U1) / (abs(U1) + abs(U2))
                    else:
                        prob_switch = 0 # in case both payoffs are 0
                    rand = np.random.random()
                    if rand < prob_switch:
                        diff = np.random.uniform(-1 * self.noise, self.noise)
                        e_new = self.G.node[fisher2]['e'] + diff
                        # ensure that e_new stays in [0,1]
                        if e_new < 0:
                            e_new = 0
                        elif e_new > 1:
                            e_new = 1
                        self.G.node[fisher1]['e_new'] = e_new
        for i in range(self.n_fishers):
            self.G.node[i]['e'] = self.G.node[i]['e_new']
            
def main():
    """Performs unit testing."""
    start_time = time.time()        
    # Parameters: n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise
    n_fishers = 10
    delta = 0.001
    q = 1
    r = 0.05
    K = 1000
    R_0 = np.full(n_fishers,K/2)
    e_0 = np.linspace(0,0.05,num=n_fishers)
    price = 1
    cost = 0.5
    noise = 0.01
    num_feedback = 5
    num_steps = 1000
    # Creating Simulation object
    my_sim2 = Simulation(n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise, num_feedback)
    my_sim2.simulate(num_steps)
    fig = plt.figure()
    plt.suptitle("Full fish movement")
    # Plotting resource levels vs. time
    ax1 = fig.add_subplot(2,2,1)
    for i in range(my_sim2.n_fishers):
        ax1.plot(np.arange(num_steps), my_sim2.R_data[i])
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Resource (K = {})".format(my_sim2.K))
    ax1.set_title("Territory Resource Levels vs. Time")
    ax1.grid(True)
    # Plotting avg payoff vs. time
    pi_avg = np.average(my_sim2.pi_data, axis=0)
    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(np.arange(num_steps), pi_avg)
    ax2.set_xlabel("Time steps")
    ax2.set_ylabel("Average payoff")
    ax2.set_title("Average Payoff vs. Time")
    ax2.grid(True)
    # Plotting avg effort vs. time
    e_avg = np.average(my_sim2.e_data, axis=0)
    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(np.arange(num_steps), e_avg)
    ax3.set_xlabel("Time steps")
    ax3.set_ylabel("Effort")
    ax3.set_title("Average Effort vs. Time")
    ax3.grid(True)
    # Plotting all efforts vs. time
    ax4 = fig.add_subplot(2,2,4)
    for i in range(my_sim2.n_fishers):
        ax4.plot(my_sim2.e_data[i])
    ax4.set_xlabel("Time steps")
    ax4.set_ylabel("Effort")
    ax4.set_title("Effort vs. Time")
    ax4.grid(True)
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    # Plotting standard deviation of effort over time
    fig2 = plt.figure()
    e_stddev = np.std(my_sim2.e_data, axis=0)
    ax = fig2.add_subplot(1,1,1)
    ax.plot(e_stddev)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Standard deviation of effort")
    plt.grid(b=True)

    print("Last time step avg payoff: {}".format(pi_avg[-1]))
    print("Last time step avg effort: {}".format(e_avg[-1]))
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()    
    
if __name__ == "__main__":
    main()
