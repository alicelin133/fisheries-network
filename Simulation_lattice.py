import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

import time

class Simulation_arrays(object):
    """A square 2D grid network of territories of fishers, where each territory harvests fish,
    and fish move between territories. The network has the following properties:

    Network attributes:
        G: 2d grid graph in networkx, each node corresponds to a territory, each edge
            corresponds to a connection b/w territories that fish can leak through
        n_fishers: the number of territories in the total network
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
        payoff_discount: ratio by which older payoffs are exponentially less
            valued than recent payoffs
        R: np array, R[i] = resource level at territory i
        e: np array, e[i] = effort level at territory i
        dR: np array, for internal use in calculating leakage.
        payoffs: np array, payoffs[i][j] = iteration i of update_resource
            feedback loop, territory j's payoff
        U: np array, U[i] = utility from current time step at territory i
    Node (territory) attributes:
        none, hopefully
    """

    def __init__(self, n_fishers_length, delta, q, r, K, R_0, e_0, price, cost, noise, num_feedback, payoff_discount, num_steps):
        """Return a Simulation object with a complete graph on *n_fishers* nodes,
        a leakage factor of *delta*, a fish catchability factor of *q*, and
        individual nodes with *R_0[i]* resource level and *e_0[i]* effort level."""
        self.check_sim_attributes(n_fishers_length, delta, q, r, K, R_0, e_0, price, cost, noise)
        self.G = nx.grid_2d_graph(n_fishers_length, n_fishers_length, periodic=False, create_using=None)
        self.n_fishers = n_fishers_length ** 2
        self.t = 0 # TODO: probably not necessary
        self.delta = delta
        self.q = q
        self.r = r
        self.K = K
        self.price = price
        self.cost = cost
        self.noise = noise
        self.num_feedback = num_feedback
        self.R_0 = np.copy(R_0) # TODO: may delete this later
        self.e_0 = np.copy(e_0) # TODO: may delete this later (needed for file name tho)
        self.R = np.copy(R_0)
        self.e = np.copy(e_0)
        self.U = np.zeros(self.n_fishers)
        self.payoff_discount = payoff_discount
        self.payoffs = np.zeros((num_feedback, self.n_fishers))
        self.num_steps = num_steps
        self.dR = np.zeros(self.n_fishers)
        self.e_new = np.zeros(self.n_fishers)
        self.e_data = np.zeros((self.n_fishers, num_steps))
        self.R_data = np.zeros((self.n_fishers, num_steps))
        self.U_data = np.zeros((self.n_fishers, num_steps))

    def check_sim_attributes(self, self.n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise):
        """Checks that values of data attributes given in parameters to 
        constructor are in the correct format."""
        # TODO: not that important, but update this to include the new parameters.
        if delta < 0 or delta > 1:
            raise ValueError("delta must be in [0,1]")
        if q < 0 or q > 1:
            raise ValueError("q must be in [0,1]")
        if np.shape(R_0)[0] != self.n_fishers:
            raise ValueError("R_0 length must match self.n_fishers")
        if np.shape(e_0)[0] != self.n_fishers:
            raise ValueError("e_0 length must match self.n_fishers")
        for i in range(np.shape(e_0)[0]):
            if e_0[i] < 0 or e_0[i] > 1:
                raise ValueError("entry {} in e_0 must be in [0,1]".format(i))
        
    def simulate(self):
        """Runs a discrete simulation of the fisheries network for self.num_steps
        time steps."""
        # e_data[nood][t] = e of node nood at time step t
        for t in range(self.num_steps):
            for nood in self.G.nodes(data=False):
                self.e_data[nood][t] = self.e[nood]
                self.R_data[nood][t] = self.R[nood]
                self.U_data[nood][t] = self.U[nood]
            self.update_resource()
            self.update_strategy()

    def update_resource(self):
        """Runs the harvest(), regrowth(), and leakage() methods in a loop
        self.num_feedback number of times. Also calculates the utility for
        each fisher based on payoffs from all self.num_feedback harvests
        combined."""
        for i in range(self.num_feedback):
            self.harvest(i)
            self.regrowth()
            self.leakage()
        self.calculate_utility()
    
    def calculate_utility(self):
        """Calculates utility for each node after one update_resource loop by
        summing payoffs from each harvest in the loop."""
        for nood in self.G.nodes(data=False):
            current_discount = 1
            U = 0
            for i in range(self.num_feedback):
                U += current_discount * self.payoffs[-1 - i][nood]
                current_discount = self.payoff_discount * current_discount
            self.U[nood] = U

    def harvest(self, num_iteration):
        """Updates resource R for each territory based on e, the territory's
        effort level, for a single time step. Also calculates payoff from
        harvest."""
        for nood in self.G.nodes(data=False):
            R = self.R[nood]
            e = self.e[nood]
            harvest = self.q * R * e
            if harvest > R: # case 1: fisher wants more fish than he could take
                harvest = R
                self.R[nood] = 0
                print("Resource is {} for node {}".format(R, nood))
            else: # case 2: fisher wants less fish than he could take
                self.R[nood] = R - harvest
            self.payoffs[num_iteration][nood] = self.price * harvest - self.cost * e

    def regrowth(self):
        """Updates resource R for each territory based on the fish population's
        logistic growth, occurs after the time step's harvest."""
        for i in self.G.nodes(data=False):
            R = self.R[i]
            self.R[i] = R + self.r * R * (1 - R/self.K)
            
    def leakage(self):
        """Updates resource R for each territory based on resource leakage
        between territories. Each territory i gives (delta * R / self.G.degree(i))
        amount of resource to each of its adjacent territories."""
        # calculate how much fish go to each territory
        for nood in self.G.nodes(data=False):
            R = self.R[nood]
            deg = self.G.degree(nood)
            for neighbor in self.G[nood]:
                self.dR[neighbor] += R / (deg + 1)
        # update resource levels for each territory
        for nood in self.G.nodes(data=False):
            R = self.R[nood]
            deg = self.G.degree(nood)
            dR = self.dR[nood]
            R_new = R * (1 - self.delta * deg / (deg + 1)) + self.delta * dR
            self.R[nood] = R_new
            self.dR[nood] = 0 # RESET dR TO 0 FOR EACH NODE
    
    def update_strategy(self):
        """Selects one fisher randomly for possible strategy change. With
        small probability, new strategy is randomly chosen from [0,1].
        Otherwise, select a second fisher to compare payoff against. If 2nd
        fisher has lower payoff, do nothing. If 2nd fisher has higher payoff,
        1st fisher switches to effort level of 2nd fisher with prob.
        proportional to difference of payoffs, and with some noise. Repeat
        process self.n_fishers times."""
        for i in range(self.n_fishers): # initialize e_new array
            self.e_new[i] = self.e[i]
        for i in range(1):
            fisher1 = np.random.randint(0,self.n_fishers)
            fisher2 = np.random.randint(0,self.n_fishers)
            U1 = self.U[fisher1]
            # global mutation
            prob_mutation = 0
            rand = np.random.random()
            if rand < prob_mutation:
                self.G.node[fisher1]['e_new'] = np.random.random()
            else:
                U2 = self.U[fisher2]
                if U1 < U2:
                    if U1 != 0 or U2 != 0:
                        # Probability that fisher1 switches to fisher2's strategy
                        prob_switch = (U2 - U1) / (abs(U1) + abs(U2))
                    else:
                        prob_switch = 0 # in case both payoffs are 0
                    rand = np.random.random()
                    if rand < prob_switch:
                        diff = np.random.uniform(-1 * self.noise, self.noise)
                        e_new = self.e[fisher2] + diff
                        # ensure that e_new stays in [0,1]
                        if e_new < 0:
                            e_new = 0
                        elif e_new > 1:
                            e_new = 1
                        self.e_new[fisher1] = e_new
        for i in range(self.n_fishers):
            self.e[i] = self.e_new[i]

def calculate_e_msr(n_fishers, q, r, K, price, cost):
    """Calculates value of e_msr (maximum sustainable rent)."""
    return r * (price * q * K * n_fishers - n_fishers * cost) / (2 * price * q * q * K * n_fishers)

def calculate_e_nash(e_msr, n_fishers):
    """Calculates value of Nash equilibrium level of effort."""
    return e_msr * 2 * n_fishers / (1 + n_fishers)

def main():
    """Performs unit testing."""
    start_time = time.time()        
    # Parameters: n_fishers, delta, q, r, K, R_0, e_0, price, cost, noise,
    #   num_feedback, payoff_discount, num_steps
    n_fishers_length = 10
    n_fishers = n_fishers_length ** 2
    delta = 0
    q = 1
    r = 0.05
    K = 5000 / n_fishers
    R_0 = np.full(n_fishers, K/2)
    e_0 = np.linspace(0, r/q, num=n_fishers)
    price = 1
    cost = 0.5
    noise = 0.001
    e_msr = calculate_e_msr(n_fishers, q, r, K, price, cost)
    e_nash = calculate_e_nash(e_msr, n_fishers)
    print("e_msr: {}".format(e_msr))
    print("e_nash: {}".format(e_nash))
    num_feedback = 50
    payoff_discount = 0.5
    num_steps = 1000
    # Creating Simulation_arrays object
    my_sim = Simulation_arrays(n_fishers_length, delta, q, r, K, R_0, e_0, price, cost,
                        noise, num_feedback, payoff_discount, num_steps)
    my_sim.simulate()
    fig = plt.figure()
    plt.suptitle("Full fish movement")
    # Plotting resource levels vs. time
    ax1 = fig.add_subplot(2,2,1)
    for i in range(my_sim.n_fishers):
        ax1.plot(np.arange(num_steps), my_sim.R_data[i])
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Resource (K = {})".format(my_sim.K))
    ax1.set_title("Territory Resource Levels vs. Time")
    ax1.grid(True)
    # Plotting avg payoff vs. time
    U_avg = np.average(my_sim.U_data, axis=0)
    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(np.arange(num_steps), U_avg)
    ax2.set_xlabel("Time steps")
    ax2.set_ylabel("Average utility")
    ax2.set_title("Average Utility vs. Time")
    ax2.grid(True)
    # Plotting avg effort vs. time
    e_avg = np.average(my_sim.e_data, axis=0)
    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(np.arange(num_steps), e_avg)
    ax3.set_xlabel("Time steps")
    ax3.set_ylabel("Effort")
    ax3.set_title("Average Effort vs. Time")
    ax3.grid(True)
    # Plotting all efforts vs. time
    ax4 = fig.add_subplot(2,2,4)
    for i in range(my_sim.n_fishers):
        ax4.plot(my_sim.e_data[i])
    ax4.set_xlabel("Time steps")
    ax4.set_ylabel("Effort")
    ax4.set_title("Effort vs. Time")
    ax4.grid(True)
    fig.subplots_adjust(wspace=0.3, hspace=0.4)

    print("Last time step avg utility: {}".format(U_avg[-1]))
    print("Last time step avg effort: {}".format(e_avg[-1]))
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()    
    
if __name__ == "__main__":
    main()
