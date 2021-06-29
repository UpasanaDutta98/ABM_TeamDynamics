import networkx as nx
import random
import numpy as np
from scipy.stats import skewnorm
import random
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Make sure networkx, numpy, scipy, and matplotlib is installed.

G = nx.fast_gnp_random_graph(n=20, p=0.2)
# This is going to be our network of interactions: A random network with 20 nodes and density = 0.2 (we can change this), meaning that the probability that an edge exists between any given pair of nodes = 20%
# I have not implemented the "emotional closeness" yet.

pos = nx.spring_layout(G)
nx.draw(G, pos) # Drawing the network

def initialize():
    initial_op = []
    credibility = []
    openness = []
    for i in G.nodes:
        G.nodes[i]['opinion'] = random.random()  # Opinion of an agent is a uniform random number between 0 to 1
        G.nodes[i]['initial_opinion'] = G.nodes[i]['opinion']  # Initial opinion
        G.nodes[i]['openness'] = skewnorm.rvs(a=-10, loc=0.9, scale=0.3, size=1)[0]  # Openness of an agent is chosen randomly from a left-skewed normal distribution
        G.nodes[i]['credibility'] =skewnorm.rvs(a=10, loc=0, scale=0.3, size=1)[0]  # credibility of an agent is chosen randomly from a right-skewed normal distribution
        initial_op.append(G.nodes[i]['initial_opinion'])
        credibility.append(G.nodes[i]['credibility'])
        openness.append(G.nodes[i]['openness'])
    
    #rescale the openness into [0,1]
    max_open = max(openness)
    min_open = min(openness)
    openness = []
    for i in G.nodes:
        G.nodes[i]['openness'] = (G.nodes[i]['openness'] - min_open )/ (max_open - min_open) #rescale the openness into [0,1]
        openness.append(G.nodes[i]['openness'])

    #rescale the credibility into [0,1]
    max_c = max(credibility)
    min_c = min(credibility)
    credibility = []
    for i in G.nodes:
        G.nodes[i]['credibility'] = (G.nodes[i]['credibility'] - min_c )/ (max_c - min_c)
        credibility.append(G.nodes[i]['credibility'])
    return initial_op, credibility, openness

# This function is not needed for now.
def observe():
    plt.cla()  # Clears the current plot
    nx.draw(G, cmap=plt.cm.Spectral, vmin=0, vmax=1,
            node_color=[G.nodes[i]['state'] for i in G.nodes],
            edge_cmap=plt.cm.binary, edge_vmin=0, edge_vmax=1,
            pos=G.pos)
    plt.show()


def update():
    for i in G.nodes: # Every node in the network gets a chance to update its opinion based on its own inital opinion, its openness, neighbours' opinions and neighbours credibility.
        Sum_neighbours_credibility = 0
        for j in G.neighbors(i):
            Sum_neighbours_credibility += G.nodes[j]['credibility']  # Obtaining sum of credibility of i's neighbours

        sigma_i = 0
        for j in G.neighbors(i):
            sigma_i += G.nodes[j]['credibility'] / Sum_neighbours_credibility * G.nodes[j][
                'opinion']  # This is obtaining the summation part of our update equation

        # Now using our full update equation, we will calculate the new opinion of node i
        open_i = G.nodes[i]['openness']
        initial_opi_i = G.nodes[i]['initial_opinion']
        G.nodes[i]['opinion'] = (1 - open_i) * initial_opi_i + open_i * sigma_i  # sigma_i was computed just above

def run_sumulation(total_timestep):
    average_opinion = []
    opinion_all = []
    time = []
    t = 0
    while t < total_timestep:
        n = G.number_of_nodes()
        avg_op = sum([G.nodes[i]['opinion'] for i in G.nodes()]) / n # Calculating average opinion of all agents after each update
        average_opinion.append(avg_op)
        time.append(t)
        opinion_all.append([G.nodes[i]['opinion'] for i in G.nodes()])
        t += 1
        update()

    return average_opinion, time, opinion_all


def plots():
    initial_op, credibility, openness =  initialize()

    average_opinion, time , opinion_all= run_sumulation(30)  # Get values over 30 timesteps
    plt.figure(figsize=(10, 5))
    plt.plot(time, opinion_all, markersize=4)
    plt.xlabel("Timestamp", fontsize=18, color="black")
    plt.ylabel("Opinion of individual members", fontsize=18, color="black")
    plt.xticks(fontsize=18, color="black")
    plt.yticks(fontsize=18, color="black")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(time, average_opinion, 'ro-', markersize=4)
    plt.xlabel("Timestamp", fontsize=18, color="black")
    plt.ylabel("Average opinion in team", fontsize=18, color="black")
    plt.xticks(fontsize=18, color="black")
    plt.yticks(fontsize=18, color="black")
    plt.show()

    final_opinion = opinion_all[29]
    shift = [abs(final_opinion[i] - initial_op[i]) for  i in range(20)]
    plt.figure()
    plt.scatter(credibility, shift, s = 100, color = "g")
    plt.xlabel("Credibility", fontsize=18, color="black")
    plt.ylabel("Absolute opinion shift", fontsize=18, color="black")
    plt.xticks(fontsize=18, color="black")
    plt.yticks(fontsize=18, color="black")
    plt.show()
    corr, _ = pearsonr(credibility, shift)
    print('Pearsons correlation credibility : %.3f' % corr)
    
    plt.figure()
    plt.scatter(openness, shift, s = 100, color = "b")
    plt.xlabel("Openness", fontsize=18, color="black")
    plt.ylabel("Absolute opinion shift", fontsize=18, color="black")
    plt.xticks(fontsize=18, color="black")
    plt.yticks(fontsize=18, color="black")
    plt.show()
    corr, _ = pearsonr(openness, shift)
    print('Pearsons correlation openness: %.3f' % corr)

    
    #degrees = [10 * G.degree[i] for i in G.nodes] #create balls around the points with radii prop to degree and openness
    area = [(200 * openness[i]) for i in range(20)]
    char = [area]
    color = ['tab:orange']
    labels = ['Openess']
    fig, ax = plt.subplots()

    for i in [0]:
        ax.scatter(credibility, shift, c=color[i], s=char[i], label=labels[i],
                   alpha=0.5, edgecolors='none')

    ax.legend(fontsize=15)
    plt.xlabel("Credibility", fontsize=18, color="black")
    plt.ylabel("Absolute opinion shift", fontsize=18, color="black")
    plt.xticks(fontsize=18, color="black")
    plt.yticks(fontsize=18, color="black")

    plt.show()


plots() 
