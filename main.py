import random
import pygraphviz as pgv
import networkx as nx
import csv
import numpy as np


def latentMatrix(data, variableNames):
    """ returns the latent variable matrix from observed data"""
    dim = len(data)
    latentVars = list()
    uniqueStructures = []
    latentMatrix = np.zeros((dim, 1), int)
    zeros = np.zeros((dim, 1), int)
    for i in range(dim):
        row = data[i].tolist()
        # Check if current row has been observed before
        if row not in uniqueStructures:
            # Make new latent variable
            latentVars.append('Latent'+str(len(latentVars)+1))
            uniqueStructures.append(row)
            if len(uniqueStructures) > 1:
                latentMatrix = np.hstack((latentMatrix, zeros))
            latentMatrix[i][len(uniqueStructures)-1] = 2
        # Check if relationship between current row and all rows above
        for j in range(i):
            anotherRow = data[j].tolist()
            if subStructure(row, anotherRow):
                if latentMatrix[j][uniqueStructures.index(row)] != 2:
                    # If row does not correspond to a latent variable, set entry to 1
                    latentMatrix[j][uniqueStructures.index(row)] = 1
            if subStructure(anotherRow, row):
                if latentMatrix[i][uniqueStructures.index(anotherRow)] != 2:
                    # If another row does not correspond to a latent variable, set entry to 1
                    latentMatrix[i][uniqueStructures.index(anotherRow)] = 1

    return np.hstack((data, latentMatrix)), variableNames + latentVars


def adjecencyMatrix(matrix):
    """ matrix is of the form [A | B], where A is the response matrix and B
     is the matrix corresponding to the latent variables"""
    # dim = number of variables
    dim = len(matrix[0])
    adjMatrix = np.zeros((dim, dim), int)
    for index in range(dim):
        column = matrix.T[index]

        nonzero = np.flatnonzero(column == 1)
        for el in nonzero:
            latentIndex = np.flatnonzero(matrix[el] == 2)
            if latentIndex:
                adjMatrix[index, latentIndex[0]] = 1

    return adjMatrix


def subStructure(sub, main):
    for i in range(len(main)):
        if sub[i] == 1 and sub[i] != main[i]:
            return False
    return True


def get_graph(matrix, labels):
    G = nx.from_numpy_array(matrix.T, create_using=nx.DiGraph)
    g = nx.transitive_reduction(G)
    mapping = dict(zip(g, labels))
    # ncenter = list(nx.topological_sort(g))
    g = nx.relabel_nodes(g, mapping)
    return g


def write_graph_dot(graph, filename):
    import os
    path = 'output/dot/'
    from networkx.drawing.nx_agraph import write_dot

    name = filename[5:]
    p = nx.nx_agraph.to_agraph(graph)
    p.graph_attr.update(rankdir='LR', splines='spline',
                        ranksep=0.5, nodesep=0.5, label='sample ' + name, labelloc='t')
    # p.node_attr.update(shape='circle', fixedsize=True, width=.4, label=' ')
    p.edge_attr.update(penwidth=3, color='orange')

    p.add_subgraph(header, rank='same')

    for node in p.nodes():
        n = p.get_node(node)

        if node in header:

            n.attr.update(color='darkgreen', style='filled', fontcolor='white')
        else:
            n.attr.update(color='royalblue3', penwidth=2)

    filename = path+filename + '.dot'

    if not os.path.exists(path):
        os.makedirs(path)

    p.write(filename)
    return p


def write_png(graph, filename):
    import os
    path = 'output/png/'

    filename = path+filename + '.png'

    if not os.path.exists(path):
        os.makedirs(path)
    graph.draw(filename, prog='dot')


def run(sample, header, filename):
    L, labels = latentMatrix(sample, header)
    A = adjecencyMatrix(L)
    G = get_graph(A, labels)
    p = write_graph_dot(G, filename)
    if png:
        write_png(p, filename)


""" Global variables
    header = list of variable names of the observed variables
    samples = list of response dichotomous matrices
"""


"""Input
    test_size = size of each sample
    png = boolean True = ouput pngs of graphs
"""

test_size = input('How many items per sample? (default =20)')
test_size = test_size if len(test_size) > 0 else 20
png = True if input('Do you want to save to pngs? (y/n)') == 'y' else False


def parse_csv():
    checkHeader = True
    header = list()
    samples = list()
    temp = list()
    counter = 1

    # Read file and store it in a array of arrays
    with open('data.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=';'))
    for row in data:
        del row[0]

        if checkHeader:
            header = row
            checkHeader = False
            continue
        temp.append(list(map(int, row)))
        if counter % test_size == 0:
            samples.append(np.asarray(temp, int))
            temp = []
        counter += 1

    return header, samples


header, samples = parse_csv()

print('Loaded ' + str(len(samples)))


for index, sample in enumerate(samples):
    filename = 'graph'+str(index)
    print('parsing sample: ' + str(index)+'...')
    run(sample, header, filename)
