import networkx as nx
import psycopg2 as pg
import bz2
import pickle 

from loadSplitEdges import loadMultiGraphEdgesSplit

class Edge:
    def __init__(self, G, u, v, idEdge, utilityValue, distanceFromSe, distanceCutOff):
        self.idEdge = idEdge
        self.utilityValue = utilityValue
        self.distanceFromSe = distanceFromSe

        self.edgesNearby = set(self.reachableEdges(G, u, distanceCutOff))
        self.edgesNearby.update(self.reachableEdges(G, v, distanceCutOff))

    #Return all id of edges adjacent to the current vertex
    def reachableEdges(self, G, u, cutoff):
        vertices = nx.single_source_dijkstra_path_length(G, u, cutoff=cutoff, weight='length').keys()

        edges = []
        for vertex in vertices:
            edges.extend([item[2]['idedge'] for item in G.edges(vertex, data=True) if item[2]['utilityvalue'] != 0])

        return edges

class Solution:
    def __init__(self, objectiveValue, allocatedEdges, forbiddenEdges, farestDistFromSe):
        self.objectiveValue = objectiveValue
        self.allocatedEdges = allocatedEdges
        self.forbiddenEdges = forbiddenEdges
        self.farestDistFromSe = farestDistFromSe

def generateInput(precisionInput=0, distanceCutOff=200):
    G = loadMultiGraphEdgesSplit(nIterations=precisionInput)

    pracaDaSe = 1407132173 #1837923352 #60641211      #26129121 is in Guarulhos       #1407132173 is in São Caetano do Sul
    distances = nx.single_source_dijkstra_path_length(G, pracaDaSe, weight='length')
    distances = sorted(distances.items(), key=lambda item: item[1])
    otherComponents = sorted(nx.connected_components(G), key=len, reverse=True)[1:]

    distanceToOtherComponent = float('inf')
    for component in otherComponents:
        for vertex in component:
            distances.append((vertex, distanceToOtherComponent))

    count = 0
    nextPrint = 1
    edgesSet = set()
    edges = []
    maxEdgeLength = 0
    for key, valueDistance in distances:
        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        for u, v, data in G.edges(key, data=True):
            if data['utilityvalue'] != 0 and data['idedge'] not in edgesSet:
                edgesSet.add(data['idedge'])
                edges.append(Edge(G, u, v, data['idedge'], data['utilityvalue'], valueDistance, distanceCutOff))

                if data['length'] > maxEdgeLength:
                    maxEdgeLength = data['length']

    return edges, distanceCutOff + maxEdgeLength

for precision in [8]:
    data = generateInput(precisionInput=precision)

    #fileName = 'SASS_input_' + str(precision) + '.bz2'
    fileName = 'SASS_Sao_Caetano_Sul_input_' + str(precision) + '.bz2'
    filehandler = bz2.BZ2File(fileName, 'wb') 
    pickle.dump(data, filehandler)
    filehandler.close()
