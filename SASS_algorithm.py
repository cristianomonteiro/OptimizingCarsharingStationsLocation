import networkx as nx
from time import time
import pickle 

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
            edges.extend([item[2] for item in G.edges(vertex, data='idedge')])

        return edges

class Solution:
    def __init__(self, objectiveValue, allocatedEdges, forbiddenEdges, farestDistFromSe):
        self.objectiveValue = objectiveValue
        self.allocatedEdges = allocatedEdges
        self.forbiddenEdges = forbiddenEdges
        self.farestDistFromSe = farestDistFromSe

def optimize(edges, distanceNearby=500):
    solutions = [Solution(objectiveValue=0, allocatedEdges=[], forbiddenEdges=set(), farestDistFromSe=0)]
    count = 0
    nextPrint = 1
    while len(edges) > 0:
        count += 1
        if count == nextPrint:
            print(count, len(edges), len(solutions), solutions[-1].objectiveValue)
            nextPrint *= 2

        edge = edges.pop(0)

        bestSoFar = solutions[-1]
        hasBraked = True
        for current in reversed(solutions):
            if edge.idEdge not in current.forbiddenEdges and current.objectiveValue + edge.utilityValue > bestSoFar.objectiveValue:
                hasBraked = False
                break

        if hasBraked and current != solutions[0]:
            print("ERRO!!")

        newBestSolution = Solution( objectiveValue=current.objectiveValue + edge.utilityValue,
                                    allocatedEdges=current.allocatedEdges + [edge.idEdge],
                                    forbiddenEdges=current.forbiddenEdges.union(edge.edgesNearby),
                                    farestDistFromSe=edge.distanceFromSe)
        solutions.append(newBestSolution) #BISECT TO KEEP THE OBJECTIVE VALUE OPTIMIZED

        i = 1
        while i < len(solutions):
            differenceDistance = edge.distanceFromSe - solutions[i].farestDistFromSe
            if differenceDistance > distanceNearby:
                del solutions[i]
            else:
                break
            i += 1
    
    return solutions[-1]

filehandler = open('SASS_input_0.data', 'rb') 
edges = pickle.load(filehandler)

startTime = time()
optimize(edges)
endTime = time()
print((endTime - startTime)/60, startTime, endTime)