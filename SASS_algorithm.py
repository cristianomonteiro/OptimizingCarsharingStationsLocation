import networkx as nx
from time import time
import bz2
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

        self.nextSolution = None

def binarySearchSolution(solutions, valueSearched):
    left = 0
    #The -1 below is because right and left are about positions of the list
    right = len(solutions) - 1
    while left <= right:
        middle = (right + left) // 2
        #The +1 and -1 below are to ignore inclusive the current left or right, in the next iteration
        if solutions[middle].objectiveValue < valueSearched:
            left = middle + 1
 
        elif solutions[middle].objectiveValue > valueSearched:
            right = middle - 1
 
        else:
            return middle
 
    #valueSearched was not found. Then middle is set to where valueSearched is supposed to be inserted
    if solutions[middle].objectiveValue < valueSearched and middle < len(solutions):
        middle += 1
    #It is already known that solutions[middle].objectiveValue != valueSearched
    elif middle > 1 and solutions[middle - 1].objectiveValue > valueSearched:
        middle -= 1

    return middle

def optimize(edges, distanceDelete=500):
    feasibleSolution = Solution(objectiveValue=0, allocatedEdges=[], forbiddenEdges=set(), farestDistFromSe=0)
    solutions = [feasibleSolution]
    lastSolution = feasibleSolution

    count = 0
    nextPrint = 1
    while len(edges) > 0:
        count += 1
        if count == nextPrint:
            print(count, len(edges), len(solutions), solutions[-1].objectiveValue)
            nextPrint *= 2

        edge = edges.pop(0)

        bestSoFar = solutions[-1]
        hasBraked = False
        for current in reversed(solutions):
            if edge.idEdge not in current.forbiddenEdges:
                hasBraked = True
                break

        if not hasBraked:
            print("ERRO!!")

        newSolution = Solution( objectiveValue=current.objectiveValue + edge.utilityValue,
                                allocatedEdges=current.allocatedEdges + [edge.idEdge],
                                forbiddenEdges=current.forbiddenEdges.union(edge.edgesNearby),
                                farestDistFromSe=edge.distanceFromSe)

        lastSolution.nextSolution = newSolution
        lastSolution = newSolution

        posSolution = binarySearchSolution(solutions, newSolution.objectiveValue)
        solutions.insert(posSolution, newSolution)
        for i, s in enumerate(solutions):
            if i + 1 < len(solutions) and s.objectiveValue > solutions[i + 1].objectiveValue:
                print("ERRO!! BINARY SEARCH")

        #Cleaning unnecessary solutions
        #if len(solutions) >= 3:
        feasibleAndNecessary = feasibleSolution.nextSolution
        earliestDistance = feasibleAndNecessary.farestDistFromSe
        gapDistance = edge.distanceFromSe - earliestDistance

        if gapDistance > distanceDelete:
            #print("DELETE!")
            #posSolutionToKeep = binarySearchSolution(solutions, firstRealSolution.objectiveValue)
            #Set the solution found to be the new alwaysFeasibleInitialSolution
            #alwaysFeasibleInitialSolution = solutions[posSolutionToKeep]
            #Updates the list of solutions, removing the unnecessary ones
            #solutions = solutions[posSolutionToKeep:]

            feasibleSolution = feasibleAndNecessary

            #Checking and changing the nextSolution to a not unnecessary one
            while feasibleSolution.nextSolution != None and feasibleSolution.objectiveValue >= feasibleSolution.nextSolution.objectiveValue:
                if feasibleSolution.nextSolution.nextSolution != None:
                    feasibleSolution.nextSolution = feasibleSolution.nextSolution.nextSolution
                else:
                    break

            #It is always position 0 because the list is shrinking while elements are deleted
            try:
                while solutions[0] != feasibleSolution:
                    #solutions[0].nextSolution = None
                    del solutions[0]
                    #break
            except:
                print("ERROR DELETING!!")

    return solutions[-1]

precision = 0
filehandler = bz2.BZ2File('SASS_input_' + str(precision) + '.bz2', 'rb')
edges, distanceDelete = pickle.load(filehandler)

startTime = time()
optimize(edges, distanceDelete)
endTime = time()
print((endTime - startTime)/60, startTime, endTime)