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
    def __init__(self, objectiveValue, allocatedEdges, farestDistFromSe):
        self.objectiveValue = objectiveValue
        self.allocatedEdges = allocatedEdges
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

def optimize(edges, distanceDelete=200):
    feasibleSolution = Solution(objectiveValue=0, allocatedEdges=[], farestDistFromSe=0)
    solutions = [feasibleSolution]
    lastSolution = feasibleSolution

    while len(edges) > 0:
        edge = edges.pop(0)

        for current in reversed(solutions):
            hasBraked = False
            for allocated in reversed(current.allocatedEdges):
                if allocated in edge.edgesNearby:
                    hasBraked = True
                    break
            #A feasible solution is found if the "for" above was not braken
            if not hasBraked:
                break

        if hasBraked:
            print("ERROR!!!!!")

        newSolution = Solution( objectiveValue=current.objectiveValue + edge.utilityValue,
                                allocatedEdges=current.allocatedEdges + [edge.idEdge],
                                farestDistFromSe=edge.distanceFromSe)

        lastSolution.nextSolution = newSolution
        lastSolution = newSolution

        posSolution = binarySearchSolution(solutions, newSolution.objectiveValue)
        solutions.insert(posSolution, newSolution)

        #Cleaning unnecessary solutions
        feasibleAndNecessary = feasibleSolution.nextSolution
        earliestDistance = feasibleAndNecessary.farestDistFromSe
        gapDistance = edge.distanceFromSe - earliestDistance

        if gapDistance > distanceDelete:
            feasibleSolution = feasibleAndNecessary

            #Checking and changing the nextSolution to a not unnecessary one
            while feasibleSolution.nextSolution != None and feasibleSolution.objectiveValue >= feasibleSolution.nextSolution.objectiveValue:
                if feasibleSolution.nextSolution.nextSolution != None:
                    feasibleSolution.nextSolution = feasibleSolution.nextSolution.nextSolution
                else:
                    break

            #It is always position 0 because the list is shrinking while elements are deleted
            while solutions[0] != feasibleSolution:
                del solutions[0]

    return solutions[-1]

for precision in [8]:
    #fileName = 'SASS_input_' + str(precision) + '.bz2'
    fileName = 'SASS_Sao_Caetano_Sul_input_' + str(precision) + '.bz2'
    filehandler = bz2.BZ2File(fileName, 'rb')
    edges, distanceDelete = pickle.load(filehandler)
    filehandler.close()

    startTime = time()
    optimalSolution = optimize(edges, distanceDelete)
    endTime = time()
    print(optimalSolution.objectiveValue)
    print((endTime - startTime)/60, startTime, endTime)

    #fileName = 'SASS_output_' + str(precision) + '.bz2'
    fileName = 'SASS_Sao_Caetano_Sul_output_' + str(precision) + '.bz2'
    filehandler = bz2.BZ2File(fileName, 'wb') 
    pickle.dump(optimalSolution, filehandler)
    filehandler.close()