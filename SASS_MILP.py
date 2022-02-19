import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from time import sleep
import pathlib
import bz2
import pickle

from loadSplitEdges import loadMultiGraphEdgesSplit

class Edge:
    def __init__(self, G, u, v, idEdge, utilityValue, distanceCutOff, variablesAlreadyCreated, model):
        self.idEdge = idEdge
        self.u = u
        self.v = v
        self.utilityValue = utilityValue
        self.tagVariable = self.idEdge + '-' + str(self.u) + '-' + str(self.v)
        self.variable = Edge.createOrGetEdgeVariable(model, self.idEdge, self.tagVariable)

        self.omegaEdgesSet, self.alphaEdgesSet = self.reachableEdges(G, u, distanceCutOff)
        edgesSetFromV, alphaEdgesSetFromV = self.reachableEdges(G, v, distanceCutOff)
        self.omegaEdgesSet.update(edgesSetFromV)
        self.alphaEdgesSet.update(alphaEdgesSetFromV)

        self.omegaEdgesSet = self.omegaEdgesSet - self.alphaEdgesSet - {self.idEdge}
        self.alphaEdgesSet = self.alphaEdgesSet - {self.idEdge}

        if variablesAlreadyCreated:
            self.omegaVariables, self.alphaVariables = self.getNeededVariables(model)


    #Return all id of edges adjacent to the current vertex
    def reachableEdges(self, G, u, cutoff):
        distances = nx.single_source_dijkstra_path_length(G, u, cutoff=cutoff, weight='length')
        vertices = distances.keys()

        edges = []
        edgesAlpha = []
        for vertex in vertices:
            for item in G.edges(vertex, data=True):
                if item[2]['utilityvalue'] != 0:
                    edges.append(item[2]['idedge'])

                    #cutoff is divided by 2 to allow edges not too far in Omega to also have stations
                    if distances[vertex] < cutoff/2:
                        edgesAlpha.append(item[2]['idedge'])

        return set(edges), set(edgesAlpha)
    
    @staticmethod
    def getVariable(model, varName):
        variable = None
        try:
            variable = model.getVarByName(varName)
        finally:
            return variable

    @staticmethod
    def createOrGetEdgeVariable(model, varName, tagVariable=None):
        #if edgeVariable == None:
        if not varName in Edge.createdVariables:
            Edge.createdVariables.add(varName)

            edgeVariable = model.addVar(name=varName, vtype=GRB.BINARY)
            #VTag is needed for finding the variable in the json file after optimizing the model
            edgeVariable.VTag = tagVariable
        else:
            edgeVariable = Edge.getVariable(model, varName)
        
        return edgeVariable

    def getNeededVariables(self, model):
        #self.variable = Edge.createOrGetEdgeVariable(model, self.idEdge)
        alphaVariables = []
        omegaVariables = []
        
        for reachedIdEdge in self.alphaEdgesSet:
            alphaVariables.append(Edge.createOrGetEdgeVariable(model, reachedIdEdge))

        for reachedIdEdge in self.omegaEdgesSet:
            omegaVariables.append(Edge.createOrGetEdgeVariable(model, reachedIdEdge))

        return omegaVariables, alphaVariables

def buildGurobiModel(nIterations=0, distanceCutOff=200):
    #Managing the created variables and constraints outside the model to avoid calling the expensive "model.update()"
    Edge.createdVariables = set()
    Edge.createdOmegaConstraints = set()

    G = loadMultiGraphEdgesSplit(nIterations=nIterations)

    #Create a Gurobi Model
    model = gp.Model("SASS")

    count = 0
    nextPrint = 1
    objective = 0
    #Create the variables
    print("Creating the variables")
    for u, v, data in G.edges(data=True):
        if data['utilityvalue'] == 0:
            continue

        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        edge = Edge(G, u, v, data['idedge'], data['utilityvalue'], distanceCutOff, False, model)

    model.update()
    count = 0
    nextPrint = 1
    objective = 0
    #Define the constraints
    print("Defining the constraints", len(model.getVars()), len(Edge.createdVariables))
    for u, v, data in G.edges(data=True):
        if data['utilityvalue'] == 0:
            continue

        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        edge = Edge(G, u, v, data['idedge'], data['utilityvalue'], distanceCutOff, True, model)

        objective += edge.utilityValue * edge.variable

        model.addConstr(edge.variable + sum(edge.alphaVariables) <= 1, 'alpha_' + edge.idEdge)
        for involvedVariable in edge.omegaVariables:
            currentAndOmega = sorted([edge.idEdge, involvedVariable.VarName])
            currentAndOmega = currentAndOmega[0] + '|' + currentAndOmega[1]
            if currentAndOmega not in Edge.createdOmegaConstraints:
                Edge.createdOmegaConstraints.add(currentAndOmega)
                model.addConstr(edge.variable + involvedVariable <= 1, 'leq_1_' + edge.idEdge + '_' + involvedVariable.VarName)
        
    #Set objective: maximize the utility value by allocating stations on edges
    model.setObjective(objective, GRB.MAXIMIZE)

    print("MODEL BUILT!!", len(model.getVars()), len(Edge.createdVariables))

    return model

nIterationsList = [0, 1, 4, 9]
modelSASS = None

folderSaveModel = 'SASS_1_Thread'
numRuns = 1 
for nIter in nIterationsList:
    #Assure that the folder to save the results is created
    folderPath = pathlib.Path('./' + folderSaveModel + '/' + str(nIter))
    folderPath.mkdir(parents=True, exist_ok=True)
    #Discover the next number for filename
    for i in range(numRuns):
        #fileName = folderPath / (str(i + 1) + '.json')
        fileName = folderPath / 'model.mps'
        if fileName.exists():
            continue
        elif modelSASS is None:
            modelSASS = buildGurobiModel(nIterations=nIter)

        try:
            modelSASS.Params.outputFlag = 0
            modelSASS.Params.Threads = 1
            #modelSASS.optimize()
            modelSASS.write(str(fileName.resolve()))
            modelSASS.reset(clearall=1)

            sleep(10)

        except gp.GurobiError as e:
            print("ERROR: " + str(e))
            break
    
    modelSASS = None
