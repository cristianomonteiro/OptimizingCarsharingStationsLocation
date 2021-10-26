import pandas as pd
import networkx as nx
from heapq import heapify, heapreplace#, heappop, heappush
import gurobipy as gp
from gurobipy import GRB
from time import sleep
import psycopg2 as pg
import pathlib
import bz2
import pickle

def loadMultiDiGraph():
    params = {'host':'localhost', 'port':'5432', 'database':'afterqualifying', 'user':'cristiano', 'password':'cristiano'}
    conn = pg.connect(**params)

    sqlQuery = '''	select	EDGE.IDVERTEXORIG_FK,
                            EDGE.IDVERTEXDEST_FK,
                            EDGE.IDEDGE,
                            EDGE.LENGTH,
                            EDGE.UTILITYVALUE
                    from	STREETSEGMENT as EDGE, COUNTY
                    where   COUNTY.DESCRIPTION = 'São Caetano do Sul' and
                            ST_Intersects(COUNTY.GEOM, EDGE.GEOM) '''
    dataFrameEdges = pd.read_sql_query(sqlQuery, conn)
    conn.close()

    G = nx.MultiDiGraph()
    for row in dataFrameEdges.itertuples():
        dictRow = row._asdict()
        
        G.add_edge(dictRow['idvertexorig_fk'], dictRow['idvertexdest_fk'],
                    key=str(dictRow['idedge']), idedge=str(dictRow['idedge']), length=dictRow['length'], utilityvalue=dictRow['utilityvalue'])

    print(G.number_of_edges(), G.number_of_nodes())

    return G

def reBuildGraph(G, edgesHeap, firstSplit):
    for item in edgesHeap:
        (heapValue, u, v, idedge, lengthOriginal, utilityValue, numSplit) = item
        #The number of segments the edge must be split into is 1 less the value stored in the heap
        numSplit = numSplit - 1
        if numSplit >= firstSplit:
            lengthSplitted = lengthOriginal/numSplit
            vertexStart = u

            G.remove_edge(u, v, key=idedge)
            for i in range(numSplit - 1):
                vertexEnd = idedge + '_' + str(i + 1)
                G.add_edge(vertexStart, vertexEnd, key=vertexEnd, idedge=vertexEnd, length=lengthSplitted, utilityvalue=utilityValue)
                vertexStart = vertexEnd
            keyLast = idedge + '_' + str(numSplit)
            G.add_edge(vertexStart, v, key=keyLast, idedge=keyLast, length=lengthSplitted, utilityvalue=utilityValue)

    return G

def loadMultiGraphEdgesSplit(nIterations=9, maxDistance=None):
    #It must be a MultiDiGraph because besides it has multiple edges between the same nodes, networkx does not assure the order of edges.
    #Using a directed graph, the start node of an edge will always be the start node, avoiding errors in the reBuildGraph function.
    G = loadMultiDiGraph()

    if nIterations > 0:
        firstSplit = 2
        #The value must be negative because the data structure is a min heap
        edgesHeap = [(-1*data['length'], u, v, data['idedge'], data['length'], data['utilityvalue'], firstSplit) for u, v, data in G.edges(data=True)]
        heapify(edgesHeap)
    
        for i in range(round(len(edgesHeap) * nIterations)):
            #The value must be multiplied by -1 because the data structure is a min heap
            if maxDistance != None and -1 * edgesHeap[0][0] <= maxDistance:
                break
            
            #(heapValue, u, v, idedge, lengthOriginal, utilityValue, numSplit) = heappop(edgesHeap)
            (heapValue, u, v, idedge, lengthOriginal, utilityValue, numSplit) = edgesHeap[0]
            #The value must be negative because the data structure is a min heap
            heapValue = -1 * lengthOriginal/numSplit
            #The numSplit is prepared for the next time the edge may be splitted (numsplit + 1)
            heapreplace(edgesHeap, (heapValue, u, v, idedge, lengthOriginal, utilityValue, numSplit + 1))

        G = reBuildGraph(G, edgesHeap, firstSplit)

    return G.to_undirected()

class Edge:
    def __init__(self, G, u, v, idEdge, utilityValue, distanceCutOff, createVariables):
        self.idEdge = idEdge
        self.u = u
        self.v = v
        self.utilityValue = utilityValue
        self.variable = None

        self.edgesSet = set(self.reachableEdges(G, u, distanceCutOff))
        self.edgesSet.update(self.reachableEdges(G, v, distanceCutOff))
        self.edgesSet = self.edgesSet - {self.idEdge}

    #Return all id of edges adjacent to the current vertex
    def reachableEdges(self, G, u, cutoff):
        vertices = nx.single_source_dijkstra_path_length(G, u, cutoff=cutoff, weight='length').keys()

        edges = []
        for vertex in vertices:
            edges.extend([item[2]['idedge'] for item in G.edges(vertex, data=True) if item[2]['utilityvalue'] != 0])

        return edges
    
    @staticmethod
    def getVariable(model, varName):
        variable = None
        try:
            variable = model.getVarByName(varName)
        finally:
            return variable

    @staticmethod
    def createOrGetEdgeVariable(model, varName):
        #if edgeVariable == None:
        if not varName in Edge.createdVariables:
            Edge.createdVariables.add(varName)

            edgeVariable = model.addVar(name=varName, vtype=GRB.BINARY)
            #edgeVariable = model.addVar(lb=0, ub=1, name=varName, vtype=GRB.CONTINUOUS)
            #VTag is needed for finding the variable in the json file after optimizing the model
            edgeVariable.VTag = varName
        else:
            edgeVariable = Edge.getVariable(model, varName)
        
        return edgeVariable

    def getOrCreateNeededVariables(self, model):
        self.variable = Edge.createOrGetEdgeVariable(model, 'main-' + self.idEdge + '-' + str(self.u) + '-' + str(self.v))
        involvedVariables = []
        invertedCloneEdgeVariables = []

        #The edge itself is also among the reached edges. In this loop, that edge must be avoided not to appear twice in the constraints
        for reachedIdEdge in self.edgesSet:
            involvedName = self.idEdge + '-' + reachedIdEdge
            involvedVariables.append(Edge.createOrGetEdgeVariable(model, 'clone-' + involvedName))

            #Already assuring the inverted clone edge (graph is non-directed) for simplifying the loop in buildGurobiModel
            invertedName = reachedIdEdge + '-' + self.idEdge
            invertedCloneEdgeVariables.append(Edge.createOrGetEdgeVariable(model, 'clone-' + invertedName))

        return involvedVariables, invertedCloneEdgeVariables

def buildGurobiModel(nIterations=0, distanceCutOff=100):
    #Managing the created variables and constraints outside the model to avoid calling the expensive "model.update()"
    Edge.createdVariables = set()

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

        edge = Edge(G, u, v, data['idedge'], data['utilityvalue'], distanceCutOff, True)

        involvedVariables, invertedCloneEdgeVariables = edge.getOrCreateNeededVariables(model)

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

        edge = Edge(G, u, v, data['idedge'], data['utilityvalue'], distanceCutOff, False)

        involvedVariables, invertedCloneEdgeVariables = edge.getOrCreateNeededVariables(model)

        objective += edge.utilityValue * edge.variable

        for involvedVariable in involvedVariables:
            model.addConstr(edge.variable + involvedVariable <= 1, 'leq_1_' + edge.idEdge + '_' + involvedVariable.VarName)
        
        for invertedCloneEdge in invertedCloneEdgeVariables:
            model.addConstr(edge.variable == invertedCloneEdge, 'eq_' + edge.idEdge + '_' + invertedCloneEdge.VarName)

        #for invertedCloneEdge in invertedCloneEdgeVariables:
        #    model.addConstr(edge.variable <= invertedCloneEdge, 'l_eq_' + edge.idEdge + '_' + invertedCloneEdge.VarName)
        #    model.addConstr(invertedCloneEdge <= edge.variable, 'g_eq_' + edge.idEdge + '_' + invertedCloneEdge.VarName)

    #Set objective: maximize the utility value by allocating stations on edges
    model.setObjective(objective, GRB.MAXIMIZE)

    countNames = 0
    for item in Edge.createdVariables:
        if item.startswith('main'):
            countNames += 1

    print("MODEL BUILT!!", len(model.getVars()), countNames)

    return model

nIterationsList = [0, 1, 2, 3]
modelSASS = None

folderSaveModel = 'SASS'
numRuns = 40
for nIter in nIterationsList:
    #Assure that the folder to save the results is created
    folderPath = pathlib.Path('./' + folderSaveModel + '/' + str(nIter))
    folderPath.mkdir(parents=True, exist_ok=True)
    #Discover the next number for filename
    for i in range(numRuns):
        fileName = folderPath / (str(i + 1) + '.json')
        if fileName.exists():
            continue
        elif modelSASS is None:
            modelSASS = buildGurobiModel(nIterations=nIter)

        try:
            modelSASS.Params.outputFlag = 0
            #modelSASS.Params.presolve = 0
            #modelSASS.Params.method = 2 #0: Primal Simplex    1: Dual Simplex   2: Barrier (Interior-points)
            modelSASS.optimize()
            modelSASS.write(str(fileName.resolve()))
            modelSASS.reset(clearall=1)

            sleep(10)

        except gp.GurobiError as e:
            print("ERROR: " + str(e))
            break
    
    modelSASS = None
