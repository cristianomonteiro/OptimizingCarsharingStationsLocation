import pandas as pd
import networkx as nx
from heapq import heapify, heapreplace#, heappop, heappush
import gurobipy as gp
from gurobipy import GRB
from time import sleep
import psycopg2 as pg

def loadMultiGraph():
    params = {'host':'localhost', 'port':'5432', 'database':'afterqualifying', 'user':'cristiano', 'password':'cristiano'}
    conn = pg.connect(**params)

    sqlQuery = '''	select	EDGE.IDVERTEXORIG_FK,
                            EDGE.IDVERTEXDEST_FK,
                            EDGE.IDEDGE,
                            EDGE.LENGTH,
                            EDGE.UTILITYVALUE
                    from	STREETSEGMENT as EDGE
                    where   EDGE.UTILITYVALUE <> 0 '''
    dataFrameEdges = pd.read_sql_query(sqlQuery, conn)
    conn.close()

    G = nx.MultiGraph()
    for row in dataFrameEdges.itertuples():
        dictRow = row._asdict()
        
        G.add_edge(dictRow['idvertexdest_fk'], dictRow['idvertexorig_fk'],
                    key=str(dictRow['idedge']), idedge=str(dictRow['idedge']), length=dictRow['length'], utilityvalue=dictRow['utilityvalue'])

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

def loadMultiGraphEdgesSplit(precision=9, maxDistance=None):
    G = loadMultiGraph()

    if precision > 0:
        firstSplit = 2
        #The value must be negative because the data structure is a min heap
        edgesHeap = [(-1*data['length'], u, v, data['idedge'], data['length'], data['utilityvalue'], firstSplit) for u, v, data in G.edges(data=True)]
        heapify(edgesHeap)
    
        for i in range(round(len(edgesHeap) * precision)):
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

    return G

class Edge:
    def __init__(self, G, u, v, idEdge, utilityValue, distanceCutOff):
        self.idEdge = idEdge
        self.utilityValue = utilityValue
        self.variable = None

        self.edgesSet = set(self.reachableEdges(G, u, distanceCutOff))
        self.edgesSet.update(self.reachableEdges(G, v, distanceCutOff))

    #Return all id of edges adjacent to the current vertex
    def reachableEdges(self, G, u, cutoff):
        vertices = nx.single_source_dijkstra_path_length(G, u, cutoff=cutoff, weight='length').keys()

        edges = []
        for vertex in vertices:
            edges.extend([item[2] for item in G.edges(vertex, data='idedge')])

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
        edgeVariable = Edge.getVariable(model, varName)
        if edgeVariable == None:
            edgeVariable = model.addVar(name=varName, vtype=GRB.BINARY)
            #VTag is needed for finding the variable in the json file after optimizing the model
            edgeVariable.VTag = varName
        
        return edgeVariable

    def getOrCreateNeededVariables(self, model):
        self.variable = Edge.createOrGetEdgeVariable(model, self.idEdge)
        involvedVariables = [self.variable]
        involvedNames = []

        invertedCloneEdgeVariables = []
        invertedNames = []
        #The edge itself is also among the reached edges. In this loop, that edge must be avoided not to appear twice in the constraints
        for reachedIdEdge in self.edgesSet - {self.idEdge}:
            involvedName = self.idEdge + '-' + reachedIdEdge
            involvedNames.append(involvedName)
            involvedVariables.append(Edge.createOrGetEdgeVariable(model, involvedName))

            #Already assuring the inverted clone edge (graph is non-directed) for simplifying the loop in buildGurobiModel
            invertedName = reachedIdEdge + '-' + self.idEdge
            invertedNames.append(invertedName)
            invertedCloneEdgeVariables.append(Edge.createOrGetEdgeVariable(model, invertedName))

        return involvedVariables, involvedNames, invertedCloneEdgeVariables, invertedNames

def buildGurobiModel():
    G = loadMultiGraphEdgesSplit(precision=0)
    distanceCutOff = 500

    #Create a Gurobi Model
    model = gp.Model("SASS")

    count = 0
    nextPrint = 1
    objective = 0
    #Create the variables and define constraints
    for u, v, data in G.edges(data=True):
        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        edge = Edge(G, u, v, data['idedge'], data['utilityvalue'], distanceCutOff)

        involvedVariables, involvedNames, invertedCloneEdgeVariables, invertedNames = edge.getOrCreateNeededVariables(model)
        objective += edge.utilityValue * edge.variable

        #It is needed to update the model for accessing the VarName inside the following loops
        #model.update()
        for involvedVariable, variableName in zip(involvedVariables[1:], involvedNames):
            model.addConstr(edge.variable + involvedVariable <= 1, 'leq_1_' + edge.idEdge + '_' + variableName)
        
        for invertedCloneEdge, variableName in zip(invertedCloneEdgeVariables, invertedNames):
            model.addConstr(edge.variable == invertedCloneEdge, 'eq_' + edge.idEdge + '_' + variableName)

    #Set objective: maximize the utility value by allocating stations on edges
    model.setObjective(objective, GRB.MAXIMIZE)

    #First optimize() call will fail - need to set NonConvex to 2
    try:
        model.optimize()
    except gp.GurobiError:
        print("Optimize failed due to non-convexity")

    # Solve bilinear model
    model.params.NonConvex = 2
    model.optimize()

    model.printAttr('x')

    # Constrain 'x' to be integral and solve again
    #x.vType = GRB.INTEGER
    model.optimize()

    model.printAttr('x')

    return model

model = buildGurobiModel()
sleep(60)

