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
        keyAndIdEdge = str(dictRow['idvertexorig_fk']) + '-' + str(dictRow['idedge']) + '-' + str(dictRow['idvertexdest_fk'])

        G.add_edge(dictRow['idvertexorig_fk'], dictRow['idvertexdest_fk'], key=keyAndIdEdge,
                    idedge=keyAndIdEdge, length=dictRow['length'], utilityvalue=dictRow['utilityvalue'])

    return G

def reBuildGraph(G, edgesHeap, firstSplit):
    for item in edgesHeap:
        (heapValue, u, v, idedge, lengthOriginal, utilityValue, numSplit) = item
        #The number of segments the edge must be split into is 1 less the value stored in the heap
        numSplit = numSplit - 1
        if numSplit >= firstSplit:
            lengthSplitted = lengthOriginal/numSplit

            G.remove_edge(u, v, key=idedge)
            idEdgeOSM = idedge.split('-')[1]
            endStrName = vertexStart = str(u)
            for i in range(numSplit - 1):
                #Getting only the end vertex to build the name
                vertexEnd = endStrName + '_' + str(i + 1)
                keyAndIdEdge = vertexStart + '-' + idEdgeOSM + '-' + vertexEnd
                if not '_' in vertexStart:
                    vertexStart = int(vertexStart)
                G.add_edge(vertexStart, vertexEnd, key=keyAndIdEdge, idedge=keyAndIdEdge, length=lengthSplitted, utilityvalue=utilityValue)
                vertexStart = vertexEnd
            #Getting only the end vertex to build the name
            keyAndIdEdge = vertexStart + '-' + idEdgeOSM + '-' + str(v)
            G.add_edge(vertexStart, v, key=keyAndIdEdge, idedge=keyAndIdEdge, length=lengthSplitted, utilityvalue=utilityValue)

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
    def __init__(self, G, start, end, idEdge, length, utilityValue, distanceCutOff, model):
        self.idEdge = idEdge
        self.start = start
        self.end = end
        self.length = length
        self.utilityValue = utilityValue

        edgesStart, edgesAlphaStart, self.distStart = self.reachableEdges(G, start, distanceCutOff)
        edgesEnd, edgesAlphaEnd, self.distEnd = self.reachableEdges(G, end, distanceCutOff)

        self.edgesSetAlpha = set(edgesAlphaStart)
        self.edgesSetAlpha.update(edgesAlphaEnd)
        self.edgesSetAlpha = self.edgesSetAlpha - {self.idEdge}

        self.edgesSetOmega = set(edgesStart)
        self.edgesSetOmega.update(edgesEnd)
        self.edgesSetOmega = self.edgesSetOmega - self.edgesSetAlpha - {self.idEdge}

        self.variable, self.posVariable, self.alphaNames, self.alphaVars, self.posAlphaVars = self.getOrCreateNeededVariables(model, self.edgesSetAlpha, G)
        self.variable, self.posVariable, self.omegaNames, self.omegaVars, self.posOmegaVars = self.getOrCreateNeededVariables(model, self.edgesSetOmega, G)

    #Return all id of edges adjacent to the current vertex
    def reachableEdges(self, G, u, cutoff):
        distances = nx.single_source_dijkstra_path_length(G, u, cutoff=cutoff, weight='length')
        vertices = distances.keys()

        edges = []
        edgesAlpha = []
        for vertex in vertices:
            edges.extend([item[2] for item in G.edges(vertex, data='idedge')])

            for edge in G.edges(vertex, data=True):
                v1, v2, ddict = edge
                if self.length + distances[vertex] + ddict['length'] < cutoff:
                    edgesAlpha.append(ddict['idedge'])

        return edges, edgesAlpha, distances
    
    @staticmethod
    def splitEdgeName(name):
        start, idEdgeOSM, end = name.split('-')

        if not '_' in start:
            start = int(start)
        
        if not '_' in end:
            end = int(end)
        
        return start, idEdgeOSM, end

    @staticmethod
    def getVariable(model, varName):
        variable = None
        try:
            variable = model.getVarByName(varName)
        finally:
            return variable

    @staticmethod
    def createOrGetEdgeVariable(model, varName, G):
        positionName = 'pos-' + varName

        startOmega, idEdgeOmegaOSM, endOmega = Edge.splitEdgeName(varName)
        lengthOmega = G[startOmega][endOmega][varName]['length']

        variable = Edge.getVariable(model, varName)
        positionVariable = Edge.getVariable(model, positionName)
        if variable == None:
            variable = model.addVar(name=varName, vtype=GRB.BINARY)
            positionVariable = model.addVar(lb=0.0, ub=lengthOmega, vtype=GRB.CONTINUOUS, name=positionName)
            #VTag is needed for finding the variable in the json file after optimizing the model
            variable.VTag = varName
            variable.VTag = positionName
        
        return variable, positionVariable

    def getOrCreateNeededVariables(self, model, setEdges, G):
        variable, positionVariable = Edge.createOrGetEdgeVariable(model, self.idEdge, G)

        names = []
        variables = []
        posVariables = []
        for reachedIdEdge in setEdges:
            names.append(reachedIdEdge)
            reachedVariable, reachedPositionVariable = Edge.createOrGetEdgeVariable(model, reachedIdEdge, G)
            variables.append(reachedVariable)
            posVariables.append(reachedPositionVariable)

        return variable, positionVariable, names, variables, posVariables

def buildGurobiModel(distanceCutOff=500):
    G = loadMultiGraphEdgesSplit(maxDistance=distanceCutOff)

    #Create a Gurobi Model
    model = gp.Model("SSMS")

    count = 0
    nextPrint = 1
    objective = 0
    #Create the variables and define constraints
    for u, v, data in G.edges(data=True):
        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        start, idEdgeOSM, end = Edge.splitEdgeName(data['idedge'])

        edge = Edge(G, start, end, data['idedge'], data['length'], data['utilityvalue'], distanceCutOff, model)

        objective += edge.utilityValue * edge.variable

        model.addConstr(edge.variable + sum(edge.alphaVars) <= 1, 'alpha_' + edge.idEdge)

        for omegaName, omegaVariable, omegaPosVar in zip(edge.omegaNames, edge.omegaVars, edge.posOmegaVars):
            startOmega, idEdgeOmegaOSM, endOmega = Edge.splitEdgeName(omegaName)

            lengthOmega = G[startOmega][endOmega][omegaName]['length']
            rightHandSide = distanceCutOff - (edge.length + lengthOmega) * (2 - edge.variable - omegaVariable)
            
            if startOmega in edge.distEnd:
                model.addConstr(edge.posVariable + edge.distEnd[startOmega] + omegaPosVar <= rightHandSide, edge.idEdge + '-e-s-' + omegaName)
            if startOmega in edge.distStart:
                model.addConstr(edge.length - edge.posVariable + edge.distStart[startOmega] + omegaPosVar <= rightHandSide, edge.idEdge + '-s-s-' + omegaName)
            if endOmega in edge.distEnd:
                model.addConstr(edge.posVariable + edge.distEnd[endOmega] + lengthOmega - omegaPosVar <= rightHandSide, edge.idEdge + '-e-e-' + omegaName)
            if endOmega in edge.distStart:
                model.addConstr(edge.length - edge.posVariable + edge.distStart[endOmega] + lengthOmega - omegaPosVar <= rightHandSide, edge.idEdge + '-s-e-' + omegaName)
                
    #Set objective: maximize the utility value by allocating stations on edges
    model.setObjective(objective, GRB.MAXIMIZE)

    return model

model = buildGurobiModel()

try:
    model.optimize()
except gp.GurobiError:
    print("Optimize failed due to non-convexity")

sleep(60)

