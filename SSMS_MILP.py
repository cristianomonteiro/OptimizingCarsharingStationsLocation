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
        keyAndIdEdge = str(dictRow['idvertexorig_fk']) + '-' + str(dictRow['idedge']) + '-' + str(dictRow['idvertexdest_fk'])

        G.add_edge(dictRow['idvertexorig_fk'], dictRow['idvertexdest_fk'], key=keyAndIdEdge,
                    idedge=keyAndIdEdge, length=dictRow['length'], utilityvalue=dictRow['utilityvalue'])

    print(G.number_of_edges(), G.number_of_nodes())

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
    #It must be a MultiDiGraph because besides it has multiple edges between the same nodes, networkx does not assure the order of edges.
    #Using a directed graph, the start node of an edge will always be the start node, avoiding errors in the reBuildGraph function.
    G = loadMultiDiGraph()

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

    return G.to_undirected()

class Edge:
    #Managing the created variables and constraints outside the model to avoid calling the expensive "model.update()"
    createdVariables = set()
    createdOmegaConstraints = set()

    def __init__(self, G, start, end, idEdge, length, utilityValue, distanceCutOff, model, createVariables):
        self.idEdge = idEdge
        self.start = start
        self.end = end
        self.length = length
        self.utilityValue = utilityValue

        if createVariables:
            self.distStart = nx.single_source_dijkstra_path_length(G, start, cutoff=distanceCutOff, weight='length')
            self.distEnd = nx.single_source_dijkstra_path_length(G, end, cutoff=distanceCutOff, weight='length')
            
            edgesStart, edgesAlphaStart = self.reachableEdges(G, self.distStart.keys(), distanceCutOff)
            edgesEnd, edgesAlphaEnd = self.reachableEdges(G, self.distEnd.keys(), distanceCutOff)

            self.edgesSetAlpha = set(edgesAlphaStart)
            self.edgesSetAlpha.update(edgesAlphaEnd)
            self.edgesSetAlpha = self.edgesSetAlpha - {self.idEdge}

            self.edgesSetOmega = set(edgesStart)
            self.edgesSetOmega.update(edgesEnd)
            self.edgesSetOmega = self.edgesSetOmega - self.edgesSetAlpha - {self.idEdge}

            self.variable, self.posVariable, self.alphaNames, self.alphaVars, self.posAlphaVars = self.getOrCreateNeededVariables(model, self.edgesSetAlpha, G)
            self.variable, self.posVariable, self.omegaNames, self.omegaVars, self.posOmegaVars = self.getOrCreateNeededVariables(model, self.edgesSetOmega, G)
        else:
            self.variable, self.posVariable = Edge.createOrGetEdgeVariable(model, self.idEdge, G)

    def checkGetShortestDistance(self, v1, v2):
        distances = []
        if v1 in self.distStart:
            distances.append(self.distStart[v1])
        if v2 in self.distStart:
            distances.append(self.distStart[v2])
        if v1 in self.distEnd:
            distances.append(self.distEnd[v1])
        if v2 in self.distEnd:
            distances.append(self.distEnd[v2])
        
        return min(distances)

    #Return all id of edges adjacent to the current vertex
    def reachableEdges(self, G, vertices, cutoff):
        edges = []
        edgesAlpha = []
        for vertex in vertices:
            edges.extend([item[2]['idedge'] for item in G.edges(vertex, data=True) if item[2]['utilityvalue'] != 0])

            for edge in G.edges(vertex, data=True):
                v1, v2, ddict = edge
                #cutoff is divided by 2 to avoid preventing edges in Omega (not too far) to also have stations
                if ddict['utilityvalue'] != 0 and self.length + self.checkGetShortestDistance(v1, v2) + ddict['length'] < cutoff/2:
                    edgesAlpha.append(ddict['idedge'])

        return edges, edgesAlpha
    
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

        #if variable == None:
        if not varName in Edge.createdVariables:
            Edge.createdVariables.add(varName)

            variable = model.addVar(name=varName, vtype=GRB.BINARY)
            positionVariable = model.addVar(lb=0.0, ub=lengthOmega, vtype=GRB.CONTINUOUS, name=positionName)
            #VTag is needed for finding the variable in the json file after optimizing the model
            variable.VTag = varName
            positionVariable.VTag = positionName
        else:
            variable = Edge.getVariable(model, varName)
            positionVariable = Edge.getVariable(model, positionName)

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

        #Avoid calling model.update() because it is slow. Updating createdVariables is faster
        #model.update()
        return variable, positionVariable, names, variables, posVariables

def defineConstraint(model, distCutOff, beginningLeft, distanceEdges, endingLeft, edgeVar, edgeLen, omegaVar, omegaLen, cnstrName):
    #rightHandSide = distCutOff - (edgeLen + distanceEdges + omegaLen) * (2 - edgeVar - omegaVar)
    rightHandSide = distCutOff - distCutOff * (2 - edgeVar - omegaVar)
    model.addConstr(beginningLeft + distanceEdges + endingLeft >= rightHandSide, cnstrName)

def printSolution(stations):
    G = loadMultiGraphEdgesSplit(500)
    for i, station in enumerate(stations.keys()):
        start, idEdgeOSM, end = Edge.splitEdgeName(station.VarName)

        stationName = 'station_' + i
        G.add_edge(start, stationName, key='in_' + stationName, length = stations[station])
        G.add_edge(stationName, end, key='out_' + stationName, length = G[start][end]['length'] - stations[station])
        G.remove_edge(start, end, key=idEdgeOSM)

    distances = nx.single_source_dijkstra_path_length('station_1', weight='length')
    for key, distance in distances:
        if key.startswith('station'):
            print(key, distance)

def buildGurobiModel(distanceCutOff=200):
    G = loadMultiGraphEdgesSplit(maxDistance=distanceCutOff)

    #Create a Gurobi Model
    model = gp.Model("SSMS")

    count = 0
    nextPrint = 1
    objective = 0
    #Create the variables and define the objective
    print("Creating the variables")
    for u, v, data in G.edges(data=True):
        if data['utilityvalue'] == 0:
            continue

        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        start, idEdgeOSM, end = Edge.splitEdgeName(data['idedge'])
        edge = Edge(G, start, end, data['idedge'], data['length'], data['utilityvalue'], distanceCutOff, model, False)
    
    model.update()
    count = 0
    nextPrint = 1
    objective = 0
    #Defining the constraints
    print("Defining the constraints")
    for u, v, data in G.edges(data=True):
        if data['utilityvalue'] == 0:
            continue

        count += 1
        if count == nextPrint:
            print(count)
            nextPrint *= 2

        start, idEdgeOSM, end = Edge.splitEdgeName(data['idedge'])

        edge = Edge(G, start, end, data['idedge'], data['length'], data['utilityvalue'], distanceCutOff, model, True)

        objective += edge.utilityValue * edge.variable

        model.addConstr(edge.variable + sum(edge.alphaVars) <= 1, 'alpha_' + edge.idEdge)
        
        for omegaName, omegaVariable, omegaPosVar in zip(edge.omegaNames, edge.omegaVars, edge.posOmegaVars):
            startOmega, idEdgeOmegaOSM, endOmega = Edge.splitEdgeName(omegaName)

            currentAndOmega = sorted([edge.idEdge, omegaName])
            currentAndOmega = currentAndOmega[0] + '|' + currentAndOmega[1]
            if currentAndOmega not in Edge.createdOmegaConstraints:
                Edge.createdOmegaConstraints.add(currentAndOmega)

                lengthOmega = G[startOmega][endOmega][omegaName]['length']

                if startOmega in edge.distEnd:
                    defineConstraint(model=model, distCutOff=distanceCutOff, beginningLeft=edge.length - edge.posVariable,
                                        distanceEdges=edge.distEnd[startOmega], endingLeft=omegaPosVar, edgeVar=edge.variable,
                                        edgeLen=edge.length, omegaVar=omegaVariable, omegaLen=lengthOmega,
                                        cnstrName=edge.idEdge + '-e-s-' + omegaName)

                if startOmega in edge.distStart:
                    defineConstraint(model=model, distCutOff=distanceCutOff, beginningLeft=edge.posVariable,
                                        distanceEdges=edge.distStart[startOmega], endingLeft=omegaPosVar, edgeVar=edge.variable,
                                        edgeLen=edge.length, omegaVar=omegaVariable, omegaLen=lengthOmega,
                                        cnstrName=edge.idEdge + '-s-s-' + omegaName)
                                        
                if endOmega in edge.distEnd:
                    defineConstraint(model=model, distCutOff=distanceCutOff, beginningLeft=edge.length - edge.posVariable,
                                        distanceEdges=edge.distEnd[endOmega], endingLeft=lengthOmega - omegaPosVar, edgeVar=edge.variable,
                                        edgeLen=edge.length, omegaVar=omegaVariable, omegaLen=lengthOmega,
                                        cnstrName=edge.idEdge + '-e-e-' + omegaName)

                if endOmega in edge.distStart:
                    defineConstraint(model=model, distCutOff=distanceCutOff, beginningLeft=edge.posVariable,
                                        distanceEdges=edge.distStart[endOmega], endingLeft=lengthOmega - omegaPosVar, edgeVar=edge.variable,
                                        edgeLen=edge.length, omegaVar=omegaVariable, omegaLen=lengthOmega,
                                        cnstrName=edge.idEdge + '-s-e-' + omegaName)
                
    #Set objective: maximize the utility value by allocating stations on edges
    model.setObjective(objective, GRB.MAXIMIZE)

    print("MODEL BUILT!!")
    return model

modelSSMS = None

#folderSaveModel = 'SSMS_Guarulhos'
folderSaveModel = 'SSMS_Sao_Caetano_Sul'
for MIPFocus in [2, 3, 0]:
    #Assure that the folder to save the results is created
    folderPath = pathlib.Path('./' + folderSaveModel + '/' + str(MIPFocus))
    folderPath.mkdir(parents=True, exist_ok=True)
    numRuns = 40
    #Discover the next number for filename
    for i in range(numRuns):
        fileName = folderPath / (str(i + 1) + '.json')
        if fileName.exists():
            continue
        elif modelSSMS is None:
            modelSSMS = buildGurobiModel()

        try:
            modelSSMS.Params.outputFlag = 0
            modelSSMS.Params.MIPFocus = MIPFocus
            modelSSMS.optimize()
            modelSSMS.write(str(fileName.resolve()))
            modelSSMS.reset(clearall=1)

            sleep(10)

        except gp.GurobiError as e:
            print("ERROR: " + str(e))
            break
