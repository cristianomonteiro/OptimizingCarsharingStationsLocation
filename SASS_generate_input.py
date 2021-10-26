import pandas as pd
import networkx as nx
from heapq import heapify, heapreplace#, heappop, heappush
import psycopg2 as pg
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

        #lengthiestEdge = sorted(G.edges(data=True), key=lambda x: x[2]['length'], reverse=True)[0]
        G = reBuildGraph(G, edgesHeap, firstSplit)
        #lengthiestEdge = sorted(G.edges(data=True), key=lambda x: x[2]['length'], reverse=True)[0]

    return G.to_undirected()

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

def generateInput(precisionInput=0, distanceCutOff=100):
    G = loadMultiGraphEdgesSplit(precision=precisionInput)

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

for precision in [0, 3]:
    data = generateInput(precisionInput=precision)

    #fileName = 'SASS_input_' + str(precision) + '.bz2'
    fileName = 'SASS_Sao_Caetano_Sul_input_' + str(precision) + '.bz2'
    filehandler = bz2.BZ2File(fileName, 'wb') 
    pickle.dump(data, filehandler)
    filehandler.close()
