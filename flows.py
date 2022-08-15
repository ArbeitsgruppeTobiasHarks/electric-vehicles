from __future__ import annotations

from typing import Dict, List
from network import *
from utilities import *
from networkLoadingEvents import *

import json

# A flow on a single edge consisting of in- and outflow rate for every commodity up to a certain time
# Also provides the aggregated flow rates, queue length, travel time
class EdgeFlow:
    # The edge this flow is defined on
    _e: Edge
    # The edge inflow rates (commodity specific)
    _fPlus: Dict[int, PWConst]
    # The aggregated inflow rate
    _fPlusA : PWConst
    # The edge outflow rates (commodity specific)
    _fMinus: Dict[(Edge, int), PWConst]
    # The aggregated outflow rate
    _fMinusA: PWConst
    # The queue
    _queue: PWLin
    # The number of commodities
    _noOfCommodities: int
    # up to what time is the inflow determined
    _upTo: number

    def __init__(self, e: Edge, numberOfCommodities: int, fPlus: List[PWConst] = None):
        self._e = e
        self._noOfCommodities = numberOfCommodities

        # initialize with the zero flow up to time zero
        # TODO: Maybe start at -infinity?
        self._queue = PWLin([zero, zero], [zero], [zero])
        self._fPlusA = PWConst([zero, zero], [zero])
        self._fMinusA = PWConst([zero, zero], [zero])
        for i in range(numberOfCommodities):
            self._fPlus[i] = PWConst([zero, zero], [zero])
            self._fMinus[i] = PWConst([zero, e.tau], [zero])
        self._upTo = zero

        # TODO: Maybe also allow to provide fMinus and then accept this without check?
        # TODO: Currently we assume that fPlus is given up to infinity (at least via a default value)
        if not(fPlus is None):
            assert(len(fPlus) == numberOfCommodities)
            theta = zero
            while theta < infinity:
                nextTheta = infinity
                rates = []
                for i in range(numberOfCommodities):
                    rates.append(fPlus[i].getValueAt(theta))
                    nextTheta = min(nextTheta, fPlus[i].getNextStepFrom(theta))
                self.extendInflow(nextTheta-theta, rates)
                theta = nextTheta




    # Extend the edge-flow for an interval of length alpha with the given inflowRates
    # inflowRates must contain one non-negative inflow rate for every commodity
    # and return events for the target node whenever the newly determined outflow rate changes
    # (one event for the end of the extension phase and possibly another one for when the queue runs empty)
    def extendInflow(self, alpha: number, inflowRates: List[number]) -> List[Event]:
        assert(len(inflowRates) == self._noOfCommodities)
        if alpha < numPrecision:
            return []

        fPlusA = zero
        for i in range(self._noOfCommodities):
            self._fPlus[i].addSegmentRel(alpha, inflowRates[i])
            fPlusA += inflowRates[i]

        self._fPlusA.addSegmentRel(alpha, fPlusA)

        # In order to determine the outflow rates from the inflow rates we have to distinguish to cases:
        # a) The queue is non-empty at the time of the inflow
        # b) The queue is empty at the time of the inflow
        # If both cases happen during the extension period, they happen in this order
        # (ignoring the start point of the extension period)

        # Thus, we first compute the point at which the queue runs empty (or the extension phase ends)...
        qeAtStart = self._queue.getValueAt(self._upTo)
        if fPlusA >= self._e.nu - numPrecision:
            x = alpha
        elif not(isZero(qeAtStart)):
            x = qeAtStart / (self._e.nu - fPlusA)
        else:
            x = zero

        # .. and then calculate the outflow rates for the interval before and after this point separately
        # a) while the queue is non-empty flow leaves at an (aggregated) rate of exactly nu_e
        #    Thus, we can convert the inflow rates to outflow rates by dividing by the following ratio
        # TODO: If the inflowRate is zero we divide by zero here - but we also only extend for an interval of length zero
        #       => So, maybe this is fine?
        kappa = fPlusA / self._e.nu
        for i in range(self._noOfCommodities):
            self._fMinus[i].addSegmentRel(x * kappa, inflowRates[i] / kappa)

        self._fMinusA.addSegmentRel(x * kappa, fPlusA / kappa)

        # b) while the queue is empty flow leaves at the same rate as it entered
        # TODO: We repeat the same code twice here - can we make this cleaner?
        for i in range(self._noOfCommodities):
            self._fMinus[i].addSegmentRel(max(zero,alpha-x), inflowRates[i])

        self._fMinusA.addSegmentRel(max(zero,alpha-x), fPlusA)

        self._queue.addSegmant(self._upTo+x,fPlusA - self._e.nu)
        self._queue.addSegmant(self._upTo+alpha,zero,zero)

        events = []
        if zero < x < alpha:
            events.append(Event(self._upTo+x,self._e.node_to,"queue on edge " + str(self._e) + " depleted"))
        self._upTo += alpha
        events.append(Event(self._upTo,self._e.node_to,"change in outflow rate due to change in inflow rate"))

        return events

    def getNextOutflowStepFrom(self, theta: number) -> number:
        assert(zero <= theta < self.T(self._upTo))

        nextTheta = infinity
        for i in range(self._noOfCommodities):
            nextTheta = min(nextTheta, self._fMinus[i].getNextStepFrom(theta))
        return nextTheta

    def getOutflowAt(self, theta: number, i : Union[int,None] = None):
        assert (zero <= theta < self.T(self._upTo))

        if i is None:
            return self._fMinusA.getValueAt(theta)
        else:
            return self._fMinus[i].getValueAt(theta)

    def getFplus(self, i : int = None) -> PWConst:
        if i is None:
            return self._fPlusA
        else:
            return self._fPlus[i]

    def getFminus(self, i : int = None) -> PWConst:
        if i is None:
            return self._fMinusA
        else:
            return self._fMinus[i]


    # The travel time over the edge at time theta
    def c(self, theta:number) -> number:
        # the flow has to be determined at least up to time theta
            assert (self._upTo >= theta)
            return self._queue.getValueAt(theta) / self._e.nu + self._e.tau

    # The arrival time at the end of edge e if entering at time theta
    def T(self, theta:number) -> number:
        return theta + self.c(theta)

    # TODO: We need functions to represent the edge flow as string and convert it to json
    # TODO: Since the flow rates are now private we probably also need functions to return those - or at least their
    #       values?
    # TODO: Should we include the feasibility tests? Or are they unnecessary now that the class itself ensures
    #       that they are guaranteed?



# A commodity is defined by a start node, a sink node and a network inflow-rate
class Commodity:
    # The source node
    _source: Node
    # The sink node
    _sink: Node
    # The network inflow rate (at the source)
    _u: PWConst

    def __init__(self, s: Node, t: Node, u: PWConst):
        self._source = s
        self._sink = t
        self._u = u

    def getS(self) -> Node:
        return self._source

    def getT(self) -> Node:
        return self._sink

    def getU(self) -> PWConst:
        return self._u

# A commodity with only a single path
class CommoditySP(Commodity):
    # The commodity's path
    _path : Path

    def __init__(self, path: Path, u: PWConst):
        self._path = path
        super().__init__(path.getStart(), path.getEnd(), u)

    def getPath(self) -> Path:
        return self._path



# A partial feasible flow with in-/outflow rates for every edges and commodity and queues for every edge
# Feasibility is not checked automatically but a check can be initiated by calling .checkFeasibility
class Flow:
    # The network the flow lives in:
    _network: Network
    # Stores for every node until which time the flow has been calculated:
    _upToAt: Dict[Node, number]
    # The edge flows for all edges
    _f: Dict[Edge, EdgeFlow]
    # The sources (one for each commodity)
    _sources: List[Node]
    # The sinks (one for each commodity)
    _sinks: List[Node]
    # The network inflow rates (one for each commodity)
    _u: List[PWConst]
    # The number of commodities
    _noOfCommodities: int

    def __init__(self, network: Network, commodities: List[Commodity], f: Dict[Edge, EdgeFlow] = {}):
        self._network = network
        self._noOfCommodities = len(commodities)

        # Initialize functions f^+,f^- and q for every edge e
        # If no edge flows are given for any edge we initialize with the zero-flow
        self._f = {}
        for e in network.edges:
            if e in f:
                self._f[e] = f[e]
            else:
                self._f[e] = EdgeFlow(e,self._noOfCommodities)

        # The zero-flow up to time zero TODO: Update this
        self._upToAt = {}
        for v in network.nodes:
            self._upToAt[v] = zero

        self._u = []
        self._sources = []
        self._sinks = []
        # Maybe just save the commodity objects instead?
        for commodity in commodities:
            self._u.append(commodity.getU())
            self._sources.append(commodity.getS())
            self._sinks.append(commodity.getT())

    # Determines the arrival time at the end of path p when starting at time theta
    def pathArrivalTime(self, p: Path, theta: number) -> number:
        if len(p) == 0:
            return theta
        else:
            firstEdge = p.edges[0]
            return self.pathArrivalTime(Path(p.edges[1:], firstEdge.node_to), self._f[firstEdge].T(theta))


# A partial flow in which every commodity only has a single path
class FlowSP:
    # The network the flow lives in:
    _network: Network
    # Stores for every node until which time the flow has been calculated:
    _upToAt: Dict[Node, number]
    # The edge flows for all edges
    _f: Dict[Edge, EdgeFlow]
    # The sources (one for each commodity)
    _sources: List[Node]
    # The sinks (one for each commodity)
    _sinks: List[Node]
    # The network inflow rates (one for each commodity)
    _u: List[PWConst]
    # The number of commodities
    _noOfCommodities: int
    # The path for every commodity
    _paths : List[Path]
    # A dictionary containing for every edge e a list of tuples (i, k) such that e is the k-th edge on the path
    # of commodity i (where the first edge has number 0)
    _edgeContainedIn : Dict[Edge, List[Tuple[int, int]]]

    def __init__(self, network: Network, commodities: List[CommoditySP]):

        self._noOfCommodities = len(commodities)
        self._network = network
        # The zero-flow up to time zero
        self._upToAt = {}
        for v in network.nodes:
            self._upToAt[v] = zero
        self._paths = []
        self._edgeContainedIn = {e : [] for e in network.edges}

        i = 0
        for commodity in commodities:
            path = commodity.getPath()
            self._paths.append(path)
            self._u.append(commodity.getU())
            self._sources.append(commodity.getS())
            self._sinks.append(commodity.getT())
            k = 0
            for e in path.edges:
                self._edgeContainedIn[e].append((i, k))
                k += 1
            i += 1

        self._f = {}
        for e in network.edges:
            self._f[e] = EdgeFlow(e, len(self._edgeContainedIn[e]))

    # Extends the flow at node v
    def extendAt(self, v: Node) -> List[Event]:
        theta = self._upToAt[v]
        nextTheta = infinity
        for e in v.incoming_edges:
            nextTheta = min(nextTheta,self._f[e].getNextOutflowStepFrom(theta))
        for i in range(self._noOfCommodities):
            if self._sources[i] == v:
                nextTheta = min(nextTheta,self._u[i].getNextStepFrom(theta))

        newEvents = []
        for e in v.outgoing_edges:
            rates = []
            j = 0
            for (i,k) in self._edgeContainedIn[e]:
                if k == 0:
                    rates.append(self._u[i].getValueAt(theta))
                else:
                    rates.append(self._f[self._paths[i].edges[k-1]].getOutflowAt(theta, j))
                j += 1
            newEvents.extend(self._f[e].extendInflow(nextTheta-theta, rates))

        self._upToAt[v] = nextTheta
        return newEvents


    def toFlow(self) -> Flow:
        f = {}
        for e in self._network.edges:
            edgeFlowP = [PWConst([zero],[],zero) for _ in range(self._noOfCommodities)]
            j = 0
            for (i,_) in self._edgeContainedIn[e]:
                edgeFlowP[i] += self._f[e].getFplus(j)
            f[e] = EdgeFlow(e, self._noOfCommodities, edgeFlowP)

        commodities = []
        for i in range(self._noOfCommodities):
            commodities.append(Commodity(self._sources[i], self._sinks[i], self._u[i]))

        return Flow(self._network, commodities, f)


# A partial feasible flow with in-/outflow rates for every edges and commodity and queues for every edge
# Feasibility is not checked automatically but a check can be initiated by calling .checkFeasibility
class PartialFlow:
    # The network the flow lives in:
    _network: Network
    # Stores for every node until which time the flow has been calculated:
    _upToAt: Dict[Node, number]
    # The edge inflow rates (commodity specific)
    _fPlus: Dict[(Edge, int), PWConst]
    # The edge outflow rates (commodity specific)
    _fMinus: Dict[(Edge, int), PWConst]
    # The queues
    _queues: Dict[Edge, PWLin]
    # The sources (one for each commodity)
    _sources: List[Node]
    # The sinks (one for each commodity)
    _sinks: List[Node]
    # The network inflow rates (one for each commodity)
    _u: List[PWConst]
    # The number of commodities
    _noOfCommodities: int

    def __init__(self, network: Network, numberOfCommodities: int):
        self._network = network
        self._noOfCommodities = numberOfCommodities

        # The zero-flow up to time zero
        self._upToAt = {}
        for v in network.nodes:
            self._upToAt[v] = zero

        # Initialize functions f^+,f^- and q for every edge e
        self._fPlus = {}
        self._fMinus = {}
        self._queues = {}
        for e in network.edges:
            self._queues[e] = PWLin([zero, zero], [zero], [zero])
            for i in range(numberOfCommodities):
                self._fPlus[(e, i)] = PWConst([zero, zero], [zero])
                self._fMinus[(e, i)] = PWConst([zero, e.tau], [zero])

        # Currently every commodity has a network inflow rate of zero
        self._u = [PWConst([zero], [], 0) for _ in range(self._noOfCommodities)]
        # Furthermore we do not know source and sink nodes yet
        self._sources = [None for _ in range(self._noOfCommodities)]
        self._sinks = [None for _ in range(self._noOfCommodities)]

    def setSource(self,commodity:int,s:Node):
        assert(commodity < self._noOfCommodities)
        self._sources[commodity] = s

    def setSink(self,commodity:int,t:Node):
        assert(commodity < self._noOfCommodities)
        self._sinks[commodity] = t

    def setU(self,commodity:int,u:PWConst):
        assert (commodity < self._noOfCommodities)
        self._u[commodity] = u


    # The travel time over an edge e at time theta
    def c(self, e:Edge, theta:number) -> number:
        # the queue on edge e has to be known up to at least time theta
        # If the flow has terminated then we can assume that all queues are empty after the
        # interval they are defined on
        if self.hasTerminated():
            if self._queues[e].segmentBorders[-1] >= theta:
                return self._queues[e].getValueAt(theta) / e.nu + e.tau
            else:
                return zero
        else:
            assert (self._queues[e].segmentBorders[-1] >= theta)
            return self._queues[e].getValueAt(theta) / e.nu + e.tau

    # The arrival time at the end of edge e if entering at time theta
    def T(self, e:Edge, theta:number) -> number:
        return theta + self.c(e, theta)

    # Determines the arrival time at the end of path p when starting at time theta
    def pathArrivalTime(self,p:Path,theta:number)->number:
        if len(p) == 0:
            return theta
        else:
            firstEdge = p.edges[0]
            return self.pathArrivalTime(Path(p.edges[1:],firstEdge.node_to),self.T(firstEdge,theta))

    def hasTerminated(self) -> bool:
        # Checks whether the flow has terminated, i.e. whether
        # all (non-zero) node inflow has been distributed to outgoing edges

        # TODO: It would be more efficient to not always do all these calculations from ground up
        #   but instead update some state variables whenever the flow changes
        #   However, this would require that the flow is only changed via class-methods

        for v in self._network.nodes:
            # Determine the last time with node inflow
            lastInflowTime = zero
            for i in range(self._noOfCommodities):
                if self._sources[i] == v:
                    lastInflowTime = max(lastInflowTime, self._u[i]._segmentBorders[-1])
                for e in v.incoming_edges:
                    fPei = self._fMinus[(e, i)]
                    # The following case distinction is necessary as the last inflow interval
                    # might be an interval with zero inflow (but not more than one as two adjacent
                    # intervals with zero flow would get unified to one by PWConst)
                    # If we change PWConst so that ending intervals of zero-value get deleted
                    # (since the default value is zero anyway), this would become unnecessary
                    if len(fPei._segmentValues) > 0 and fPei._segmentValues[-1] == 0:
                        lastInflowTime = max(lastInflowTime, fPei._segmentBorders[-2])
                    else:
                        lastInflowTime = max(lastInflowTime, fPei._segmentBorders[-1])

            if self._upToAt[v] < lastInflowTime:
                # There is still future inflow at node v which has not been redistributed
                return False

        # No nodes have any future inflow which still has to be redistributed
        return True

    def checkFlowConservation(self,v: Node,upTo: number,commodity: int) -> bool:
        # Checks whether flow conservation holds at node v during the interval [0,upTo]
        # i.e. \sum_{e \in \delta^-(v)}f_e,i^-(\theta) = \sum_{e \in \delta^+(v)}f_e,i^+(\theta)
        # for all nodes except source and sink of commodity i
        # For the source the same has to hold with the network inflow rate u of commodity i added on the left side
        # For the sink the = is replaced by >=
        theta = zero
        # Since all flow rates are assumed to be right-constant, it suffices to check the conditions
        # at every stepping point (for at least one of the relevant flow rate functions)
        while theta < upTo:
            nextTheta = infinity
            flow = zero
            # Add up node inflow rate (over all incoming edges)
            for e in v.incoming_edges:
                flow += self._fMinus[e, commodity].getValueAt(theta)
                nextTheta = min(nextTheta, self._fMinus[e, commodity].getNextStepFrom(theta))
            # If node v is commodity i's source we also add the network inflow rate
            if v == self._sources[commodity]:
                flow += self._u[commodity].getValueAt(theta)
                nextTheta = min(nextTheta, self._u[commodity].getNextStepFrom(theta))
            # Subtract all node outflow (over all outgoing edges)
            for e in v.outgoing_edges:
                flow -= self._fPlus[e, commodity].getValueAt(theta)
                nextTheta = min(nextTheta, self._fPlus[e, commodity].getNextStepFrom(theta))

            # Now check flow conservation at node v
            # First, a special case for the sink node
            if v == self._sinks[commodity]:
                if flow < 0:
                    print("Flow conservation does not hold at node ", v, " (sink!) at time ", theta)
                    return False

            # Then the case for all other nodes:
            elif flow != 0:
                print("Flow conservation does not hold at node ",v," at time ",theta)
                return False

            # The next stepping point:
            theta = nextTheta
        return True

    # Checks whether queues operate at capacity, i.e. whether the edge outflow rate is determined by
    # f^-_e(\theta+\tau_e) = \nu_e,                     if q_e(\theta) > 0
    # f^-_e(\theta+\tau_e) = \min{\nu_e,f^+_e(\theta)}, else
    def checkQueueAtCap(self, e: Edge, upTo: number) -> bool:
        theta = zero

        while theta < upTo:
            nextTheta = infinity
            outflow = zero
            inflow = zero
            for i in range(self._noOfCommodities):
                outflow += self._fMinus[(e, i)].getValueAt(theta + e.tau)
                inflow += self._fPlus[(e, i)].getValueAt(theta)
                nextTheta = min(nextTheta, self._fMinus[(e, i)].getNextStepFrom(theta + e.tau), self._fPlus[(e, i)].getNextStepFrom(theta))
            if self._queues[e].getValueAt(theta) > 0:
                if outflow != e.nu:
                    print("Queue on edge ",e, " does not operate at capacity at time ", theta)
                    return False
            else:
                assert(self._queues[e].getValueAt(theta) == 0)
                if outflow != min(inflow,e.nu):
                    print("Queue on edge ", e, " does not operate at capacity at time ", theta)
                    return False
            theta = nextTheta
        return True

    # Checks whether the queue-lengths are correct, i.e. determined by
    # q_e(\theta) = F_e^+(\theta) - F_e^-(\theta+\tau_e)
    def checkQueue(self,e: Edge,upTo: number):
        # Assumes that f^-_e = 0 on [0,tau_e)
        theta = zero
        currentQueue = zero
        if self._queues[e].getValueAt(theta) != 0:
            print("Queue on edge ", e, " does not start at 0")
            return False
        while theta < upTo:
            nextTheta = self._queues[e].getNextStepFrom(theta)
            inflow = zero
            outflow = zero
            for i in range(self._noOfCommodities):
                outflow += self._fMinus[(e, i)].getValueAt(theta + e.tau)
                inflow += self._fPlus[(e, i)].getValueAt(theta)
                nextTheta = min(nextTheta, self._fPlus[(e, i)].getNextStepFrom(theta), self._fMinus[(e, i)].getNextStepFrom(theta + e.tau))
            currentQueue += (inflow-outflow)*(nextTheta-theta)
            if nextTheta < infinity and currentQueue != self._queues[e].getValueAt(nextTheta):
                print("Queue on edge ", e, " wrong at time ", nextTheta)
                print("Should be ", currentQueue, " but is ", self._queues[e].getValueAt(nextTheta))
                return False
            theta = nextTheta
        return True

    # Check feasibility of the given flow up to the specified time horizon
    def checkFeasibility(self,upTo: number) -> bool:
        # Does not check FIFO (TODO)
        # Does not check non-negativity (TODO?)
        # Does not check whether the edge outflow rates are determined as far as possible
        # (i.e. up to time e.T(theta) if theta is the time up to which the inflow is given)
        # We implicitely assume that this is the case, but currently the user is responsible for ensuring this (TODO)
        feasible = True
        for i in range(self._noOfCommodities):
            for v in self._network.nodes:
                feasible = feasible and self.checkFlowConservation(v,upTo,i)
        for e in self._network.edges:
            feasible = feasible and self.checkQueueAtCap(e,upTo)
            feasible = feasible and self.checkQueue(e,upTo)
        return feasible

    # Returns a string representing the queue length functions as well as the edge in and outflow rates
    def __str__(self):
        s = "Queues:\n"
        for e in self._network.edges:
            s += " q_" + str(self._network.edges.index(e)) + str(e) + ": " + str(self._queues[e]) + "\n"
        s += "----------------------------------------------------------\n"
        for i in range(self._noOfCommodities):
            s += "Commodity " + str(i) + ":\n"
            for e in self._network.edges:
                s += "f_" + str(e) + ": " + str(self._fPlus[(e, i)]) + "\n"
                s += "f_" + str(e) + ": " + str(self._fMinus[(e, i)]) + "\n"
            s += "----------------------------------------------------------\n"
        return s

    # Creates a json file for use in Michael's visualization tool:
    # https://github.com/ArbeitsgruppeTobiasHarks/dynamic-prediction-equilibria/tree/main/visualization
    # Similar to https://github.com/ArbeitsgruppeTobiasHarks/dynamic-prediction-equilibria/tree/main/predictor/src/visualization
    # Does not set meaningful coordinates for nodes and sets all commodity's colors to DodgerBlue
    def to_json(self, filename:str):
        with open(filename, "w") as file:
            json.dump({
                "network": {
                    "nodes": [{"id": v.name, "x": 0, "y": 0} for v in self._network.nodes],
                    "edges": [{"id": id,
                               "from": e.node_from.name,
                               "to": e.node_to.name,
                               "capacity": e.nu,
                               "transitTime": e.tau}
                              for (id,e) in enumerate(self._network.edges)],
                    "commodities": [{ "id": id, "color": "dodgerblue"} for id in range(self._noOfCommodities)]
                },
                "flow": {
                    "inflow": [
                        [
                            {"times": [theta for theta in self._fPlus[(e, i)]._segmentBorders[:-1]],
                             "values": [val for val in self._fPlus[(e, i)]._segmentValues],
                             "domain": [0.0, "Infinity"]}
                            for i in range(self._noOfCommodities)
                        ]
                        for e in self._network.edges
                    ],
                    "outflow": [
                        [
                            {"times": [theta for theta in self._fMinus[(e, i)]._segmentBorders[:-1]],
                             "values": [val for val in self._fMinus[(e, i)]._segmentValues],
                             "domain": [0.0, "Infinity"]}
                            for i in range(self._noOfCommodities)
                        ]
                        for e in self._network.edges
                    ],
                    "queues": [
                        {"times": [theta for theta in self._queues[e].segmentBorders[:-1]],
                         "values": [self._queues[e].getValueAt(theta) for theta in self._queues[e].segmentBorders[:-1]],
                         "domain": ["-Infinity", "Infinity"], "lastSlope": 0.0, "firstSlope": 0.0
                        }
                        for e in self._network.edges
                    ]
                }
            },file)


# A path based description of feasible flows
# i.e. a network with a set of commodities and for every commodity a dictionary
# mapping paths of this commodity to path inflow rates
class PartialFlowPathBased:
    network: Network
    fPlus: List[Dict[Path, PWConst]]
    sources: List[Node]
    sinks: List[Node]
    noOfCommodities: int

    def __init__(self, network: Network, numberOfCommodities: int):
        self.network = network
        self.noOfCommodities = numberOfCommodities

        # Initialize functions f^+
        self.fPlus = [{} for _ in range(numberOfCommodities)]

        # Currently every commodity has a network inflow rate of zero
        self.u = [PWConst([zero],[],0) for _ in range(self.noOfCommodities)]
        # Furthermore we do not know source and sink nodes yet
        self.sources = [None for _ in range(self.noOfCommodities)]
        self.sinks = [None for _ in range(self.noOfCommodities)]

    # Arguments: commodity (id), paths:List[Path], pathinflows:List[PWConst]):
    def setPaths(self, commodity:int, paths:List[Path], pathinflows:List[PWConst]):
        assert (0 <= commodity < self.noOfCommodities)
        assert (len(paths) > 0)
        self.sources[commodity] = paths[0].getStart()
        self.sinks[commodity] = paths[0].getEnd()

        assert (len(paths) == len(pathinflows))
        for i in range(len(paths)):
            p = paths[i]
            # print("Checking path: P", i, printPathInNetwork(p,self.network))
            assert (p.getStart() == self.sources[commodity])
            assert (p.getEnd() == self.sinks[commodity])
            fp = pathinflows[i]
            # print("fp for path: ", i, sep = '', end = ' ')
            # print(str(p), fp)
            # print(printPathInNetwork(p,self.network), fp)
            if (p in self.fPlus[commodity]):
                print('p: ',printPathInNetwork(p,self.network))
                # print('comm: ', self.fPlus[commodity])
            assert (not p in self.fPlus[commodity])
            self.fPlus[commodity][p] = fp

    def getNoOfCommodities(self) -> int:
        return self.noOfCommodities

    def getEndOfInflow(self, commodity:int) -> number:
        assert (0 <= commodity < self.noOfCommodities)
        endOfInflow = zero
        for P in self.fPlus[commodity]:
            fP = self.fPlus[commodity][P]
            endOfInflow = max(endOfInflow, fP._segmentBorders[-1])
        return endOfInflow

    def __str__(self) -> str:
        s = "Path inflow rates \n"
        # TODO: Temporary: count of paths with positive flow
        fPosCount = 0
        for i in range(self.noOfCommodities):
            s += "  of commodity " + str(i) + "\n"
            # print("fplus ", self.fPlus)
            for j,P in enumerate(self.fPlus[i]):
                if self.fPlus[i][P]._noOfSegments > 1 or\
                        (self.fPlus[i][P]._noOfSegments == 1 and
                         self.fPlus[i][P]._segmentValues[0] > 0):
                    # show edge ids with paths here
                    s += "    into path P" + str(j) + " " + self.network.printPathInNetwork(P) +\
                            ": energy cons.: " + str(P.getNetEnergyConsump()) +\
                            ": latency: " + str(P.getFreeFlowTravelTime()) +\
                            ": price: " + str(P.getPrice()) +\
                            ": \n"
                    s += str(self.fPlus[i][P]) + "\n"
                    fPosCount += 1
        print('Number of paths with positive flow: %d'%fPosCount)
        return s

# A partial flow on a path, i.e. a path and for every edge e on this path a tuple of f^+_e and f^-_e
class PartialPathFlow:
    path: Path
    fPlus: List[PWConst]
    fMinus: List[PWConst]

    def __init__(self,path):
        self.path = path
        self.fPlus = []
        self.fMinus = []
        for e in path.edges:
            self.fPlus.append(PWConst([zero, zero], [zero], zero))
            self.fMinus.append(PWConst([zero, e.tau], [zero], zero))

