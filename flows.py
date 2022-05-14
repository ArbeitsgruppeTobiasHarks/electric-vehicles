from __future__ import annotations

from typing import Dict, List
from network import *
from utilities import *

import json

# A partial feasible flow with in-/outflow rates for every edges and commodity and queues for every edge
# Feasibility is not checked automatically but a check can be initiated by calling .checkFeasibility
class PartialFlow:
    # The network the flow lives in:
    network: Network
    # Stores for every node until which time the flow has been calculated:
    upToAt: Dict[Node, number]
    # TODO: Currently the user has to make sure the values of upToAt are kept up to date
    #   It would be better if this is done within this class!
    #   This would require to only allow changes of the flow via class-methods
    #   Is it possible to enforce this in python (i.e. are there private methods?)
    # The edge inflow rates (commodity specific)
    fPlus: Dict[(Edge,int), PWConst]
    # The edge outflow rates (commodity specific)
    fMinus: Dict[(Edge,int), PWConst]
    # The queues
    queues: Dict[Edge, PWLin]
    # The sources (one for each commodity)
    sources: List[Node]
    # The sinks (one for each commodity)
    sinks: List[Node]
    # The network inflow rates (one for each commodity)
    u: List[PWConst]
    # The number of commodities
    noOfCommodities: int

    def __init__(self, network: Network, numberOfCommodities: int):
        self.network = network
        self.noOfCommodities = numberOfCommodities

        # The zero-flow up to time zero
        self.upToAt = {}
        for v in network.nodes:
            self.upToAt[v] = zero

        # Initialize functions f^+,f^- and q for every edge e
        self.fPlus = {}
        self.fMinus = {}
        self.queues = {}
        for e in network.edges:
            self.queues[e] = PWLin([zero, zero], [zero], [zero])
            for i in range(numberOfCommodities):
                self.fPlus[(e,i)] = PWConst([zero,zero],[zero])
                self.fMinus[(e,i)] = PWConst([zero,e.tau],[zero])

        # Currently every commodity has a network inflow rate of zero
        self.u = [PWConst([zero],[],0) for _ in range(self.noOfCommodities)]
        # Furthermore we do not know source and sink nodes yet
        self.sources = [None for _ in range(self.noOfCommodities)]
        self.sinks = [None for _ in range(self.noOfCommodities)]

    def setSource(self,commodity:int,s:Node):
        assert(commodity < self.noOfCommodities)
        self.sources[commodity] = s

    def setSink(self,commodity:int,t:Node):
        assert(commodity < self.noOfCommodities)
        self.sinks[commodity] = t

    def setU(self,commodity:int,u:PWConst):
        assert (commodity < self.noOfCommodities)
        self.u[commodity] = u


    # The travel time over an edge e at time theta
    def c(self, e:Edge, theta:number) -> number:
        # the queue on edge e has to be known up to at least time theta
        # If the flow has terminated then we can assume that all queues are empty after the
        # interval they are defined on
        # TODO: This should somehow happen automatically!
        if self.hasTerminated():
            if self.queues[e].segmentBorders[-1] >= theta:
                return self.queues[e].getValueAt(theta)/e.nu + e.tau
            else:
                return zero
        else:
            assert (self.queues[e].segmentBorders[-1] >= theta)
            return self.queues[e].getValueAt(theta)/e.nu + e.tau

    # The arrival time at the end of edge e if entering at time theta
    def T(self, e:Edge, theta:number) -> number:
        return theta + self.c(e, theta)

    # Determines the arrival time at the end of path p when starting at time theta
    def pathArrivalTime(self,p:Path,theta:number)->number:
        # TODO: check whether all necessary queues are available
        if len(p) == 0:
            return theta
        else:
            firstEdge = p.edges[0]
            return self.pathArrivalTime(Path(p.edges[1:],firstEdge.node_to),self.T(firstEdge,theta))


    # The travel time over an edge e at time theta as a function of theta
    def cf(self, e:Edge) -> PWLin:
        delay = self.queues[e].smul(1/e.nu)
        return PWLin([delay.segmentBorders[0],delay.segmentBorders[-1]],[zero],[e.tau]) + delay


    # The arrival time at the end of edge e if entering at time theta as function of theta
    def Tf(self, e:Edge) -> PWLin:
        return PWLin([zero,infinity],[makeNumber(1)],[zero]) + self.cf(e)

    # Determines the arrival time at the end of path p when starting at time theta as function of theta
    def pathArrivalTimef(self,p:Path)-> PWLin:
        if len(p) == 0:
            return PWLin([zero,infinity],[makeNumber(1)],[zero])
        else:
            firstEdge = p.edges[0]
            return self.pathArrivalTimef(Path(p.edges[1:],firstEdge.node_to)).composeWith(self.Tf(firstEdge))

    def hasTerminated(self) -> bool:
        # Checks whether the flow has terminated, i.e. whether
        # all (non-zero) node inflow has been distributed to outgoing edges

        # TODO: It would be more efficient to not always do all these calculations from ground up
        #   but instead update some state variables whenever the flow changes
        #   However, this would require that the flow is only changed via class-methods

        for v in self.network.nodes:
            # Determine the last time with node inflow
            lastInflowTime = zero
            for i in range(self.noOfCommodities):
                if self.sources[i] == v:
                    lastInflowTime = max(lastInflowTime,self.u[i].segmentBorders[-1])
                for e in v.incoming_edges:
                    fPei = self.fMinus[(e,i)]
                    # The following case distinction is necessary as the last inflow interval
                    # might be an interval with zero inflow (but not more than one as two adjacent
                    # intervals with zero flow would get unified to one by PWConst)
                    # If we change PWConst so that ending intervals of zero-value get deleted
                    # (since the default value is zero anyway), this would become unnecessary
                    if len(fPei.segmentValues) > 0 and fPei.segmentValues[-1] == 0:
                        lastInflowTime = max(lastInflowTime,fPei.segmentBorders[-2])
                    else:
                        lastInflowTime = max(lastInflowTime, fPei.segmentBorders[-1])

            if self.upToAt[v] < lastInflowTime:
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
                flow += self.fMinus[e,commodity].getValueAt(theta)
                nextTheta = min(nextTheta,self.fMinus[e,commodity].getNextStepFrom(theta))
            # If node v is commodity i's source we also add the network inflow rate
            if v == self.sources[commodity]:
                flow += self.u[commodity].getValueAt(theta)
                nextTheta = min(nextTheta, self.u[commodity].getNextStepFrom(theta))
            # Subtract all node outflow (over all outgoing edges)
            for e in v.outgoing_edges:
                flow -= self.fPlus[e, commodity].getValueAt(theta)
                nextTheta = min(nextTheta, self.fPlus[e, commodity].getNextStepFrom(theta))

            # Now check flow conservation at node v
            # First, a special case for the sink node
            if v == self.sinks[commodity]:
                if flow < 0:
                    # TODO: Fehlermeldung
                    print("Flow conservation does not hold at node ", v, " (sink!) at time ", theta)
                    return False

            # Then the case for all other nodes:
            elif flow != 0:
                # TODO: Fehlermeldung
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
            for i in range(self.noOfCommodities):
                outflow += self.fMinus[(e,i)].getValueAt(theta+e.tau)
                inflow += self.fPlus[(e,i)].getValueAt(theta)
                nextTheta = min(nextTheta,self.fMinus[(e,i)].getNextStepFrom(theta+e.tau),self.fPlus[(e,i)].getNextStepFrom(theta))
            if self.queues[e].getValueAt(theta) > 0:
                if outflow != e.nu:
                    # TODO: Fehlermeldung
                    print("Queue on edge ",e, " does not operate at capacity at time ", theta)
                    return False
            else:
                assert(self.queues[e].getValueAt(theta) == 0)
                if outflow != min(inflow,e.nu):
                    # TODO: Fehlermeldung
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
        if self.queues[e].getValueAt(theta) != 0:
            # TODO: Fehlermeldung
            print("Queue on edge ", e, " does not start at 0")
            return False
        while theta < upTo:
            nextTheta = self.queues[e].getNextStepFrom(theta)
            inflow = zero
            outflow = zero
            for i in range(self.noOfCommodities):
                outflow += self.fMinus[(e, i)].getValueAt(theta + e.tau)
                inflow += self.fPlus[(e, i)].getValueAt(theta)
                nextTheta = min(nextTheta,self.fPlus[(e,i)].getNextStepFrom(theta),self.fMinus[(e,i)].getNextStepFrom(theta+e.tau))
            currentQueue += (inflow-outflow)*(nextTheta-theta)
            if nextTheta < infinity and currentQueue != self.queues[e].getValueAt(nextTheta):
                # TODO: Fehlermeldung
                print("Queue on edge ", e, " wrong at time ", nextTheta)
                print("Should be ", currentQueue, " but is ", self.queues[e].getValueAt(nextTheta))
                return False
            theta = nextTheta
        return True

    # Check feasibility of the given flow up to the specified time horizon
    def checkFeasibility(self,upTo: number) -> bool:
        # Does not check FIFO (TODO)
        # Does not check non-negativity (TODO?)
        # Does not check whether the edge outflow rates are determined as ar as possible
        # (i.e. up to time e.T(theta) if theta is the time up to which the inflow is given)
        # We implicitely assume that this is the case, but currently the user is responible for ensuring this (TODO)
        feasible = True
        for i in range(self.noOfCommodities):
            for v in self.network.nodes:
                feasible = feasible and self.checkFlowConservation(v,upTo,i)
        for e in self.network.edges:
            feasible = feasible and self.checkQueueAtCap(e,upTo)
            feasible = feasible and self.checkQueue(e,upTo)
        return feasible

    # Returns a string representing the queue length functions as well as the edge in and outflow rates
    def __str__(self):
        s = "Queues:\n"
        for e in self.network.edges:
            s += " q_" + str(self.network.edges.index(e)) + str(e) + ": " + str(self.queues[e]) + "\n"
        s += "----------------------------------------------------------\n"
        for i in range(self.noOfCommodities):
            s += "Commodity " + str(i) + ":\n"
            for e in self.network.edges:
                s += "f_" + str(e) + ": " + str(self.fPlus[(e,i)]) + "\n"
                s += "f_" + str(e) + ": " + str(self.fMinus[(e,i)]) + "\n"
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
                    "nodes": [{"id": v.name, "x": 0, "y": 0} for v in self.network.nodes],
                    "edges": [{"id": id,
                               "from": e.node_from.name,
                               "to": e.node_to.name,
                               "capacity": e.nu,
                               "transitTime": e.tau}
                              for (id,e) in enumerate(self.network.edges)],
                    "commodities": [{ "id": id, "color": "dodgerblue"} for id in range(self.noOfCommodities)]
                },
                "flow": {
                    "inflow": [
                        [
                            {"times": [theta for theta in self.fPlus[(e,i)].segmentBorders[:-1]],
                             "values": [val for val in self.fPlus[(e,i)].segmentValues],
                             "domain": [0.0, "Infinity"]}
                            for i in range(self.noOfCommodities)
                        ]
                        for e in self.network.edges
                    ],
                    "outflow": [
                        [
                            {"times": [theta for theta in self.fMinus[(e,i)].segmentBorders[:-1]],
                             "values": [val for val in self.fMinus[(e,i)].segmentValues],
                             "domain": [0.0, "Infinity"]}
                            for i in range(self.noOfCommodities)
                        ]
                        for e in self.network.edges
                    ],
                    "queues": [
                        {"times": [theta for theta in self.queues[e].segmentBorders[:-1]],
                         "values": [self.queues[e].getValueAt(theta) for theta in self.queues[e].segmentBorders[:-1]],
                         "domain": ["-Infinity", "Infinity"], "lastSlope": 0.0, "firstSlope": 0.0
                        }
                        for e in self.network.edges
                    ]
                }
            },file)


# A path based description of feasible flows:
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
            endOfInflow = max(endOfInflow, fP.segmentBorders[-1])
        return endOfInflow

    def __str__(self) -> str:
        s = "Path inflow rates \n"
        # TODO: Temporary: count of paths with positive flow
        fPosCount = 0
        for i in range(self.noOfCommodities):
            s += "  of commodity " + str(i) + "\n"
            # print("fplus ", self.fPlus)
            for j,P in enumerate(self.fPlus[i]):
                if self.fPlus[i][P].noOfSegments > 1 or\
                        (self.fPlus[i][P].noOfSegments == 1 and
                                self.fPlus[i][P].segmentValues[0] > 0):
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

