from __future__ import annotations

from typing import List

from flows import *
from utilities import *

import heapq


class Event:
    # A class for events in the network loading algorithm
    # An event consists of a time and a node
    # Such an event denotes a time at which a new flow split has to be calculated at the given node
    # (because the amount of incoming flow into the node potentially changes after this time)
    time: number
    v: Node
    description: str

    def __init__(self, time: number, v: Node, desc: str=""):
        self.time = time
        self.v = v
        self.description = desc

    # Events are ordered by their trigger time
    # (this is used to be able to place events in a FIFO queue)
    def __lt__(self, other):
        return self.time < other.time

    def __str__(self):
        s = "Event at node " + str(self.v) + " at time " + str(float(self.time)) + " ≈ " + str(self.time)
        if self.description != "":
            s += ": " + self.description
        return s

class EventQueue:
    # A queue of events, where events are ordered by non-decreasing trigger time
    events: List[Event]

    def __init__(self):
        self.events = []

    # Adds a new event to the queue
    def pushEvent(self,time:number,v:Node,desc:str=""):
        heapq.heappush(self.events, Event(time, v, desc))

    # Returns the next event and removes it from the queue
    def popEvent(self) -> Event:
        return heapq.heappop(self.events)

    # Indicates whether there are any events left in the queue
    def isEmpty(self) -> bool:
        return len(self.events) == 0


# Determines the edge based description of a flow given the path inflow rates
# The arguments are:
# - pathBasedFlows: An object of class PartialFlowPathBased containing a set of commodities and for every commodity
#   its source, sink, network inflow rate and a list of tuples (path p, inflow rate into path p)
# - timeHorizon (optional): the flow will be determined for the time interval [0,timeHorizon]; default is infinity
# - verbose: If True, more information is printed during the network loading procedure; default is False
# TODO: Currently, this procedure relies on the fact that a final interval of value zero in a piecewise-constant
#   function does not get removed by the simplifying procedure
def networkLoading(pathBasedFlows : PartialFlowPathBased, timeHorizon: number=infinity, verbose:bool=False) -> PartialFlow:
    network = pathBasedFlows.network

    # We start by creating a list of partial path flows.
    # For every commodity and each of its used paths, the corresponding path flow will contain edge in- and outflow rates
    # for every edge this path
    # Note, that edges may occur multiple times on a path and then also have multiple in-/outflow functions here
    # This is important to keep track on where particles leaving an edge are supposed to travel next
    partialPathFlows = []
    for i in range(pathBasedFlows.getNoOfCommodities()):
        partialPathFlows.extend([PartialPathFlow(path) for path in pathBasedFlows.fPlus[i]])

    # During the network loading we consider every path of each commodity as its own commodity:
    # TODO: maybe convert back into a flow with only one in-/outflow function for every true commodity at the end?
    noOfCommodities = len(partialPathFlows)

    # This will be the actual flow (i.e. edge in-/outflow rates for all edges and commodities and queues for all edges
    flow = PartialFlow(network,noOfCommodities)


    # For all commodities set source, sink and network inflow rate
    j = 0
    for i in range(pathBasedFlows.getNoOfCommodities()):
        for p in pathBasedFlows.fPlus[i]:
            flow.setSource(j, pathBasedFlows.sources[i])
            flow.setSink(j, pathBasedFlows.sinks[i])
            flow.setU(j, pathBasedFlows.fPlus[i][p])
            j += 1

    # A queue of node events:
    eventQueue = EventQueue()
    # During the network loading the following invariant always holds (at each node v separately):
    # There is some time \theta such that the edge inflow rates for all edges leaving v are defined up to time \theta
    # and if \theta < timeHorizon there is an event (\theta,v) in the queue (signifying that at that time a new flow
    # distribution has to be determined at node v
    for v in network.nodes:
        eventQueue.pushEvent(zero,v,"first flow")

    # While there are events left to handle
    while not (eventQueue.isEmpty() or flow.hasTerminated()):
        event = eventQueue.popEvent()
        if verbose: print("Handling ", event)
        v = event.v
        theta = event.time
        # currently all edge inflow rates for edges leaving v are defined up to time \theta
        assert(flow.upToAt[v] == theta)

        # We will now extend them for some additional time interval [theta,nextTheta]:

        # At most, we extend until the given timeHorizon
        nextTheta = timeHorizon
        nextThetaReason = "End of time horizon"

        # For each edge leaving v we will add up (over all commodities) the total rate of flow we have to send over
        # this edge
        flowTo = {e: 0 for e in v.outgoing_edges}
        for i in range(noOfCommodities):
            if flow.sources[i] == v:
                # If node v is the source of the current commodity i then this commodity's network inflow rate u
                # is added to the inflow into the first edge of this commodity's path
                flowTo[partialPathFlows[i].path.edges[0]] += flow.u[i].getValueAt(theta)
                # A change in this network inflow rate will require a new flow distribution at v
                if flow.u[i].getNextStepFrom(theta) < nextTheta:
                    nextTheta = flow.u[i].getNextStepFrom(theta)
                    nextThetaReason = "change in network inflow rate of commodity " + str(i)
                assert(nextTheta>theta)

            for j in range(len(partialPathFlows[i].path.edges)-1):
                # Stepping through the current commodities path to check whether node v occurs on it as any edge's
                # endnode (the last edge can be ignored as flow leaving this edges leaves the network)
                if partialPathFlows[i].path.edges[j].node_to == v:
                    # If so, the outflow from this edge will be sent to the next edge on the path
                    flowTo[partialPathFlows[i].path.edges[j+1]] += partialPathFlows[i].fMinus[j].getValueAt(theta)
                    # and a change in the edge outflow rate from the previous edge will require a new flow distribution at v
                    if partialPathFlows[i].fMinus[j].getNextStepFrom(theta) < nextTheta:
                        nextTheta = partialPathFlows[i].fMinus[j].getNextStepFrom(theta)
                        nextThetaReason = "change in edge outflow rate of commodity " + str(i) + " from edge " + str(partialPathFlows[i].path.edges[j])
                    assert (nextTheta > theta)

        # Check whether queues on outgoing edges deplete before nextTheta
        # and reduce nextTheta in that case
        for e in v.outgoing_edges:
            # The queue length at time theta:
            qeAtTheta = flow.queues[e].getValueAt(theta)
            # Check whether queue is non-empty at \theta and will be shrinking after \theta:
            if qeAtTheta > numPrecision and flowTo[e] < e.nu - numPrecision:
                # the time at which the queue on e will run empty (assuming the inflow into e remains constant)
                x = qeAtTheta/(e.nu-flowTo[e])
                # If this is before the current value of nextTheta we reduce nextTheta to the time the queue runs empty
                if theta < theta+x < nextTheta:
                    nextTheta = theta+x
                    nextThetaReason = "Queue depletes on edge " + str(e)
                assert (nextTheta > theta)

        # Now we have all the information to extend the current flow at node v over the time interval [theta,nextTheta]:
        # using constant in- and outflows for all edges leaving v

        # Determine queues on the interval [theta,nextTheta]
        # (here we use the fact that nextTheta has been chosen such that no queue leaving v depletes before nextTheta)
        for e in v.outgoing_edges:
            if flow.queues[e].getValueAt(theta) > numPrecision:
                flow.queues[e].addSegmant(nextTheta,flowTo[e]-e.nu)
            else:
                flow.queues[e].addSegmant(nextTheta, max(flowTo[e] - e.nu,0))

        # Redistribute the node inflow to outgoing edges
        for i in range(noOfCommodities):
            # First we determine the values of f^+ and f^- in terms of the path flows
            # (i.e. if an edge occurs multiple times on a path its distinct occurrences are considered as different edges)
            for j in range(len(partialPathFlows[i].path.edges)):
                if partialPathFlows[i].path.edges[j].node_from == v:
                    e = partialPathFlows[i].path.edges[j]
                    # Determine inflow into edge e
                    if j == 0:
                        # The edge is the first on commodity i's path.
                        # Thus, the inflow into this edge is determined by the network inflow rate u_i
                        inflow = flow.u[i].getValueAt(theta)
                    else:
                        # Otherwise the inflow is determined by the outflow of the previous edge on i's path
                        inflow = partialPathFlows[i].fMinus[j-1].getValueAt(theta)
                    # Extend fPlus of edge e
                    partialPathFlows[i].fPlus[j].addSegment(nextTheta,inflow)

                    # Determine outflow from edge e
                    if flowTo[e] > 0:
                        if flow.queues[e].getValueAt(theta) > numPrecision:
                            outflowRate = partialPathFlows[i].fPlus[j].getValueAt(theta) / flowTo[e] * e.nu
                        else:
                            outflowRate = partialPathFlows[i].fPlus[j].getValueAt(theta) / flowTo[e] * min(flowTo[e], e.nu)
                    else:
                        outflowRate = 0
                    # Extend fMinus of edge e
                    if nextTheta < infinity:
                        partialPathFlows[i].fMinus[j].addSegment(flow.T(e, nextTheta), outflowRate)
                    else:
                        partialPathFlows[i].fMinus[j].addSegment(infinity, outflowRate)

            # Now we convert the path flows into the actual edge in- and outflow rates
            # (i.e. if an edge occurs multiple times on a commodity's path we add up the corresponding rates
            # from the path flow)
            for e in v.outgoing_edges:
                inflowRate = zero
                outflowRate = zero
                for j in range(len(partialPathFlows[i].path.edges)):
                    if partialPathFlows[i].path.edges[j] == e:
                        inflowRate += partialPathFlows[i].fPlus[j].segmentValues[-1]
                        outflowRate += partialPathFlows[i].fMinus[j].segmentValues[-1]
                flow.fPlus[(e,i)].addSegment(nextTheta,inflowRate)
                if nextTheta < infinity:
                    flow.fMinus[(e, i)].addSegment(flow.T(e,nextTheta), outflowRate)
                else:
                    assert (outflowRate < numPrecision)
                    flow.fMinus[(e, i)].addSegment(infinity, zero)

        # Now the extension at node v is done -> we have a flow up to time nextTheta at node v
        flow.upToAt[v] = nextTheta

        # At time nextTheta we will have to determine a new flow distribution at node v
        # (unless we already reached our desired time horizon then)
        if nextTheta < timeHorizon:
            eventQueue.pushEvent(nextTheta,v,nextThetaReason)

    return flow
