from __future__ import annotations

from typing import List

from network import *

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