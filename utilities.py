from __future__ import annotations
from typing import List

import matplotlib.pyplot as plt

from extendedRational import *

# A right-constant function
class PWConst:
    noOfSegments: int
    segmentBorders: List[number]
    segmentValues: List[number]
    #TODO: rename the following parameter
    default: number
    autoSimplify: bool

    def __init__(self, borders: List[number], values: List[number],
                 default: number = None, autoSimplyfy: bool = True):
        # autoSimplify=True means that adjacent segments with the same value are automatically unified to a single segment
        self.default = default
        assert (len(borders) == len(values) + 1)
        self.noOfSegments = len(values)
        # TODO: check that borders are non-decreasing
        self.segmentBorders = borders
        self.segmentValues = values

        self.autoSimplify = autoSimplyfy

    def addSegment(self, border: number, value: number):
        # Adds a new constant segment at the right side
        assert (self.segmentBorders[-1] <= border + numPrecision)

        # Only add new segment if it is larger than the given precision
        if self.segmentBorders[-1] - numPrecision < border < self.segmentBorders[-1]+numPrecision:
            return

        if self.autoSimplify and len(self.segmentValues) > 0 and self.segmentValues[-1] == value:
            self.segmentBorders[-1] = border
        else:
            self.segmentBorders.append(border)
            self.segmentValues.append(value)
            self.noOfSegments += 1

    def getValueAt(self, x: number) -> number:
        if x < self.segmentBorders[0] or x >= self.segmentBorders[-1]:
            # x is outside the range of the function
            return self.default
        else:
            for i in range(0, self.noOfSegments):
                if x < self.segmentBorders[i + 1]:
                    return self.segmentValues[i]

    def getNextStepFrom(self, x: number) -> number:
        if x >= self.segmentBorders[-1]:
            if self.default is None:
                # TODO
                pass
            else:
                return infinity
        else:
            for i in range(0, self.noOfSegments + 1):
                if x < self.segmentBorders[i]:
                    return self.segmentBorders[i]

    def __add__(self, other: PWConst) -> PWConst:
        # Add two piecewise constant functions
        # TODO: Depends on which function does not have default...
        if self.default is None or other.default is None:
            default = None
            leftMost = max(self.segmentBorders[0], other.segmentBorders[0])
            rightMost = min(self.segmentBorders[-1], other.segmentBorders[-1])
        else:
            default = self.default + other.default
            leftMost = min(self.segmentBorders[0], other.segmentBorders[0])
            rightMost = max(self.segmentBorders[-1], other.segmentBorders[-1])

        sum = PWConst([leftMost], [], default, self.autoSimplify and other.autoSimplify)

        x = leftMost
        while x < rightMost:
            val = self.getValueAt(x) + other.getValueAt(x)
            x = min(self.getNextStepFrom(x), other.getNextStepFrom(x))
            sum.addSegment(x, val)

        return sum

    def smul(self, mu: number) -> PWConst:
        # Creates a new piecewise constant function by scaling the current one by mu

        if self.default is None:
            default = None
        else:
            default = mu * self.default
        scaled = PWConst([self.segmentBorders[0]], [], default, self.autoSimplify)
        for i in range(len(self.segmentValues)):
            scaled.addSegment(self.segmentBorders[i + 1], mu * self.segmentValues[i])

        return scaled

    def restrictTo(self, a: number, b: number, default: number = None) -> PWConst:
        # Creates a new piecewise constant function by restricting the current one to the interval [a,b]
        restrictedF = PWConst([a], [], default=default)
        x = a
        while x <= self.segmentBorders[-1] and x < b:
            val = self.getValueAt(x)
            x = min(self.getNextStepFrom(x), b)
            restrictedF.addSegment(x, val)
        if x < b and (not self.getValueAt(x) is None):
            restrictedF.addSegment(b, self.getValueAt(x))
        return restrictedF

    def __abs__(self) -> PWConst:
        # Creates a new piecewise constant functions |f|
        if self.default is None:
            default = None
        else:
            default = self.default.__abs__()
        absf = PWConst([self.segmentBorders[0]], [], default, self.autoSimplify)
        for i in range(len(self.segmentValues)):
            absf.addSegment(self.segmentBorders[i + 1], self.segmentValues[i].__abs__())

        return absf

    def integrate(self, a: number, b: number) -> number:
        # Determines the value of the integral of the given piecewise function from a to b
        assert (self.default is not None or (a >= self.segmentBorders[0] and b <= self.segmentBorders[-1]))

        integral = zero
        x = a
        while x < b:
            y = min(self.getNextStepFrom(x), b)
            integral += (y - x) * self.getValueAt(x)
            x = y

        return integral

    def norm(self) -> number:
        # Computes the L1-norm
        assert (self.default is None or self.default == 0)
        return self.__abs__().integrate(self.segmentBorders[0], self.segmentBorders[-1])

    def drawGraph(self, start: number, end: number):
        current = start
        x = []
        y = []
        while self.getNextStepFrom(current) < end:
            x.append(current)
            x.append(self.getNextStepFrom(current))
            y.append(self.getValueAt(current))
            y.append(self.getValueAt(current))
            current = self.getNextStepFrom(current)
        x.append(current)
        x.append(end)
        y.append(self.getValueAt(current))
        y.append(self.getValueAt(current))
        plt.plot(x, y)
        return plt

    def getXandY(self, start: number, end: number):
        current = start
        x = []
        y = []
        while self.getNextStepFrom(current) < end:
            x.append(current)
            x.append(self.getNextStepFrom(current))
            y.append(self.getValueAt(current))
            y.append(self.getValueAt(current))
            current = self.getNextStepFrom(current)
        x.append(current)
        x.append(end)
        y.append(self.getValueAt(current))
        y.append(self.getValueAt(current))
        return x,y

    def __str__(self):
        f = "|" + str(round(float(self.segmentBorders[0]),2)) + "|"
        for i in range(len(self.segmentValues)):
            # f += "-" + str(self.segmentValues[i]) + "-|" + str(self.segmentBorders[i + 1]) + "|"
            f += " " + str(round(float(self.segmentValues[i]),2)) + " |"
            if self.segmentBorders[i+1] < infinity:
                f += str(round(float(self.segmentBorders[i + 1]),2)) + "|"
            else:
                f += str(self.segmentBorders[i + 1]) + "|"
        return f


# A piecewise linear function, i.e. a function f which is of the form
#   f(x) = m_i*x + t_i for x \in [b_i,b_{i+1})
class PWLin:
    noOfSegments: int
    autoSimplify: bool
    segmentBorders: List[number]
    segmentTvalues: List[number]
    segmentMvalues: List[number]

    def __init__(self, borders: List[number], mvalues: List[number],
                 tvalues: List[number], autoSimplify: bool = True):
        # autoSimplify=True means that adjacent segments are automatically unified whenever possible
        # TODO: This should then be done in the initialisation!
        self.autoSimplify = autoSimplify

        self.noOfSegments = len(mvalues)
        assert (len(tvalues) == len(mvalues))
        assert (len(borders) == self.noOfSegments + 1)

        # TODO: check that borders are non-decreasing
        self.segmentBorders = borders
        self.segmentMvalues = mvalues
        self.segmentTvalues = tvalues


    def addSegmant(self, border: number, m: number, t: number = None):
        # Adds a new segment on the right side
        # If no t value is provided the function is extended continuously
        if t is None:
            assert (self.noOfSegments > 0)
            t = self.segmentTvalues[-1] + (self.segmentBorders[-1] - self.segmentBorders[-2]) * self.segmentMvalues[-1]

        if self.autoSimplify and self.noOfSegments > 0 and self.segmentMvalues[-1] == m and self.getValueAt(
                self.segmentBorders[-1]) == t:
            self.segmentBorders[-1] = border
        else:
            self.segmentBorders.append(border)
            self.segmentMvalues.append(m)
            self.segmentTvalues.append(t)
            self.noOfSegments += 1


    def getValueAt(self, x: number) -> number:
        # TODO: Should the second one be >=?
        if x < self.segmentBorders[0] or x > self.segmentBorders[-1]:
            # x is outside the range of the function
            assert(False)
        else:
            for i in range(0, self.noOfSegments):
                if x <= self.segmentBorders[i + 1]:
                    return self.segmentTvalues[i] + (x - self.segmentBorders[i]) * self.segmentMvalues[i]

    def getSlopeAt(self, x: number) -> number:
        if x < self.segmentBorders[0] or x >= self.segmentBorders[-1]:
            # x is outside the range of the function
            assert(False)
        else:
            for i in range(0, self.noOfSegments):
                if x < self.segmentBorders[i + 1]:
                    return self.segmentMvalues[i]


    def getNextStepFrom(self, x: number) -> number:
        if x >= self.segmentBorders[-1]:
            # TODO: Implement default value and/or better error handling
            assert(False)
        else:
            for i in range(0, self.noOfSegments + 1):
                if x < self.segmentBorders[i]:
                    return self.segmentBorders[i]

    def getPrevStepFrom(self, x: number) -> number:
        if x < self.segmentBorders[0]:
            # TODO: Implement default value and/or better error handling
            assert(False)
        else:
            for i in range(0, self.noOfSegments):
                if x < self.segmentBorders[i+1]:
                    return self.segmentBorders[i]

    def __add__(self, other : PWLin) -> PWLin:
        leftMost = max(self.segmentBorders[0], other.segmentBorders[0])
        rightMost = min(self.segmentBorders[-1], other.segmentBorders[-1])

        sum = PWLin([leftMost], [], [], self.autoSimplify and other.autoSimplify)

        b = leftMost
        while b < rightMost:
            t = self.getValueAt(b) + other.getValueAt(b)
            m = self.getSlopeAt(b) + other.getSlopeAt(b)
            b = min(self.getNextStepFrom(b), other.getNextStepFrom(b))
            sum.addSegmant(b,m,t)

        return sum

    def smul(self, mu: number) -> PWLin:
        return PWLin(self.segmentBorders, [mu*m for m in self.segmentMvalues],
                     [mu*t for t in self.segmentTvalues], self.autoSimplify)



    # Composes the function with another piecewise linear function
    # i.e. if the current function is f then this results in f∘g defined on the maximal possible interval containing
    # the start of the domain of g
    def composeWith(self, g:PWLin) -> PWLin:
        assert self.segmentBorders[0] <= g.getValueAt(g.segmentBorders[0]) and g.getValueAt(g.segmentBorders[0]) < self.segmentBorders[-1]
        # TODO: Can we also handle other cases? Maybe if we use default values?
        h = PWLin([g.segmentBorders[0]], [], [], self.autoSimplify and g.autoSimplify)
        for i in range(g.noOfSegments):
            gt = g.segmentTvalues[i]
            if self.segmentBorders[-1] <= gt:
                return h
            gm = g.segmentMvalues[i]
            if gm == 0:
                h.addSegmant(g.segmentBorders[i+1],zero,self.getValueAt(gt))
            elif gm > 0:
                # The right end of the interval we can extend for:
                bMax = min(g.segmentBorders[i+1],g.segmentBorders[i]+(self.segmentBorders[-1]-gt)/gm)
                b = g.segmentBorders[i]
                while b < bMax:
                    # Determine next break point
                    bNew = min(bMax, b + (self.getNextStepFrom(gt) - gt) / gm)
                    # Determine t-value for next interval
                    t = self.getValueAt(g.getValueAt(b))
                    # Determine slope for next interval
                    # Take the slope in the middle of the interval in order to avoid choosing the wrong slope
                    # because of rounding issues
                    # TODO: Also for t?
                    m = self.getSlopeAt(g.getValueAt((b+bNew)/2))*gm

                    h.addSegmant(bNew,m,t)
                    b = bNew
                    gt = g.getValueAt(b)
            elif gm < 0:
                if gt <= self.segmentBorders[0]:
                    return h
                # The right end of the interval we can extend for:
                bMax = min(g.segmentBorders[i+1],g.segmentBorders[i]+(self.segmentBorders[0]-gt)/gm)
                b = g.segmentBorders[i]
                while b < bMax:
                    # Determine next break point
                    bNew = min(bMax, b + (self.getPrevStepFrom(gt) - gt) / gm)
                    # Determine t-value for next interval
                    t = self.getValueAt(g.getValueAt(b))
                    # Determine slope for next interval
                    # Take the slope in the middle of the interval in order to avoid choosing the wrong slope
                    # because of rounding issues
                    # TODO: Also for t?
                    m = self.getSlopeAt(g.getValueAt((b+bNew)/2))*gm

                    h.addSegmant(bNew,m,t)
                    b = bNew
                    gt = g.getValueAt(b)

        return h


    def drawGraph(self, start: number, end: number):
        x = [start]
        y = [self.getValueAt(start)]
        while self.getNextStepFrom(x[-1]) < end:
            x.append(self.getNextStepFrom(x[-1]))
            y.append(self.getValueAt(x[-1]))
        x.append(end)
        y.append(self.getValueAt(end))
        plt.plot(x, y)
        return plt


    def segment_as_str(self,i:int,ommitStart:bool=True) -> str:
        # Creates a string of the form |2|3 4|4| for the i-th segment
        # |2| is omitted if ommitStart=True (standard)
        assert(i<self.noOfSegments)
        s = ""
        if not ommitStart:
            s += "|" + str(round(float(self.segmentBorders[i]),2)) + "|"
        s += str(round(float(self.segmentTvalues[i]),2)) + " "
        if self.segmentBorders[i+1] < infinity:
            s += str(round(float(self.segmentTvalues[i] + (self.segmentBorders[i + 1] - self.segmentBorders[i]) * self.segmentMvalues[i]),2))
        else:
            if self.segmentMvalues[i] == 0:
                s += "0"
            elif self.segmentMvalues[i] > 0:
                s += "infty"
            else:
                s += "-infty"
        s += "|" + str(round(float(self.segmentBorders[i+1]),2)) + "|"
        return s

    def __str__(self) -> str:
        f = "|" + str(round(float(self.segmentBorders[0]),2)) + "|"
        for i in range(len(self.segmentMvalues)):
            f += self.segment_as_str(i)

        return f

    # def __str__(self):
        # f = "|" + str(self.segmentBorders[0]) + "|"
        # for i in range(len(self.segmentMvalues)):
            # f += str(self.segmentTvalues[i]) + "-" \
                 # + str(
                # self.segmentTvalues[i] + (self.segmentBorders[i + 1] - self.segmentBorders[i]) * self.segmentMvalues[i]) \
                 # + "|" + str(self.segmentBorders[i + 1]) + "|"

        # return f