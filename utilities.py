from __future__ import annotations
from typing import List, Tuple

import matplotlib.pyplot as plt

from extendedRational import *

# A right-constant function with finitely many steps
class PWConst:
    _noOfSegments: int
    _segmentBorders: List[number]
    _segmentValues: List[number]
    _defaultValue: number
    _autoSimplify: bool

    def __init__(self, borders: List[number], values: List[number],
                 defaultValue: number = None, autoSimplyfy: bool = True):
        # autoSimplify=True means that adjacent segments with the same value are automatically unified
        # to a single segment
        # If a defaultValue is given this is the value of the function outside of the given borders
        # If none is given the function is undefined there
        # Note, that the first or last interval can have the same value as the default value and will not be deleted
        # by autoSimplify
        self._defaultValue = defaultValue
        assert (len(borders) == len(values) + 1)
        self._noOfSegments = len(values)
        # TODO: check that borders are non-decreasing
        self._segmentBorders = borders
        self._segmentValues = values

        self._autoSimplify = autoSimplyfy

    def addSegment(self, border: number, value: number):
        # Adds a new constant segment at the right side
        assert (self._segmentBorders[-1] <= border + numPrecision)

        # Only add new segment if it is larger than the given precision
        if self._segmentBorders[-1] - numPrecision < border < self._segmentBorders[-1]+numPrecision:
            return

        # If autoSimplify is active and the new intervals value is the same (up to the given precision) as the one of
        # the last interval, we extend the last interval instead of creating a new one
        if self._autoSimplify and len(self._segmentValues) > 0 and value - numPrecision <= self._segmentValues[-1] <= value + numPrecision:
            self._segmentBorders[-1] = border
        else:
            self._segmentBorders.append(border)
            self._segmentValues.append(value)
            self._noOfSegments += 1

    def addSegmentRel(self, length: number, value: number):
        # Adds a new constant segment of length length at the right side
        self.addSegment(self._segmentBorders[-1]+length,value)

    def getValueAt(self, x: number) -> number:
        # Returns the value of the function at x
        if x < self._segmentBorders[0] or x >= self._segmentBorders[-1]:
            # x is outside the range of the function
            return self._defaultValue
        else:
            for i in range(0, self._noOfSegments):
                if x < self._segmentBorders[i + 1]:
                    return self._segmentValues[i]

    def getNextStepFrom(self, x: number) -> number:
        # get the next step of the function strictly after x
        if x >= self._segmentBorders[-1]:
            if self._defaultValue is None:
                # TODO: Raise an error
                pass
            else:
                return infinity
        else:
            for i in range(0, self._noOfSegments + 1):
                if x < self._segmentBorders[i]:
                    return self._segmentBorders[i]

    def __add__(self, other: PWConst) -> PWConst:
        # Add two piecewise constant functions

        # If at least one of the functions is undefined outside its borders the sum of the two function can only be
        # defined within the boundaries of that function (the intersection of the two boundaries if both functions
        # are undefined outside their boundaries)
        if self._defaultValue is None and other._defaultValue is None:
            default = None
            leftMost = max(self._segmentBorders[0], other._segmentBorders[0])
            rightMost = min(self._segmentBorders[-1], other._segmentBorders[-1])
        elif self._defaultValue is None and not (other._defaultValue is None):
            default = None
            leftMost = self._segmentBorders[0]
            rightMost = self._segmentBorders[-1]
        elif not(self._defaultValue is None) and other._defaultValue is None:
            default = None
            leftMost = other._segmentBorders[0]
            rightMost = other._segmentBorders[-1]
        else:
            default = self._defaultValue + other._defaultValue
            leftMost = min(self._segmentBorders[0], other._segmentBorders[0])
            rightMost = max(self._segmentBorders[-1], other._segmentBorders[-1])

        sum = PWConst([leftMost], [], default, self._autoSimplify and other._autoSimplify)

        x = leftMost
        while x < rightMost:
            val = self.getValueAt(x) + other.getValueAt(x)
            x = min(self.getNextStepFrom(x), other.getNextStepFrom(x))
            sum.addSegment(x, val)

        return sum

    def smul(self, mu: number) -> PWConst:
        # Creates a new piecewise constant function by scaling the current one by mu

        if self._defaultValue is None:
            default = None
        else:
            default = mu * self._defaultValue
        scaled = PWConst([self._segmentBorders[0]], [], default, self._autoSimplify)
        for i in range(len(self._segmentValues)):
            scaled.addSegment(self._segmentBorders[i + 1], mu * self._segmentValues[i])

        return scaled

    def restrictTo(self, a: number, b: number, default: number = None) -> PWConst:
        # Creates a new piecewise constant function by restricting the current one to the interval [a,b)
        # and setting it to default outside [a,b)

        x = max(a, self._segmentBorders[0])
        restrictedF = PWConst([x], [], defaultValue=default)

        while x <= self._segmentBorders[-1] and x < b:
            val = self.getValueAt(x)
            x = min(self.getNextStepFrom(x), b)
            restrictedF.addSegment(x, val)

        if x < b and (not self.getValueAt(x) is None):
            restrictedF.addSegment(b, self.getValueAt(x))

        return restrictedF

    def isZero(self) -> bool:
        # Checks whether the function is zero (up to the given precision) wherever it is defined
        if self._defaultValue is not None and -infinity < self._segmentBorders[0] \
                and self._segmentBorders[-1] < infinity and not(isZero(self._defaultValue)):
            # If the default value is not zero, the function is not zero
            return False
        for y in self._segmentValues:
            if not(isZero(y)):
                # If there is one segment where the function is non-zero, the function is not zero
                # (this assume that there are no zero-length intervals!)
                return False
        return True

    def __abs__(self) -> PWConst:
        # Creates a new piecewise constant functions |f|
        if self._defaultValue is None:
            default = None
        else:
            default = self._defaultValue.__abs__()
        absf = PWConst([self._segmentBorders[0]], [], default, self._autoSimplify)
        for i in range(len(self._segmentValues)):
            absf.addSegment(self._segmentBorders[i + 1], self._segmentValues[i].__abs__())

        return absf

    def integrate(self, a: number, b: number) -> number:
        # Determines the value of the integral of the given piecewise function from a to b
        assert (self._defaultValue is not None or (a >= self._segmentBorders[0] and b <= self._segmentBorders[-1]))

        integral = zero
        x = a
        while x < b:
            y = min(self.getNextStepFrom(x), b)
            integral += (y - x) * self.getValueAt(x)
            x = y

        return integral

    def norm(self) -> number:
        # Computes the L1-norm of the function
        # requires the function to either be undefined or zero outside its borders
        # (otherwise the L1-norm would be undefined/+-infty)
        assert (self._defaultValue is None or self._defaultValue == 0)
        return self.__abs__().integrate(self._segmentBorders[0], self._segmentBorders[-1])

    def drawGraph(self, start: number, end: number):
        # Draws a graph of the function between start and end
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

    def getXandY(self, start: number, end: number) -> Tuple[List[number],List[number]]:
        # Returns two vectors x and y representing the function between start and end in the following form:
        # x = [a_0,a_1,a_1,a_2,a_2,...,a_n], y = [b_0,b_0,b_1,b_1,...,b_{n-1}]
        # such that [a_i,a_{i+1}) form a partition of [start,nextStep(end))
        # into maximal (if autoSimplify=True) intervals of constant value b_i of the function
        # i.e. for even i x[i] is the left boundary of such an interval and y[i] the value in it
        #      for odd i  x[i] is the right boundary of such an interval and y[i] the value in it
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
        f = "|" + str(round(float(self._segmentBorders[0]), 2)) + "|"
        for i in range(len(self._segmentValues)):
            # f += "-" + str(self.segmentValues[i]) + "-|" + str(self.segmentBorders[i + 1]) + "|"
            f += " " + str(round(float(self._segmentValues[i]), 2)) + " |"
            if self._segmentBorders[i + 1] < infinity:
                f += str(round(float(self._segmentBorders[i + 1]), 2)) + "|"
            else:
                f += str(self._segmentBorders[i + 1]) + "|"
        return f


# A piecewise linear function with finitely many break points
# I.e. each piece of the function is of the form f(x) = mx + t for all x in some interval (a,b]
class PWLin:
    noOfSegments: int
    autoSimplify: bool
    segmentBorders: List[number]
    segmentTvalues: List[number]
    segmentMvalues: List[number]

    def __init__(self, borders: List[number], mvalues: List[number],
                 tvalues: List[number], autoSimplify: bool = True):
        # autoSimplify=True means that adjacent segments are automatically unified whenever possible
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
        # Returns the value of the function at x
        if x < self.segmentBorders[0] or x > self.segmentBorders[-1]:
            # x is outside the range of the function
            pass # TODO: Raise error
        else:
            for i in range(0, self.noOfSegments):
                if x <= self.segmentBorders[i + 1]:
                    return self.segmentTvalues[i] + (x - self.segmentBorders[i]) * self.segmentMvalues[i]

    def getNextStepFrom(self, x: number) -> number:
        # Returns the next break point strictly after x
        if x >= self.segmentBorders[-1]:
            # TODO: Implement default value and/or better error handling
            pass
        else:
            for i in range(0, self.noOfSegments + 1):
                if x < self.segmentBorders[i]:
                    return self.segmentBorders[i]

    def drawGraph(self, start: number, end: number):
        # Draws a graph of the function between start and end
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
