from __future__ import annotations

import math

from fractions import Fraction

# A class for rational numbers including +/- infinity
# (basically just the class Fraction with +/- infinity)
class ExtendedRational(Fraction):
    def __new__(cls, numerator=0, denominator=None, *, _normalize=True):
        if not denominator is None and denominator == 0:
            self = super(Fraction, cls).__new__(cls)
            if numerator > 0:
                self.isInfinite = True
                self._numerator = 1
                return self
            elif numerator < 0:
                self.isInfinite = True
                self._numerator = -1
                return self
            else:
                # TODO
                pass
        else:
            cls.isInfinite = False
            # TODO: Solve the 'infty' case
            # if numerator == 'infty':
                # self = super(Fraction, cls).__new__(cls)
                # self.isInfinite = True
                # self._numerator = 1
                # return self
            return Fraction.__new__(cls, numerator, denominator)

    def __str__(self):
        if self.isInfinite:
            if self._numerator > 0:
                return "infty"
            else:
                return "-infty"
        else:
            return Fraction.__str__(self)

    def _richcmp(self, other, op):
        if self.isInfinite:
            if self._numerator > 0:
                return op(math.inf, other)
            else:
                return op(-math.inf, other)
        elif isinstance(other, ExtendedRational) and other.isInfinite:
            if other._numerator > 0:
                return op(self, math.inf)
            else:
                return op(self, -math.inf)
        else:
            return Fraction._richcmp(self, other, op)

    def __add__(self, other):
        if self.isInfinite and isinstance(other, ExtendedRational) and other.isInfinite:
            if self.numerator > 0 and other.numerator > 0:
                return ExtendedRational(1,0)
            elif self.numerator < 0 and other.numerator < 0:
                return ExtendedRational(-1,0)
            else:
                pass # TODO: Should throw an error as we cannot add +infty and - infty
        elif self.isInfinite:
            return self
        elif isinstance(other, ExtendedRational) and other.isInfinite:
            return other
        else:
            return Fraction.__add__(self,other)


    def __mul__(self, other):
        if self.isInfinite:
            if other > 0:
                return self
            elif other < 0:
                return ExtendedRational(-self.numerator,0)
            else:
                pass
        elif isinstance(other, ExtendedRational) and other.isInfinite:
            if self > 0:
                return other
            elif self < 0:
                return ExtendedRational(-other.numerator,0)
            else:
                pass # TODO: Should throw an error as we can't multiply 0 and +/-infty
        else:
            return Fraction.__mul__(self,other)

    def __float__(self) -> float:
        if self.isInfinite:
            if self.numerator > 0:
                return float(math.inf)
            elif self.numerator < 0:
                return float(-math.inf)
            else:
                return float(0/0)
        else:
            return Fraction.__float__(self)



# If exact=0, we use floating point numbers for all calculaion
# Otherwise we use extendedRationals
#TODO: make it user-defined (command line argument)
exact = 0
# The type has to be set here (outside the if-clause)
# Otherwise the typing system breaks (at least the way we do it right now)
# However, typing should be irrelevant for the correctness of the code itself anyway
# so it shouldn't matter if we forget to change this
number = float
#number = ExtendedRational

if exact == 0:
    infinity = math.inf
    zero = 0.0
    def makeNumber(n):
        return float(n)
    numPrecision = 0.00001
    def IsZero(n):
        return -numPrecision <= n <= numPrecision
else:
    infinity = ExtendedRational(1,0)
    zero = ExtendedRational(0)
    def makeNumber(n):
        return ExtendedRational(n)
    numPrecision = zero
    def isZero(n):
        return n == zero

