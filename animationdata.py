from __future__ import annotations
from typing import List, Tuple

from flows import *
from utilities import *

def generateCSV(f: PartialFlow, start: number, end: number, noSteps: int):
    steplength = (end-start)/(noSteps-1)
    N = f.network
    for i in range(0,noSteps):
        currentTime = i*steplength
        line = str(currentTime) + ";"
        for e in N.edges:
            # Flow segments
            curBlockStart = currentTime
            segments = ""
            while curBlockStart < currentTime+e.tau:
                curBlockEnd = currentTime+e.tau
                totflow = 0
                for i in range(1,f.noOfCommodities):
                    curBlockEnd = min(curBlockEnd,f.fMinus[(e,i)].getNextStepFrom(curBlockStart))
                    totflow += f.fMinus[(e,i)].getValueAt(curBlockStart)
                if totflow > 0:
                    block = str((currentTime+e.tau-curBlockEnd)/e.tau) + "/" + str((curBlockEnd-curBlockStart)/e.tau) + "/"
                    for i in range(1,f.noOfCommodities):
                        block += str(f.fMinus[(e,i)].getValueAt(curBlockStart)) + ","
                    segments = block.removesuffix(",") + "I" + segments
                curBlockStart = curBlockEnd
            line += segments.removesuffix("I")
            line += ";"

            # Queue-segments:
            curBlockStart = currentTime+e.tau
            segments = ""
            while curBlockStart < currentTime + e.tau + f.queues[e].getValueAt(currentTime)/e.nu:
                curBlockEnd = currentTime + e.tau + f.queues[e].getValueAt(currentTime)/e.nu
                for i in range(1, f.noOfCommodities):
                    curBlockEnd = min(curBlockEnd, f.fMinus[(e, i)].getNextStepFrom(curBlockStart))

                block =  str((curBlockEnd - curBlockStart) / e.nu) + "/"
                for i in range(1, f.noOfCommodities):
                    block += str(f.fMinus[(e, i)].getValueAt(curBlockStart)/e.nu) + ","
                segments += block.removesuffix(",") + "I"

                curBlockStart = curBlockEnd
            line += segments.removesuffix("I")
            line += ";"

        line = line.removesuffix(";")
        print(line)