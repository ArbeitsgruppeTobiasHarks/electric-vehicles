from typing import List, Dict, Tuple

from networkloading import *
from dynamic_dijkstra import dynamic_dijkstra

# For the root finding problem
from scipy import optimize
import sys, time
import numpy as np

def findShortestSTpath(s: Node, t: Node, flow: PartialFlow, time: number) -> Path:
    (arrivalTimes, realizedTimes) = dynamic_dijkstra(time, s, t, flow)
    p = Path([], t)
    while p.getStart() != s:
        v = p.getStart()
        for e in v.incoming_edges:
            if e in realizedTimes and arrivalTimes[e.node_from] + realizedTimes[e] == arrivalTimes[v]:
                p.add_edge_at_start(e)
                break
    return p


def findShortestFeasibleSTpath(time: number, s: Node, t: Node, flow:
        PartialFlow, budget: number) -> Path:
    (arrivalTimes, realizedTimes) = dynamicFeasDijkstra(time, s, t, flow, budget)
    for i in enumerate(arrivalTimes):
        print("times ", i, i[0], i[1])
    for i in enumerate(realizedTimes):
        print("times ", i, i[0], i[1])
    p = Path([], t)
    while p.getStart() != s:
        v = p.getStart()
        for e in v.incoming_edges:
            if e in realizedTimes and arrivalTimes[e.node_from] + realizedTimes[e] == arrivalTimes[v]:
                p.add_edge_at_start(e)
                break

    print("shortest ST path ", printPathInNetwork(p, flow._network))
    return p


def fixedPointUpdate(currentFlow: PartialFlow, oldPathInflows: PartialFlowPathBased, timeHorizon:
        number, alpha: float, timestepSize, priceToTime: float, commodities, verbose:
        bool) -> PartialFlowPathBased:

    newPathInflows = PartialFlowPathBased(oldPathInflows.network, oldPathInflows.getNoOfCommodities())

    # record the difference of derived times and shortest path times
    for i,comd in enumerate(commodities):
        flowValue = [None]*len(oldPathInflows.fPlus[i])
        travelTime = [None]*len(oldPathInflows.fPlus[i])
        price = [None]*len(oldPathInflows.fPlus[i])
        if False: print("Considering commodity ", i)
        newPathInflows.setPaths(i,[P for P in oldPathInflows.fPlus[i]],[PWConst([zero],[],zero) for P in oldPathInflows.fPlus[i]])
        s = newPathInflows.sources[i]
        t = newPathInflows.sinks[i]
        theta = zero
        meanIter = 0
        # We subdivide the time into intervals of length timestepSize
        k = -1
        while theta < oldPathInflows.getEndOfInflow(i):
            k += 1
            # For each subinterval i we determine the dual variable v_i
            # (and assume that it will stay constant for the whole interval)
            # if verbose: print("timeinterval [", theta, ",", theta+timestepSize,"]")

	    # Set up the update problem for each subinterval
            # Get path travel times for this subinterval
            for j,P in enumerate(oldPathInflows.fPlus[i]):
                 fP = oldPathInflows.fPlus[i][P]
                 # converting to float (optimize.root does not work with fractions)
                 travelTime[j] = float(currentFlow.pathArrivalTime(P,\
                     theta + timestepSize/2) - (theta + timestepSize/2))
                 flowValue[j] = float(fP.getValueAt(theta))
                 price[j] = P.getPrice()

            # Find integral value, ubar, of (piecewise constant) function u in this
            # subinterval
            ubar = comd[4].integrate(theta, theta + timestepSize)

            # For adjusting alpha
            # uval = ubar/timestepSize
            # alpha = uval/min(travelTime)
            # alpha = 0.5*uval*(1/min(travelTime) + 1/max(travelTime))
            # alpha = 0.5*uval*(1/max(travelTime))
            # alpha = uval/max(travelTime)
            # adjAlpha = [uval/min(travelTime) for j,_ in enumerate(flowValue)]
            # alphaList.append(adjAlpha[0])
            # print('adjAlpha: ', theta + timestepSize/2, uval, [round(i, 4) for
                # i in travelTime], [round(i,4) for i in adjAlpha])
            # adjAlpha = [alpha*flowValue[j]/(2*travelTime[j]) for j,_ in enumerate(flowValue)]
            # adjAlpha = [flowValue[j]/(2*travelTime[j]) for j,_ in enumerate(flowValue)]

            # adjmin = min([i for i in adjAlpha if i > 0])
            # adjmax = max([i for i in adjAlpha if i > 0])
            # adjAlpha = [adjmin if j == 0 else j for j in adjAlpha]
            # adjAlpha = [adjmax if j == 0 else j for j in adjAlpha]

            # TODO: Find a good starting point
            # A trivial guess: assume all terms to be positive and solve for the dual variable
            # TODO: adjust for price
            x0 = ((-sum(flowValue) + alpha*sum(travelTime))*timestepSize +
                    ubar)/(len(flowValue)*timestepSize)
            # optimize.show_options(solver='root', method='broyden1', disp=True)
            # TODO: Find a way to run optimize.root quietly
            bracketLeft = 0
            # TODO: adjust for price
            # bracketRight = abs(max(list(map(float.__sub__, list(map(lambda x: alpha*x,
                # travelTime)), flowValue)))) + ubar + 1
            bracketRight = 0
            for j,_ in enumerate(travelTime):
                bracketRight += max(bracketRight, -flowValue[j] +
                        alpha*(travelTime[j] + priceToTime*price[j] )) + ubar + 1

            # Newton's method using a routine that return value and derivative
            # TODO: pass an aggregation function value based on travel time and price
            # of paths
            # sol = optimize.root_scalar(dualVarRootFuncComb, (adjAlpha, flowValue, travelTime,
            if priceToTime == 0:
                sol = optimize.root_scalar(dualVarRootFuncComb, (alpha, flowValue, travelTime,
                    timestepSize, ubar), x0=x0, bracket=[bracketLeft, bracketRight],
                    fprime=True, method='newton')
            else:
                sol = optimize.root_scalar(dualVarRootFuncCombPrice, (alpha, flowValue,
                    travelTime, priceToTime, price, timestepSize, ubar), x0=x0,
                    bracket=[bracketLeft, bracketRight], fprime=True,
                    method='newton')

            if not sol.converged:
                print("The optimize.root_scalar() method has failed to converge due to the following reason:")
                print("\"", sol.flag, "\"")
                exit(0)
                # alpha = alpha/2 + uval/(2*max(travelTime))
                sol = optimize.root_scalar(dualVarRootFuncComb, (alpha, flowValue, travelTime, price, priceToTime,
                    timestepSize, ubar), x0=x0, bracket=[bracketLeft, bracketRight],
                    fprime=True, method='newton')
                if not sol.converged:
                    print("Adjusted alpha! The optimize.root_scalar() method has still failed to converge because:")
                    print("\"", sol.flag, "\"")
                    exit(0)
                else:
                    meanIter += sol.iterations
            else:
                meanIter += sol.iterations
            for j,P in enumerate(oldPathInflows.fPlus[i]):
                # CAUTION: Price term to be included here
                newFlowVal = max(flowValue[j] - alpha*(travelTime[j] + priceToTime*price[j]) + sol.root, 0)
                newPathInflows.fPlus[i][P].addSegment(makeNumber(theta+timestepSize), makeNumber(newFlowVal))
            theta = theta + timestepSize
        tmpVar = max(timestepSize,1/timestepSize)
        if False: print("Mean # of root.scalar() iterations ",\
                float(round(meanIter/(tmpVar*oldPathInflows.getEndOfInflow(i)),2)),\
                " for ", tmpVar*oldPathInflows.getEndOfInflow(i), " subintervals")
    return newPathInflows, alpha

def dualVarRootFunc(x, alpha, flowValue, travelTime, timestepSize, ubar):
    termSum = 0
    for j,fv in enumerate(flowValue):
        termSum += max(flowValue[j] - alpha*travelTime[j] + x, 0)*timestepSize
    return float(termSum - ubar)


def dualVarRootFuncGrad(x, alpha, flowValue, travelTime, timestepSize, ubar):
    termSum = 0
    for j,fv in enumerate(flowValue):
        if (flowValue[j] - alpha*travelTime[j] + x) > 0:
            termSum += timestepSize
    return float(termSum)


def dualVarRootFuncComb(x, alpha, flowValue, travelTime, timestepSize, ubar):
    # TODO: Read as input with each commodity
    termSum = 0
    gradTermSum = 0
    for j,fv in enumerate(flowValue):
        tmp = flowValue[j] - alpha*travelTime[j] + x
        if tmp > 0:
            termSum += tmp*timestepSize
            gradTermSum += timestepSize
    return float(termSum - ubar), float(gradTermSum)


def dualVarRootFuncCombPrice(x, alpha, flowValue, travelTime, priceToTime, price, timestepSize, ubar):
    termSum = 0
    gradTermSum = 0
    for j,fv in enumerate(flowValue):
        tmp = flowValue[j] - alpha*(travelTime[j] + priceToTime*price[j]) + x
        if tmp > 0:
            termSum += tmp*timestepSize
            gradTermSum += timestepSize
    return float(termSum - ubar), float(gradTermSum)


def differenceBetweenPathInflows(oldPathInflows : PartialFlowPathBased, newPathInflows : PartialFlowPathBased) -> number:
    assert (oldPathInflows.getNoOfCommodities() == newPathInflows.getNoOfCommodities())
    #TODO: Also check if the time horizon for both the pathInflows is same or not

    difference = zero

    for i in range(oldPathInflows.getNoOfCommodities()):
        for path in oldPathInflows.fPlus[i]:
            if path in newPathInflows.fPlus[i]:
                difference += (oldPathInflows.fPlus[i][path] + newPathInflows.fPlus[i][path].smul(-1)).norm()
            else:
                difference += oldPathInflows.fPlus[i][path].norm()
        for path in newPathInflows.fPlus[i]:
            if path not in oldPathInflows.fPlus[i]:
                difference += newPathInflows.fPlus[i][path].norm()

    return difference

# Find the sum of norm (integration) of path inflow fPlus functions
def sumNormOfPathInflows(pathInflows : PartialFlowPathBased) -> number:
    sumNorm = zero
    for i in range(pathInflows.getNoOfCommodities()):
        for path in pathInflows.fPlus[i]:
            sumNorm += (pathInflows.fPlus[i][path]).norm()
    # TODO: This should be equal to the integration of u (if required, put an assert)
    return sumNorm

# Function arguments: (network, precision, List[source node, sink node, ?], time
# horizon, maximum allowed number of iterations, verbosity on/off)
# TODO: warm-start using an available path flow?
def fixedPointAlgo(N : Network, pathList : List[Path], precision : float, commodities :
        List[Tuple[Node, Node, PWConst]], timeHorizon:
        number=infinity, maxSteps: int = None, timeLimit: int = infinity, timeStep: int = None,
        alpha : float = None, priceToTime : float = None, verbose : bool = False) -> PartialFlowPathBased:
    tStartAbs = time.time()
    step = 0

    ## Initialize:
    # Create zero-flow (PP: why?)
    pathInflows = PartialFlowPathBased(N,0)
    # TODO: Conform with LG if this can be removed
    # zeroflow = networkLoading(pathInflows)

    pathInflows = PartialFlowPathBased(N, len(commodities))
    # Initial flow: For every commodity, select the shortest s-t path and send
    # all flow along this path (and 0 flow along all other paths)
    for i,(s,t,_,_,u) in enumerate(commodities):
        flowlist = [PWConst([0, u._segmentBorders[-1]], [0], 0)] * (len(pathList[i]) - 1)
        flowlist.insert(0,u)
        pathInflows.setPaths(i, pathList[i], flowlist)

    if False: print("Starting with flow: \n", pathInflows)

    oldAbsDiffBwFlows = infinity
    oldRelDiffBwFlows = infinity
    gamma = makeNumber(1)
    alphaIter = []
    absDiffBwFlowsIter = []
    relDiffBwFlowsIter = []
    travelTime = []
    qopiIter = []  # qualityOfPathInflows
    qopiMeanIter = []  # mean of qopi
    qopiFlowIter = []  # qopii per unit flow per unit time
    qopiPathComm = []  # mean of qopi
    shouldStop = not (maxSteps is None or step < maxSteps)


    # alphaStr = ''
    # alphaStr = 'uByTmin'
    # alphaStr = 'uBy2Tmin'
    # alphaStr = 'meanUByTminTmax'
    # alphaStr = 'uByTmax'
    # alphaStr = r'($\gamma$)'
    # alphaStr = r'($\gamma\alpha$)'
    alphaStr = r'expoSmooth($\gamma$)'
    # alphaStr = r'expoSmooth($\gamma/2$)'
    # alphaStr = r'relExpoSmooth($\gamma/2$)'
    # alphaStr = r'min2ExpoSmooth($\gamma/2$)'

    totDNLTime = 0
    totFPUTime = 0
    tStart = time.time()
    iterFlow = networkLoading(pathInflows)
    totDNLTime += time.time()-tStart
    print("\nTime taken in networkLoading(): ", round(time.time()-tStart,4))

    ## Iteration:
    while not shouldStop:
        if verbose: print("STARTING ITERATION #", step)
        tStart = time.time()
        # TODO: Read priceToTime as input per commodity
        newpathInflows, alpha = fixedPointUpdate(iterFlow, pathInflows, timeHorizon, alpha,
                timeStep, priceToTime, commodities, verbose)
        totFPUTime += time.time() - tStart
        print("\nTime taken in fixedPointUpdate(): ", round(time.time()-tStart,4))

        newAbsDiffBwFlows = differenceBetweenPathInflows(pathInflows,newpathInflows)
        newRelDiffBwFlows = newAbsDiffBwFlows/sumNormOfPathInflows(pathInflows)

        # Check Stopping Conditions
        if newAbsDiffBwFlows < precision:
            shouldStop = True
            stopStr = "Attained required (absolute) precision!"
        # elif newRelDiffBwFlows < precision/10:
            # shouldStop = True
            # stopStr = "Attained required (relative) precision!"
        elif not (maxSteps is None or step < maxSteps):
            shouldStop = True
            stopStr = "Maximum number of steps reached!"

        elif (time.time() - tStartAbs > timeLimit):
            shouldStop = True
            stopStr = "Maximum time limit reached!"

        qopi = infinity
        qopiMean = infinity
        qopiFlow = infinity
        if not shouldStop:
            # Update Alpha
            if newAbsDiffBwFlows == 0:
                gamma = 0
            else:
                if step > 0: gamma = 1 - abs(newAbsDiffBwFlows - oldAbsDiffBwFlows)/(newAbsDiffBwFlows +
                        oldAbsDiffBwFlows)

            # Alpha Update Rule
            # oldalpha = alpha
            # alpha = gamma # equal to factor
            # alpha = gamma*alpha # multiplied by factor
            alpha = (gamma)*(gamma*alpha) + (1-gamma)*alpha # expo smooth using gamma
            # alpha = (0.5*gamma)*(0.5*gamma*alpha) + (1-0.5*gamma)*alpha # expo smooth using gamma/2
            # alpha = max(0.2, (0.5*gamma)*(0.5*gamma*alpha) + (1-0.5*gamma)*alpha) # expo smooth using gamma/2
            # if step > 1 and oldalpha == alpha:
                # print('Changing alpha')
                # alpha = alpha*(0.5) #step/maxSteps

            # Measure quality of the path inflows
            tStart = time.time()
            iterFlow = networkLoading(newpathInflows)
            tEnd = time.time()
            totDNLTime += tEnd - tStart
            print("\nTime taken in networkLoading(): ", round(tEnd-tStart,4))

            qopi = 0
            qopiFlow = 0
            for i,comd in enumerate(commodities):
                qopiInt = 0
                if False: print('comm ', i)
                fP = newpathInflows._fPlus[i]
                theta = zero
                oldqopi = np.zeros(len(newpathInflows._fPlus[i]))
                while theta < newpathInflows.getEndOfInflow(i):
                    tt = np.empty(len(newpathInflows._fPlus[i]))
                    for j,P in enumerate(newpathInflows._fPlus[i]):
                        tt[j] = iterFlow.pathArrivalTime(P,theta + timeStep/2) - (theta + timeStep/2)
                    tmin = min(tt)
                    fval = []
                    for j,P in enumerate(newpathInflows._fPlus[i]):
                        val = fP[P].getValueAt(theta + timeStep/2)
                        fval.append(val)
                        newqopi = (tt[j] - tmin)*val
                        qopi += newqopi
                        qopiInt += ((newqopi-oldqopi[j])/2 + min(newqopi, oldqopi[j]))*timeStep/tmin
                        oldqopi[j] = newqopi
                    theta = theta + timeStep
                # Integrate also over the interval [finalTheta, T]
                for j,_ in enumerate(newpathInflows._fPlus[i]):
                    qopiInt += ((oldqopi[j]-0)/2)*timeStep/tmin

                commFlow = comd[4].integrate(comd[4]._segmentBorders[0], comd[4]._segmentBorders[-1])
                qopiFlow += qopiInt/commFlow

            if verbose: print("Norm of change in flow (abs.) ", round(float(newAbsDiffBwFlows),4),\
                    " previous change ", round(float(oldAbsDiffBwFlows),4), " alpha ",\
                    round(float(alpha),4), ' qopi ', round(qopi,4), ' qopiFlow ', round(qopiFlow,4))
            if verbose: print("Norm of change in flow (rel.) ", round(float(newRelDiffBwFlows),4),\
                    " previous change ", round(float(oldRelDiffBwFlows),4))

            # update iteration variables
            pathInflows = newpathInflows
            oldAbsDiffBwFlows = newAbsDiffBwFlows
            oldRelDiffBwFlows = newRelDiffBwFlows

        qopiIter.append(qopi)
        qopiMeanIter.append(qopiMean)
        qopiFlowIter.append(qopiFlow)
        # alphaIter.append(alphaList)
        alphaIter.append(alpha)
        absDiffBwFlowsIter.append(newAbsDiffBwFlows)
        relDiffBwFlowsIter.append(newRelDiffBwFlows)

        step += 1

    print(stopStr)
    # Find path travel times for the final flow
    finalFlow = networkLoading(pathInflows, verbose=False)
    for i,comd in enumerate(commodities):
        fP = newpathInflows._fPlus[i]
        ttravelTime = np.empty([len(pathInflows.fPlus[i]),\
                math.ceil(pathInflows.getEndOfInflow(i)/timeStep)])
        qopiPath = np.empty([len(pathInflows.fPlus[i]),\
                math.ceil(pathInflows.getEndOfInflow(i)/timeStep)])
        theta = zero
        k = -1
        commFlow = comd[4].integrate(comd[4]._segmentBorders[0], comd[4]._segmentBorders[-1])
        while theta < pathInflows.getEndOfInflow(i):
            k += 1
            for j,P in enumerate(pathInflows.fPlus[i]):
                 ttravelTime[j][k] = finalFlow.pathArrivalTime(P,\
                     theta + timeStep/2) - (theta + timeStep/2)
            tmin = np.min(ttravelTime[:,k])
            for j,P in enumerate(pathInflows.fPlus[i]):
                val = fP[P].getValueAt(theta + timeStep/2)
                qopiPath[j][k] = (ttravelTime[j][k] - tmin)*val/(tmin*commFlow)
            theta = theta + timeStep
        travelTime.append(ttravelTime)
        qopiPathComm.append(qopiPath)
    return pathInflows, alphaIter, absDiffBwFlowsIter, relDiffBwFlowsIter,\
            travelTime, stopStr, alphaStr, qopiIter, qopiFlowIter,\
            qopiPathComm, totDNLTime, totFPUTime
