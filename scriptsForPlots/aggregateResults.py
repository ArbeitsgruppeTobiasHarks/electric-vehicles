from __future__ import annotations

import heapq, math
from typing import List, Dict

from network import *
from flows import *
from utilities import *

from networkloading import *
from fixedPointAlgorithm import *
import os
import time, sys
import numpy as np
import matplotlib.pyplot as plt
import re
# for axhline
import matplotlib.transforms as transforms

# TODO: Use pickle to store class objects
data = np.load(sys.argv[1], allow_pickle=True);
fname = os.path.splitext(os.path.split(sys.argv[1])[1])[0]
print(re.split('[_]', fname), len(re.split('[_]', fname)))
if len(re.split('[_]', fname)) > 9:
    [insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,priceToTime,numThreads,alphaStr] = re.split('[_]', fname)
else:
    if len(re.split('[_]', fname)) > 8:
        [insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,numThreads,alphaStr] = re.split('[_]', fname)
    else:
        [insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,alphaStr] = re.split('[_]', fname)
runTime = round(float(data['time']),2)
print("Data: ", data.files)
print("Termination message: ", data['stopStr'])
print("Time taken: ", runTime)
print("Iterations: ", len(data['alphaIter']))

commodities = data['commodities']
energyBudget = commodities[0][2]

# print(fname,insName,timeHorizon,maxIter,precision,alpha,timeStep)
# print("f: ", f, type(f), f.size, f.shape, f[()])
# print("fPlus: ")

colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'orange',
        'darkviolet','magenta','darkorchid','darkgreen', 'c']
# linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'offset',\
        # 'on-off-dash-seq', '-' , '--' , '-.' , ':' , 'None' ,\
        # ' ' , '']
fontsizes = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted', '-', '--',\
        '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot',\
        'dotted']
locs = ["upper left", "lower left", "center left", "center right", "upper right"]

G = data['G']
f = data['f']

## Multicommodity Plots
#-----------------
# QoPI and FlowDiff per iteration
#-----------------
# figD, axsD = plt.subplots(1)
# axsQ = axsD.twinx()

absDiffBwFlowsIter = data['absDiffBwFlowsIter']
qopiFlowIter = data['qopiFlowIter']
alphaIter = data['alphaIter']
relDiffBwFlowsIter = data['relDiffBwFlowsIter']
qopiIter = data['qopiIter']

#----------------
# ENERGY PROFILES
#----------------
# bmax = 30
# figB, axsB = plt.subplots(1)
# yBsum1 = []
# for c,p in enumerate(f[()].fPlus):
    # print('Comm.',c)
    # u = sum([f[()].fPlus[c][p1].integrate(0,1) for p1 in f[()].fPlus[c]])
    # # bmax = 1 + max([2*p.getNetEnergyConsump() for p in f[()].fPlus[c]])
    # # bmax =13
    # # bmin = min([p.getNetEnergyConsump() for p in f[()].fPlus[c]])
    # # for p in f[()].fPlus[c]:
      # # # print(p, p.getNetEnergyConsump())
    # # print('bmin ',bmin)
    # yBsum = []
    # # yBsum1 = []
    # tt = data['travelTime']
    # xB1 = [float(timeStep)/2 + x*float(timeStep) for x in range(int((len(tt[0][0])-0)))]

    # for i,p1 in enumerate(f[()].fPlus[c]):
        # x,y = f[()].fPlus[c][p1].getXandY(0,f[()].getEndOfInflow(c))
        # # yB = [p1.getNetEnergyConsump()*v/u for v in y]
        # y1 = [f[()].fPlus[c][p1].getValueAt(i) for i in xB1]
        # yB1 = [p1.getNetEnergyConsump()*v/u for v in y1]
        # # print('x', x)
        # # print('y', y)
        # # print('xB1', xB1)
        # # print('yB1', yB1)
        # # TODO: set a compact logic for the following line
        # if (len(y) > 2) or (len(y) > 1 and y[0]>0):
            # xB = x
            # # [lambda: value_false, lambda: value_true][<test>]()
            # # yBsum = [lambda:yB, lambda:[yBsum[i]+yB[i] for i,_ in enumerate(yB)]][len(yBsum)>0]()
            # yBsum1 = [lambda:yB1, lambda:[yBsum1[i]+yB1[i] for i,_ in enumerate(yB1)]][len(yBsum1)>0]()
            # # print('i', i, p1.getNetEnergyConsump())
            # # print(yB)
            # # axsB.plot(x,yB,label='w%d'%(i+1), color=colors[i+1], linestyle=linestyles[1], linewidth=10)
        # # print('i', i)
        # # print('y', len(y), y, p1.getNetEnergyConsump())
        # # print('yB', len(yB), yB)
        # print('yBsum1', len(yBsum1), yBsum1)
    # print('yBsum1', len(yBsum1), yBsum1)

      # # a,b = [int(c) for c in x],[int(c) for c in y]
      # # print("i: ", i,a,b)
      # # axsB.plot(x,yB,label='Total', color=colors[i], linestyle=linestyles[1],
              # # linewidth=10)
# print('yBsum1', yBsum1)
# axsB.plot(xB1,yBsum1,label=r'no r/c', color='cyan', linestyle=linestyles[1],
      # linewidth=10)
# # axsB.legend(loc='best', fontsize=80, frameon=False)
# axsB.set_xlabel(r'time ($\theta$)', fontsize=80)

# # Temporary: uncomment if y-ticks and y-labels are not needed
# # axsB.set_ylabel(r'Battery Consump. / Unit Flow', fontsize=80)

# # axsB.set_ylim([0, bmax])
# # axsB.set_ylim(bottom=float(energyBudget))

# plt.setp(axsB.get_xticklabels(), fontsize=80)
# plt.setp(axsB.get_yticklabels(), fontsize=80)

#----------------
# ENERGY PROFILES
#----------------
if True:
    bmax = 30
    figB, axsB = plt.subplots(1)
    for j in range(len(sys.argv)-1):
        # print('j',j,len(sys.argv))
        data1 = np.load(sys.argv[1+j], allow_pickle=True);
        f1 = data1['f']
        fname1 = os.path.splitext(os.path.split(sys.argv[1+j])[1])[0]
        print(re.split('[_]', fname1))
        [insName1,timeHorizon1,maxIter1,timeLimit1,precision1,alpha1,timeStep1,priceToTime,numThreads,alphaStr1] = re.split('[_]', fname1)

        yBsum1 = []
        for c,p in enumerate(f[()]._fPlus):
            u = sum([f[()]._fPlus[c][p1].integrate(0, 1) for p1 in f[()]._fPlus[c]])
            yBsum = []
            tt = data1['travelTime']
            xB1 = [float(timeStep)/2 + x*float(timeStep) for x in range(int((len(tt[0][0])-0)))]
            for i,p1 in enumerate(f1[()]._fPlus[c]):
                x,y = f1[()]._fPlus[c][p1].getXandY(0, f[()].getEndOfInflow(c))
                y1 = [f1[()]._fPlus[c][p1].getValueAt(i) for i in xB1]
                # yB = [p1.getNetEnergyConsump()*v/u for v in y]
                yB1 = [p1.getNetEnergyConsump()*v/u for v in y1]
                # print('x', x)
                # print('y', y)
                # print('xB1', xB1)
                # print('yB1', yB1)
                if (len(y) > 2) or (len(y) > 1 and y[0]>0):
                  xB = x
                  # [lambda: value_false, lambda: value_true][<test>]()
                  yBsum1 = [lambda:yB1, lambda:[yBsum1[i]+yB1[i] for i,_ in
                      enumerate(yB1)]][len(yBsum1)>0]()
                  # print('i', i, p1.getNetEnergyConsump())
                  # print(yB)
                  # axsB.plot(x,yB,label='w%d'%(i+1), color=colors[i+1], linestyle=linestyles[1], linewidth=10)
        print('yBsum1', yBsum1)
        if j == 0:
            lname = r'no r/c'
            clr = 'cyan'
        else:
            lname = r'$\widetilde{\lambda}_i = %d$'%int(priceToTime)
            lname = re.split('0',insName1)[1]
            # lname = insName1
            if not lname: lname = 'R3'
            clr = colors[j]
        axsB.plot(xB1,yBsum1,label=lname, color=clr, linestyle=linestyles[1],
              linewidth=10)
        # axsB.plot(xB1,yBsum1,label=r'with recharging', color=colors[j], linestyle=linestyles[1],
              # linewidth=10)
        # axsB.plot(xB,[float(energyBudget) for i in xB], label=r'$b^{max}$', color=colors[-2],
                # linestyle='dotted', linewidth=10)
axsB.legend(loc='best', fontsize=80, frameon=False, ncol=3)
axsB.set_xlabel(r'time ($\theta$)', fontsize=80)

axsB.set_ylabel(r'Battery Consump. / Unit Flow', fontsize=80)
# axsB.set_ylim([0, bmax])
# axsB.set_ylim(bottom=float(15),top=float(25))
# axsB.set_ylim([4, 5.5])
# axsB.set_ylim([11, 25])
axsB.set_ylim([20, 40])
# axsB.set_ylim(bottom=float(energyBudget))
# axsB.set_ylim([float(energyBudget), bmax])

plt.setp(axsB.get_xticklabels(), fontsize=80)
plt.setp(axsB.get_yticklabels(), fontsize=80)

#-----------------------
# MIN./MEAN TRAVEL TIMES
#-----------------------
if True:
    print('\nTimes\n')
    figTC, axsTC = plt.subplots(1)
    for j in range(len(sys.argv)-1):
        # print('j',j,len(sys.argv))
        data1 = np.load(sys.argv[1+j], allow_pickle=True);
        g = data1['f']
        fname1 = os.path.splitext(os.path.split(sys.argv[1+j])[1])[0]
        print('\n',re.split('[_]', fname1))
        [insName1,timeHorizon1,maxIter1,timeLimit1,precision1,alpha1,timeStep1,priceToTime,numThreads,alphaStr1] = re.split('[_]', fname1)
        # if j > 0: lname = re.split('0',insName1)[1]
        # if not lname: lname = 'R3'
        # lname = insName1

        tmin = []
        tmax = []
        tsum = []
        tavg = []
        for c,_ in enumerate(f[()]._fPlus):
            print('\nComm.',c)
            tt = data1['travelTime']
            # print(tt)
            # print(tt[0][0], len(tt), len(tt[0]), len(tt[0][0]))
            # print('tt ', tt[c], len(tt[c]), len(tt[c][0]))

            x = [float(timeStep)/2 + x*float(timeStep) for x in\
                range(int((len(tt[0][0])-0)))]
            # ttmax = np.amax(tt[c]) + 1
            tminc = []
            tmaxc = []
            tcavg = []
            for p,q in enumerate(g[()]._fPlus[c]):
            # for p in range(len(tt[c])):
                y = tt[c][p]
                flow = [g[()]._fPlus[c][q].getValueAt(i) for i in x]
                if max(flow) > 0:
                    # tminc = [lambda:y, lambda:[min(tminc[i],y[i]) for i,_ in enumerate(y)]][len(tminc)>0]()
                    print('p%d'%p, [round(float(i),4) for i in y])
                    print('flow', [round(float(i),4) for i in flow])
                    # print('tminc', [round(float(i),4) for i in tminc])
            for t,k in enumerate(x):
                # pospaths = [p for p in g[()].fPlus[c] if max([g[()].fPlus[c][p].getValueAt(i) for i in x]) > 0]
                # print('ttcqt', [tt[c][q][t] for q,r in enumerate(pospaths)])
                # tminc.append(min([tt[c][q][t] for q,r in enumerate(pospaths) if g[()].fPlus[c][r].getValueAt(k) > 0]))
                val = math.inf
                maxval = 0
                sumval = 0
                sumcnt = 0
                for p,q in enumerate(g[()]._fPlus[c]):
                    if g[()]._fPlus[c][q].getValueAt(k) > 0:
                        val = min(val, tt[c][p][t])
                        maxval = max(val, tt[c][p][t])
                        sumval += tt[c][p][t]
                        sumcnt += 1
                tminc.append(val)
                tmaxc.append(maxval)
                tcavg.append(sumval/sumcnt)
            # print('\ntmin', tmin)
            print('\ntminc', tminc)
            print('tmaxc', tmaxc)
            # print('x', x)
            # print('len tmin', len(tmin), len(x))
            # tmin = [lambda:y, lambda:[min(tmin[i],tminc[i]) for i,_ in enumerate(tminc)]][c>0]()
            tmin = [lambda:tminc, lambda:[min(tmin[i],tminc[i]) for i,_ in enumerate(tminc)]][c>0]()
            tmax = [lambda:tmaxc, lambda:[max(tmax[i],tmaxc[i]) for i,_ in enumerate(tmaxc)]][c>0]()
            # tsum = [lambda:y, lambda:[(tsum[i] + tminc[i]) for i,_ in enumerate(y)]][c>0]()
            # tsum = [lambda:tmin, lambda:[(tsum[i] + tmin[i]) for i,_ in enumerate(tmin)]][c>0]()
            tsum = [lambda:tmax, lambda:[(tsum[i] + tmax[i]) for i,_ in enumerate(tmax)]][c>0]()
            tavg = [lambda:tcavg, lambda:[(tavg[i] + tcavg[i]) for i,_ in enumerate(tcavg)]][c>0]()
        # Mean (over commodities) min. travel times
        tmean = [t/(c+1) for t in tsum]
        tavg = [t/(c+1) for t in tavg]
        print('tmean', tmean)

        if j == 0:
            lname = r'no r/c'
            clr = 'cyan'
        else:
            lname = r'$\widetilde{\lambda}_i = %d$'%int(priceToTime)
            lname = re.split('0',insName1)[1]
            # lname = insName1
            if not lname: lname = 'R3'
            clr = colors[j]
        # axsTC.plot(x,tmin,label=lname, color=colors[j], linestyle=linestyles[1], linewidth=10)
        # axsTC.plot(x,tmax,label=lname, color=colors[j], linestyle=linestyles[1], linewidth=10)
        # axsTC.plot(x,tmean,label=lname, color=colors[j], linestyle=linestyles[1], linewidth=10)
        # if j==0 or (j>=7 and j<=7):
        axsTC.plot(x,tavg,label=lname, color=clr, linestyle=linestyles[1], linewidth=10)
        # axsTC.set_ylabel(r'Min. Travel Time', fontsize=80)
        # axsTC.set_ylabel(r'Max. Travel Time', fontsize=80)
        # axsTC.set_ylabel(r'Mean Min. Travel Time', fontsize=80)
        # axsTC.set_ylabel(r'Mean Max. Travel Time', fontsize=80)
        axsTC.set_ylabel(r'Mean Travel Time', fontsize=80)
        axsTC.legend(loc='best', fontsize=80, frameon=False, ncol=3)
        axsTC.set_xlabel(r'time ($\theta$)', fontsize=80)
        axsTC.set_ylim([400, 2600])
        plt.setp(axsTC.get_yticklabels(), fontsize=80)
        plt.setp(axsTC.get_xticklabels(), fontsize=80)

        print('\ntmin', [round(float(i),4) for i in tmin])
        print('tmax', [round(float(i),4) for i in tmin])
        print('tsum', [round(float(i),4) for i in tsum])
        print('tmean', [round(float(i),4) for i in tmean])

##################################
MIN. TRAVEL TIMES PER COMMODITY
if False:
    figTC, axsTC = plt.subplots(1)
    for c,p in enumerate(f[()]._fPlus):
        print('Comm.',c)
        g = data['f']
        tt = data['travelTime']
        # print(tt)
        # print(tt[0][0], len(tt), len(tt[0]), len(tt[0][0]))
        # print(tt[c], len(tt[c]), len(tt[c][0]))

        x = [float(timeStep)/2 + x*float(timeStep) for x in\
            range(int((len(tt[0][0])-0)))]
        tminc = []
        # for p,q in enumerate(g[()].fPlus[c]):
            # y = tt[c][p]
            # flow = [g[()].fPlus[c][q].getValueAt(i) for i in x]
            # if max(flow) > 0:
                # print('p%d'%p, [round(float(i),4) for i in y])
                # print('flow', [round(float(i),4) for i in flow])
        for t,k in enumerate(x):
            val = math.inf
            for p,q in enumerate(g[()]._fPlus[c]):
                if g[()]._fPlus[c][q].getValueAt(k) > 0:
                    val = min(val, tt[c][p][t])
            tminc.append(val)
        print('\ntminc', tminc)
        print('x', x)
        axsTC.plot(x,tminc,label='comm%d'%c, color=colors[c], linestyle=linestyles[1],
                linewidth=10)
        axsTC.legend(loc='best', fontsize=80, frameon=False, ncol=2)
        axsTC.set_xlabel(r'time ($\theta$)', fontsize=80)
        axsTC.set_ylabel(r'Min. Travel Time', fontsize=80)
        axsTC.set_ylim([400, 2500])
        plt.setp(axsTC.get_yticklabels(), fontsize=80)
        plt.setp(axsTC.get_xticklabels(), fontsize=80)
##################################

figs = []
for i in plt.get_fignums():
    mng = plt.figure(i).canvas.manager
    mng.full_screen_toggle()
    # Save reference to the figures, else they are deleted after plt.show()
    figs.append(plt.figure(i))

print('-------')
print('Summary')
print('-------')
print("Termination message: ", data['stopStr'])
print("\nAttained DiffBwFlows (abs.): %.5f"%absDiffBwFlowsIter[-2])
print("Attained DiffBwFlows (rel.): %.5f"%relDiffBwFlowsIter[-2])
print("\nAttained QoPI (abs.): %.5f"%qopiIter[-2])
# print("Attained QoPI (mean): %.4f"%qopiMeanIter[-2])
print("Attained QoPI (per unit flow): %.5f"%qopiFlowIter[-2])
print("\nIterations : ", len(alphaIter))
print("Elapsed wall time: ", runTime)


# Save figures
dirname = os.path.expanduser('./figures')
# fname1 = fname + '_comm%d'%c
# print(fname)
plt.show()
for i,fig in enumerate(figs):
    figname = os.path.join(dirname, fname)
    if i == 5:
        figname += '_combQoPIFlowDiff'
    elif i == 0:
        figname += '_combEnerProfs'
    elif i == 1:
        # figname += '_commTravTimes'
        # figname += '_commMinTravTimes'
        # figname += '_commMaxTravTimes'
        figname += '_commMeanTravTimes'
        # figname += '_commMeanMaxTravTimes'
        # figname += '_commMeanMinTravTimes'
    # elif i == 3:
        # figname += '_travTimes'
    # elif i == 4:
        # figname += '_qopiWalks'
    figname += '.png'
    print("\noutput saved to file: %s"%figname)
    if i == 0 or 1:
        fig.savefig(figname, format='png', dpi=fig.dpi, bbox_inches='tight')
plt.close()
