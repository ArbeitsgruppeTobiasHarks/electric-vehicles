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

# TODO: Use pickle to store class objects
data = np.load(sys.argv[1], allow_pickle=True);
fname = os.path.splitext(os.path.split(sys.argv[1])[1])[0]
print(re.split('[_]', fname))
[insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,energyBudget,alphaStr] = re.split('[_]', fname)
runTime = round(float(data['time']),2)
print("Data: ", data.files)
print("Termination message: ", data['stopStr'])
print("Time taken: ", runTime)
print("Iterations: ", len(data['alphaIter']))
# print("QoPI (per unit flow): ", data['qopiFlowIter'])
# print(fname,insName,timeHorizon,maxIter,precision,alpha,timeStep)
# print("f: ", f, type(f), f.size, f.shape, f[()])
# print("fPlus: ")

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange',
        'darkviolet','magenta','darkorchid','darkgreen']
# linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'offset',\
        # 'on-off-dash-seq', '-' , '--' , '-.' , ':' , 'None' ,\
        # ' ' , '']
fontsizes = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted', '-', '--',\
        '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot',\
        'dotted']
locs = ["upper left", "lower left", "center left", "center right", "upper right"]

f = data['f']
G = data['G']

fig, axs = plt.subplots(3)
for c,p in enumerate(f[()]._fPlus):
    # print('comm:%d'%c, f[()].fPlus[c], f[()].getEndOfInflow(c))
    # exit(0)
    print('Commodity: %d'%c)

    # Making figures for each commodity

    # Set fontsize
    params = {'legend.fontsize': 'x-large',
            # 'figure.figsize': (15, 5),
            'axes.labelsize': 'xx-large',
            'axes.titlesize': 'xx-large',
            'xtick.labelsize':'xx-large',
            'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    # print(plt.rcParams.keys())
    # plt.rc('legend',fontsize=20) # using a size in points
    # plt.rc('legend',fontsize=fontsizes[5]) # using a named size
    alphaStr = data['alphaStr']

    # TODO: Update and check the code to cater to each commodity
    # Final Results
    #-----------------
    # Path Inflows
    #-----------------
    fig.suptitle(r'ins=[%s], $T$=[%s], $maxIter$=[%s], $\epsilon$=[%s], $\alpha_0$=[%s],'\
    r' timeStep=[%s], $\alpha-$update rule: [%s]''\n runTime=%.2f'%(insName,timeHorizon,maxIter,precision,alpha,\
    timeStep,alphaStr,runTime), fontsize='xx-large')
    # r' timeStep=[%s], runTime=%.2f'%(insName,timeHorizon,maxIter,precision,alpha,\
    # timeStep,runTime), fontsize='xx-large')

    k = -1
    k += 1
    figB, axsB = plt.subplots(1)
    u = sum([f[()]._fPlus[c][p1].integrate(0, 1) for p1 in f[()]._fPlus[c]])
    fmax = 0.2 + u
    bmax = 1 + max([2 * p.getNetEnergyConsump() for p in f[()]._fPlus[c]])
    bmin = min([2 * p.getNetEnergyConsump() for p in f[()]._fPlus[c]])
    print('bmin ',bmin)
    yBsum = []
    # for p in f[()].fPlus[c]:
        # fmax += f[()].fPlus[c][p].integrate(0,1)
    for i,p1 in enumerate(f[()]._fPlus[c]):
        # k += 1
        #TODO: Determine the right end of x-axis for plots
        # x,y = f[()].fPlus[0][p].getXandY(0,20)
        x,y = f[()]._fPlus[c][p1].getXandY(0, f[()].getEndOfInflow(c))
        yB = [p1.getNetEnergyConsump()*2*v/u for v in y]
        if (len(y) > 2):
            xB = x
            # [lambda: value_false, lambda: value_true][<test>]()
            yBsum = [lambda:yB, lambda:[yBsum[i]+yB[i] for i,_ in enumerate(yB)]][len(yBsum)>0]()
            axsB.plot(x,yB,label='path%d'%(i), color=colors[i], linestyle=linestyles[1], linewidth=10)
        # print('i', i)
        # print('y', len(y), y, p1.getNetEnergyConsump())
        # print('yB', len(yB), yB)
        # print('yBsum', len(yBsum), yBsum)

        # a,b = [int(c) for c in x],[int(c) for c in y]
        # print("i: ", i,a,b)
        # if max(y)>0:
        axs[k].plot(x,y,label='path%d'%i, color=colors[i], linestyle=linestyles[i])
        # axs[k].plot(x,y,label='path%d'%i, linestyle=linestyles[i])
        # axs[k].legend()
        # else:
            # k -= 1
        # axsB.plot(x,yB,label='Total', color=colors[i], linestyle=linestyles[1],
                # linewidth=10)
    # print('yBsum', yBsum)
    # exit(0)
    axsB.plot(xB,yBsum,label='Total', color=colors[-3], linestyle=linestyles[2],
            linewidth=10)
    axsB.plot(xB,[2*float(energyBudget) for i in xB], label=r'\b_{max}', color=colors[-2], linestyle=linestyles[2],
            linewidth=10)
    axs[k].legend(loc='upper right')
    axs[k].set_title('Path Inflows', fontsize='xx-large')
    # plt.show()
    axsB.legend(loc='best', fontsize=80, frameon=False, ncol=2)
    axsB.set_xlabel(r'time ($\theta$)', fontsize=80)

    # Temporary: uncomment if y-ticks and y-labels are not needed
    axsB.set_ylabel(r'Battery Cons. Per Unit Flow', fontsize=80)

    axsB.set_ylim([0, bmax])

    plt.setp(axsB.get_xticklabels(), fontsize=80)
    plt.setp(axsB.get_yticklabels(), fontsize=80)

    #------------------------
    # QoPI
    #------------------------
    tt = data['travelTime']
    qopi = data['qopiPathComm']
    qopiFlow = data['qopiFlowIter']
    # print(tt)
    # print(tt[0][0], len(tt), len(tt[0]), len(tt[0][0]))
    # print(tt[c], len(tt[c]), len(tt[c][0]))

    x = [float(timeStep)/2 + x*float(timeStep) for x in\
        range(int((len(tt[0][0])-0)))]
    # print(timeHorizon, timeStep, x)
    k += 1
    # for i in range(len(tt)):
        # k += 1
    ttmax = np.amax(tt[c]) + 1
    qmax = np.amax(qopi[c]) + np.amin(qopi[c][np.nonzero(qopi[c])])
    qmax = 0.001

    for p in range(len(tt[c])):
        y = tt[c][p]
        yQ = qopi[c][p]
        # print(y)
        axs[k].plot(x,y,label='path%d'%p, color=colors[p], linestyle=linestyles[p])

        axsQ.set_ylim([0, qmax])
        # axs[k].plot(x,y,label='path%d'%p, linestyle=linestyles[p])
    axs[k].legend(loc='best')
    axs[k].set_xlabel('time', fontsize='xx-large')
    axs[k].set_title('Travel Times', fontsize='xx-large')
    # plt.show()

    plt.setp(axsQ.get_xticklabels(), fontsize=80)
    plt.setp(axsQ.get_yticklabels(), fontsize=80)
    # axsT.set_title('Travel Times', fontsize='xx-large')

    #-----------------
    # Alpha and FlowDiff per iteration
    #-----------------
    alphaIter = data['alphaIter']
    absDiffBwFlowsIter = data['absDiffBwFlowsIter']
    relDiffBwFlowsIter = data['relDiffBwFlowsIter']
    qopiIter = data['qopiIter']
    # qopiMeanIter = data['qopiMeanIter']
    qopiFlowIter = data['qopiFlowIter']
    # a,b = [round(float(c),2) for c in alphaIter],[round(float(c),2) for c in diffBwFlowsIter]
    # print(a,b)
    x = [x+1 for x in range(len(alphaIter))]

    k += 1
    axs[k].plot(x,alphaIter,label=r'$\alpha$', color=colors[0], linestyle=linestyles[0])
    axs2 = axs[k].twinx()
    axs2.plot(x,absDiffBwFlowsIter,label=r'$\Delta$ f', color=colors[1], linestyle=linestyles[1])
    axs2.plot(x,relDiffBwFlowsIter,label=r'($\Delta$ f / f)', color=colors[2], linestyle=linestyles[1])
    axs2.plot([1,len(alphaIter)], [float(precision), float(precision)],label=r'$\epsilon$',\
            color=colors[3], linestyle=linestyles[2])
    axs2.legend(loc=locs[3], fontsize='x-large')
    # axs[k].set_xlabel('iteration', fontsize='xx-large')
    # axs[k].set_title(r'$\alpha$ and $\Delta f^{k}$', fontsize='xx-large')
    # plt.show()
   # TODO: Avoid this hardcoding
    # axs[2].legend(loc=locs[2])
    axs[k].legend(loc=locs[1], fontsize='x-large')

    k += 1
    # axs[k].plot(x,qopiIter,label='QoPI', color=colors[3], linestyle=linestyles[1])
    # axs[k].plot(x,qopiMeanIter,label='QoPIMean', color=colors[4], linestyle=linestyles[1])
    axs[k].plot(x,qopiFlowIter,label='QoPI (per unit flow)', color=colors[4], linestyle=linestyles[1])
    axs[k].set_xlabel('iteration', fontsize='xx-large')
    axs[k].legend(fontsize='x-large')

    # Set label and xtick sizes for axes
    for i in range(len(axs)):
        # axs[i].legend(fontsize='x-large')
        plt.setp(axs[i].get_xticklabels(), fontsize='x-large')
        plt.setp(axs[i].get_yticklabels(), fontsize='x-large')

    plt.ylim(bottom=0)

    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    figs = []
    for i in plt.get_fignums():
        # print(plt.gcf, plt.fignum_exists(i), i)
        mng = plt.figure(i).canvas.manager
        mng.full_screen_toggle()
        # Save reference to the figures, else they are deleted after plt.show()
        figs.append(plt.figure(i))
    # plt.show()
    # plt.show(block=False)
    # plt.pause(.1)
    # plt.close()

    print('-------')
    print('Summary')
    print('-------')
    print("Termination message: ", data['stopStr'])
    print("\nAttained DiffBwFlows (abs.): %.4f"%absDiffBwFlowsIter[-2])
    print("Attained DiffBwFlows (rel.): %.4f"%relDiffBwFlowsIter[-2])
    print("\nAttained QoPI (abs.): %.4f"%qopiIter[-2])
    print("Attained QoPI (per unit flow): ", qopiFlowIter[-2])
    # print("Attained QoPI (mean): %.4f"%qopiMeanIter[-2])
    print("\nIterations : ", len(alphaIter))
    print("Elapsed wall time: ", runTime)

    # print("\noutput saved to file: %s"%figname)

    # Save figures
    dirname = os.path.expanduser('./figures')
    fname1 = fname + '_comm%d'%c
    # print(fname)
    plt.show()
    # for i,fig in enumerate(figs):
        # figname = os.path.join(dirname, fname1)
        # if i == 1:
            # figname += '_pathFlows'
        # elif i == 2:
            # figname += '_enerProfs'
        # elif i == 3:
            # figname += '_travTimes'
        # elif i == 4:
            # figname += '_qopiPaths'
        # figname += '.png'
        # print("\noutput saved to file: %s"%figname)
        # if i == 4:
            # fig.savefig(figname, format='png', dpi=fig.dpi, bbox_inches='tight')
    plt.close()
