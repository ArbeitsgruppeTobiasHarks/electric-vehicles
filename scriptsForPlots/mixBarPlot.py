import numpy as np
import matplotlib.pyplot as plt
import os
# set width of bar
barWidth = 0.125
# fig = plt.subplots(figsize =(12, 8))
fig,axs = plt.subplots(1)

# Set height of bars (unfortunately hardcoded currently due to time constraints!)
m0 	= [778.25,	0,	928.8,	0,	1075.08,	0,	1001.45,	0]
m20	= [1016.83,	769.25,	941.35,	945.87,	1096.75,	1101.07,	1020.8,	1027.43]
m40	= [1058.53,	766.61,	984.24,	977.89,	1143.54,	1143.89,	1072.95,	1054.18]
m60	= [1120.9,	768.48,	1043.79,	937.28,	1213.79,	1210.81,	1143.33,	1021.6]
m80	= [1183.73,	769.45,	1107.57,	914.25,	1389.57,	1163.29,	1318.1,	902.16]
m100 = [	0,	1215.49,	0,	1140.39,	0,	1583.51,	0,	1513.24]

# Set position of bar on X axis
br1 = np.arange(len(m0))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]

# Make the plot
plt.bar(br1, m0, color ='c', width = barWidth,
		edgecolor ='grey', label ='0%')
plt.bar(br2, m20, color ='b', width = barWidth,
		edgecolor ='grey', label ='20%')
plt.bar(br3, m40, color ='g', width = barWidth,
		edgecolor ='grey', label ='40%')
plt.bar(br4, m60, color ='r', width = barWidth,
		edgecolor ='grey', label ='60%')
plt.bar(br5, m80, color ='m', width = barWidth,
		edgecolor ='grey', label ='80%')
plt.bar(br6, m100, color ='y', width = barWidth,
		edgecolor ='grey', label ='100%')

# Adding Xticks
plt.xticks([r + barWidth for r in range(len(m0))],
        ['comm0-C','comm0-EV','comm1-C','comm1-EV','comm2-C','comm2-EV','comm3-C','comm3-EV'],
        fontsize=30, rotation = 30)
axs.set_ylabel(r'Mean Max. Travel Time', fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=30, frameon=False, ncols=3)

figs = []
for i in plt.get_fignums():
    mng = plt.figure(i).canvas.manager
    mng.full_screen_toggle()
    figs.append(plt.figure(i))

# Save figures
dirname = os.path.expanduser('./figures')
plt.show()
fname=''
for i,fig in enumerate(figs):
    figname = os.path.join(dirname, fname)
    if i == 0:
        figname += 'mixEVmultiBarPlot'
    figname += '.png'
    print("\noutput saved to file: %s"%figname)
    if i == 0:
        fig.savefig(figname, format='png', dpi=fig.dpi, bbox_inches='tight')


plt.savefig('./figures/mixEVmultiBarPlot', format='png', dpi=fig.dpi, bbox_inches='tight')
plt.close()
