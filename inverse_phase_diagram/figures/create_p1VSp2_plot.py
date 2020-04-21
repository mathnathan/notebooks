import matplotlib.pyplot as plt
import numpy as np
import fillplots as fp


# Initialize boundaries individually, so that they are recognized as
# one line rather than line per region.
negdet = fp.boundary(lambda x: -1.0/x, (0.1,5))
negdet.config.line_args = {'color': 'r', 'label': "$\Delta=0$", 'lw': 2}

posdet = fp.boundary(lambda x: -1.0/x, (-5,-0.1))
posdet.config.line_args = {'color': 'r', 'lw': 2}

center = fp.boundary(lambda x: -x, (-5,5))
center.config.line_args = {'color': 'k', 'label': "$\\tau=0$", 'lw': 2}

line1 = fp.boundary(lambda x: x+2, (-5,5))
line1.config.line_args = {'color': 'b', 'label': "$\\tau^2-4\Delta=0$", 'lw': 2}

line2 = fp.boundary(lambda x: x-2, (-5,5))
line2.config.line_args = {'color': 'b', 'lw': 2}

plotter = fp.Plotter([
    [(center,)], # Unstable Node
    [(center, True)], # Stable Node
    [(posdet,)], # Saddle
    [(negdet, True)], # Saddle
    [(line1, True), (line2,), (center,)], # Unstable Spiral
    [(line1, True), (line2,), (center, True)], # Stable Spiral
], xlim=(-5,5), ylim=(-5,5))

plotter.regions[0].config.fill_args['facecolor']='lightcoral' # unstable node
plotter.regions[4].config.fill_args['facecolor']='coral' # unstable spiral
plotter.regions[1].config.fill_args['facecolor']='cyan' # stable node
plotter.regions[5].config.fill_args['facecolor']='darkcyan' # stable spiral
plotter.regions[2].config.fill_args['facecolor']='yellowgreen' # saddle
plotter.regions[3].config.fill_args['facecolor']='yellowgreen' # saddle

plotter.plot()
plotter.ax.legend(loc='best')
#annotate_regions(plotter.regions, 'Saddle')


# --- Make Legend ---
#(ineq0,ineq1) = plotter.regions[1].inequalities
#ineq0.boundary.config.line_args['label'] = '$p_1=p_2+2$'
#ineq1.boundary.config.line_args['label'] = '$p_1=p_2-2$'

# --- Make Axes ---
#plt.plot((-5,5),(0,0),"k",linestyle="dashed")
#plt.plot((0,0),(-5,5),"k",linestyle="dashed")

plt.ylabel('$p_1$',size="17")
plt.xlabel('$p_2$',size="17")
plt.axes().set_aspect('equal')
plt.show()
