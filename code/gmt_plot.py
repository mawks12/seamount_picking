#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import pygmt


# In[2]:


def gmt_plot(data,cmap_name="rainbow",c_min=.5,c_max=4.5,anot=1,region_size=[0,360,-81,81],plot_name='slope',unit='microradian',fname=None):
    fig = pygmt.Figure()
    pygmt.makecpt(cmap=cmap_name, series=[c_min, c_max, (c_max-c_min)/40.0],background='o')
    fig.grdimage(
        grid = data,
        projection="M15c",
        region = region_size,
        cmap = True,
    )
    fig.coast(
        shorelines = True,
        land = 'gray',
        frame = ["af", 'WSne+t\"'+plot_name+'\"'],
#         frame = ["af", 'WSne+t"Slope"'],
    )
#     fig.colorbar(frame=["a"+str(anot), "x+lslope_variability", "y+lmicroradian"])
    fig.colorbar(frame=["a"+str(anot), "y+l"+unit])
    
    if fname:
        print('saving figure on %s' %fname)
        fig.savefig(fname, dpi=300)
    fig.show()
    
    return fig
    
    


# In[9]:


# %%sh

# jupyter nbconvert gmt_plot.ipynb --to python

