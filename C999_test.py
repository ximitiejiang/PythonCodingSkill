#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:32:50 2019

@author: suliang
"""

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

bbox = np.array([217.62, 240.54, 255.61, 297.29], dtype=np.float32)
xy = [bbox[0], bbox[1]]
w = bbox[2] - bbox[0]
h = bbox[3] - bbox[1]

ax.add_patch(plt.Rectangle(xy, w, h, fill=False,
                           edgecolor=(1,1,0), linewidth=1.5, alpha=1))