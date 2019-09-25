#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:56:35 2019

@author: ubuntu
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(name='Hello_world_app',
      ext_modules=cythonize("hello.pyx"))