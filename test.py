#!/usr/bin/env python
# coding: utf-8



#from IPython import get_ipython
#get_ipython().run_line_magic('cd', 'NAFNet')
#%cd NAFNet
import os
os.chdir("NAFNet")

import shutil

uploadd_folder = 'seck/image'
upload_folder = 'seck/input'
result_folder = 'seck/output'
result_inp = 'seck/dec_inp'
result_out = 'seck/dec_out'

if os.path.isdir(upload_folder):
    shutil.rmtree(upload_folder)
if os.path.isdir(result_folder):
    shutil.rmtree(result_folder)
if os.path.isdir(uploadd_folder):
    shutil.rmtree(uploadd_folder)
if os.path.isdir(result_inp):
    shutil.rmtree(result_inp)
if os.path.isdir(result_out):
    shutil.rmtree(result_out)
os.makedirs(upload_folder)
os.makedirs(uploadd_folder)
os.makedirs(result_folder)
os.makedirs(result_inp)
os.makedirs(result_out)