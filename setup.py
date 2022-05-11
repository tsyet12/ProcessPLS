from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

import os
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] 
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()


setup(name='ProcessPLS', 
version='1.01', 
license='BSD 2-Clause',
description="Implementation of ProcessPLS in Python",
author='Sin Yong Teng',
long_description=long_description,
long_description_content_type="text/markdown",
author_email='tsyet12@gmail.com',
keywords = ['Path Modelling', 'Chemometrics', 'Process Analytical Technology', 'Machine Learning'],
packages=find_packages(),
setup_requires=install_requires,
install_requires=install_requires,
classifiers=[
    'Development Status :: 4 - Beta',      
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD License', 
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  )