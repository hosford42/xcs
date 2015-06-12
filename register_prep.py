#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Prepares for registration.

# TODO: Clean this hacked together script up!

__author__ = 'Aaron Hosford'

import glob
import os
import zipfile

from xcs import __version__
import build_readme

os.chdir('.\\doc')
try:
    os.system('ipython nbconvert XCSTutorial.ipynb')
finally:
    os.chdir('..')

# This is already done in setup.py. No need to do it here.
# build_readme.build_readme()

os.system('python setup.py sdist bdist_wheel')

with open('xcs.egg-info/PKG-INFO', encoding='utf-8', mode='rU') as infile:
    with open('xcs.egg-info/PKG-INFO-FIXED', encoding='utf-8', mode='w') as outfile:
        prev_skipped = False
        for line in infile:
            if line.strip() or prev_skipped:
                outfile.write(line)
                prev_skipped = False
            else:
                prev_skipped = True
os.remove('xcs.egg-info/PKG-INFO')
os.rename('xcs.egg-info/PKG-INFO-FIXED', 'xcs.egg-info/PKG-INFO')

dist = glob.glob('dist/*' + __version__ + '*.whl')[-1]
print(dist)
os.system('pip install ' + os.path.join('dist', os.path.basename(dist)) + ' --upgrade')


zip_path = os.path.join('dist/pythonhosted.zip')
tutorial_path = 'doc/XCSTutorial.html'

if os.path.isfile(zip_path):
    os.remove(zip_path)
with zipfile.ZipFile(zip_path, mode="w") as zf:
    zf.write(tutorial_path, 'index.html')

# TODO: Run all tests.
