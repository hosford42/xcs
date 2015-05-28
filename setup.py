#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup script for xcs."""

__author__ = 'Aaron Hosford'

import traceback
from setuptools import setup
from codecs import open
from os import path

from xcs import __version__

here = path.abspath(path.dirname(__file__))


# Default long description
long_description = """
xcs
===

* Project Home: https://github.com/hosford42/xcs
* Download: https://pypi.python.org/pypi/xcs
* Tutorial: https://github.com/hosford42/xcs/blob/master/doc/Tutorial.ipynb
* Wiki: https://github.com/hosford42/xcs/wiki
* FAQ: https://github.com/hosford42/xcs/wiki/FAQ

A Python implementation of the Accuracy-based Learning Classifier Systems (XCS),
roughly as described in the 2001 paper "An algorithmic description of XCS" by
Martin Butz and Stewart Wilson.

Butz, M. and Wilson, S. (2001). An algorithmic description of XCS.
    In Lanzi, P., Stolzmann, W., and Wilson, S., editors, Advances in Learning
    Classifier Systems: Proceedings of the Third International Workshop, volume
    1996 of Lecture Notes in Artificial Intelligence, pages 253–272. Springer-Verlag
    Berlin Heidelberg.


Related projects:
    * Pier Luca Lanzi's xcslib (C++): http://xcslib.sourceforge.net/
    * Ryan J. Urbanowicz's implementations (Python): http://gbml.org/2010/03/24/python-lcs-implementations-xcs-ucs-mcs-for-snp-environment/
""".strip()


# Try to build the readme file, but don't fail out if it doesn't work.
try:
    from build_readme import build_readme
except ImportError:
    traceback.format_exc()
else:
    try:
        build_readme(here)
    except Exception:
        traceback.print_exc()


# Get the long description from the relevant file. First try README.rst, then fall back on
# the default string defined here in this file.
if path.isfile(path.join(here, 'README.rst')):
    with open(path.join(here, 'README.rst'), encoding='utf-8', mode='rU') as description_file:
        long_description = description_file.read()

setup(
    name='xcs',
    version=__version__,
    description='XCS (Accuracy-based Classifier System)',
    long_description=long_description,
    url='http://hosford42.github.io/xcs',
    author=__author__,
    author_email='hosford42@gmail.com',
    license='Revised BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='xcs accuracy classifier lcs machine learning',
    packages=['xcs'],
    # install_requires=['numpy'],  # No longer required
)
