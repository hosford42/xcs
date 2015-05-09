#!/usr/bin/env python

"""Setup script for xcs."""

# TODO: This is basically just copied from https://github.com/pypa/sampleproject
#       and minimally modified. Go back through and clean it up.

__author__ = 'Aaron Hosford'

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='xcs',
    version='1.0.0a',
    description='XCS (Accuracy-based Classifier System)',
    long_description=long_description,
    url='https://github.com/hosford42/xcs',
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
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='xcs accuracy classifier lcs machine learning',
    modules=['xcs'],
    install_requires=['numpy'],
)
