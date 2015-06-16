#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup script for xcs."""

__author__ = 'Aaron Hosford'

from setuptools import setup
from codecs import open
from os import path

from xcs import __version__

here = path.abspath(path.dirname(__file__))


# Default long description
long_description = """

XCS
===

*Accuracy-based Learning Classifier Systems for Python 3*

Links
-----

-  `Project Home <http://hosford42.github.io/xcs/>`__
-  `Tutorial <https://pythonhosted.org/xcs/>`__
-  `Source <https://github.com/hosford42/xcs>`__
-  `Distribution <https://pypi.python.org/pypi/xcs>`__

The package is available for download under the permissive `Revised BSD
License <https://github.com/hosford42/xcs/blob/master/LICENSE>`__.

""".strip()


# Get the long description from the relevant file. First try README.rst,
# then fall back on the default string defined here in this file.
if path.isfile(path.join(here, 'README.rst')):
    with open(path.join(here, 'README.rst'),
              encoding='utf-8',
              mode='rU') as description_file:
        long_description = description_file.read()

# See https://pythonhosted.org/setuptools/setuptools.html for a full list
# of parameters and their meanings.
setup(
    name='xcs',
    version=__version__,
    author=__author__,
    author_email='hosford42@gmail.com',
    url='http://hosford42.github.io/xcs',
    license='Revised BSD',
    platforms=['any'],
    description='XCS (Accuracy-based Classifier System)',
    long_description=long_description,

    # See https://pypi.python.org/pypi?:action=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular,
        # ensure that you indicate whether you support Python 2, Python 3
        # or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='xcs accuracy classifier lcs reinforcement machine learning',
    packages=['xcs'],
    # install_requires=['numpy'],  # No longer required

    test_suite="tests",
    tests_require=["numpy"],
)
