#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
See https://coderwall.com/p/qawuyq/use-markdown-readme-s-in-python-modules (Thanks for the idea, Will McKenzie!)
This module attempts to build a README.rst from a README.md.
It requires pandoc to be installed (http://pandoc.org/) as well as pypandoc, the python bindings for pandoc.
"""

__author__ = 'Aaron Hosford'

import os


def convert_md_to_rst(source, destination=None):
    """Try to convert the source, an .md (markdown) file, to an .rst (reStructuredText) file at the destination. If
    the destination isn't provided, it defaults to be the same as the source path except for the filename extension.
    If the destination file already exists, it will be overwritten. In the event of an error, the destination file will
    be left untouched."""

    # Doing this in the function instead of the module level ensures the error occurs when the function is called,
    # rather than when the module is evaluated.
    try:
        import pypandoc
    except ImportError:
        # Don't give up right away; first try to install the python module.
        os.system("pip install pypandoc")
        import pypandoc

    # Set our destination path to a default, if necessary
    destination = destination or (os.path.splitext(source)[0] + '.rst')

    # If there's already a file at the destination path, move it out of the way, but don't delete it.
    if os.path.isfile(destination):
        if os.path.isfile(destination + '.bak'):
            os.remove(destination + '.bak')
        os.rename(destination, destination + '.bak')

    try:
        # Try to convert the file.
        pypandoc.convert(source, 'rst', format='md', outputfile=destination)
    except:
        # If for any reason the conversion fails, try to put things back like we found them.
        if os.path.isfile(destination):
            os.remove(destination)
        if os.path.isfile(destination + '.bak'):
            os.rename(destination + '.bak', destination)
        raise

    # The .bak is intentionally left in place; it's easy to add a .gitignore line for it, and it's important to keep
    # the previous version handy just in case the conversion doesn't go as planned.


def build_readme(base_path=None):
    """Call the conversion routine on README.md to generate README.rst. Why do all this? Because pypi requires
    reStructuredText, but markdown is friendlier to work with and is nicer for GitHub."""
    if base_path:
        path = os.path.join(base_path, 'README.md')
    else:
        path = 'README.md'
    convert_md_to_rst(path)
    print("Successfully converted README.md to README.rst")

