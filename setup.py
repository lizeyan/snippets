#!/usr/bin/env python

try:  # for pip >= 10
    # noinspection PyProtectedMember
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
from setuptools import setup
import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')
install_requirements = parse_requirements('requirements.txt', session='hack')

setup(name='Snippets',
      version='0.1',
      python_requires='>3.7',
      description='Pytorch ml snippets',
      author='Zeyan LI',
      author_email='lizytalk@outlook.com',
      install_requirements=install_requirements
      )
