#!/usr/bin/env python

from setuptools import setup, find_packages

from miles import version


classifiers = '''\
Development Status :: 3 - Alpha
Environment :: Console
Environment :: X11 Applications
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 3.5
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics
Topic :: Scientific/Engineering :: Chemistry
'''

setup(name='miles',
      version=version.v_short,
      description='Automated tool for milestoning simulations',
      long_description=('Automated tool for running Milestoning simulations '
                        'using third-party Molecular Dynamics packages.'),
      license='MIT License',
      classifiers=classifiers.splitlines(),
      url='http://github.com/clsb/miles',
      author='Juan M. Bello-Rivas',
      author_email='jmbr@superadditive.com',
      packages=find_packages(),
      package_dir={'miles': 'miles'},
      package_data={'': ['LICENSE']},
      install_requires=['scipy', 'numpy', 'mpi4py', 'networkx'],
      extras_require={
          'plotting': ['matplotlib', 'seaborn'],
          'bells-and-whistles': ['argcomplete', 'colorama',
                                 'termcolor', 'tqdm']
      },
      entry_points={
          'console_scripts': 'miles = miles.cli:main'
      })
