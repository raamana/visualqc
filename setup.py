#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'scipy',
    'numpy',
    'matplotlib',
    'mrivis',
    'nibabel',
    'scikit-learn',
    'pybids',
    'nilearn'
    ]

setup_requirements = requirements

test_requirements = requirements

import versioneer

setup(
    name='visualqc',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Assistive tool for the quality control of neuroimaging data",
    long_description=readme + '\n\n' + history,
    author="Pradeep Reddy Raamana",
    author_email='raamana@gmail.com',
    url='https://github.com/raamana/visualqc',
    packages=find_packages(include=['visualqc']),
    entry_points={
        'console_scripts': [
            'visualqc_t1_mri=visualqc.__t1_mri__:main',
            'visualqc_anatomical=visualqc.__t1_mri__:main',
            'visualqc_defacing=visualqc.__defacing__:main',
            'visualqc_func_mri=visualqc.__func_mri__:main',
            'visualqc_diffusion=visualqc.__diffusion__:main',
            'visualqc_freesurfer=visualqc.__freesurfer__:main',
            'visualqc_alignment=visualqc.__alignment__:main',
            # shortcuts
            'vqcdeface=visualqc.__defacing__:main',
            'vqct1=visualqc.__t1_mri__:main',
            'vqcanat=visualqc.__t1_mri__:main',
            'vqcfunc=visualqc.__func_mri__:main',
            'vqcdwi=visualqc.__diffusion__:main',
            'vqcfs=visualqc.__freesurfer__:main',
            'vqcalign=visualqc.__alignment__:main'
            ],
        },
    include_package_data=True,
    install_requires=requirements,
    license="Apache license",
    zip_safe=False,
    keywords='visualqc',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    )
