#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'matplotlib',
    'mrivis',
    'nibabel'
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

import versioneer

setup(
    name='visualqc',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Tool to automate the quality control of pial and white matter surfaces from Freesurfer Parcellation",
    long_description=readme + '\n\n' + history,
    author="Pradeep Reddy Raamana",
    author_email='raamana@gmail.com',
    url='https://github.com/raamana/visualqc',
    packages=find_packages(include=['visualqc']),
    entry_points={
        'console_scripts': [
            'visualqc=visualqc.__main__:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='visualqc',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
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
