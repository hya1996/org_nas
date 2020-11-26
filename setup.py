#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

REQUIRED = [
	"Pillow",
	"numpy",
	"toolz",
	"pybind11",
]

setup(
	name='org_nas',
	version='1.0.0',
	description='nas of own orange',
	long_description_content_type='text/markdown',
	author='hya1996',
	author_email='hya1996@126.com',
	# python_requires=REQUIRES_PYTHON,
	# url=URL,
	# packages=find_packages(exclude=('tests',)),
	install_requires=REQUIRED,
	# dependency_links=DEPENDENCY_LINKS,
	# include_package_data=True,
	license='MIT',
	# ext_modules=get_numpy_extensions(),
)
