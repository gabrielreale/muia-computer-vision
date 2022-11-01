# Copyright (C) GMV - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# The receiver of this file is allowed to use it exclusively for the purposes
# explicitly defined or contractually agreed between the company and the receiver,
# observing in addition legal regulations in intellectual property and other legal
# requirements where applicable.
# Proprietary and confidential
#
# $Rev::                      $:  Revision of last commit.
# $Author::                   $:  Author of last commit.
# $Date::                     $:  Date of last commit.

from setuptools import setup

install_requires = []

with open("README.md", "r") as longdesc:
    long_description = longdesc.read()
    setup(
        name = "computervision",
        description = "",
        long_description = long_description,
        author = "Gabriel Reale",
        version = "0.1.0",
        packages=['computervision',
                    'computervision/base', 'computervision/data', 'computervision/ffnn'],
        install_requires=install_requires,
        entry_points = {}
    )
