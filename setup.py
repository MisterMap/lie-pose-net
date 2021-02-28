#!/usr/bin/env python

import os
from distutils.core import setup

folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

filtred_install_requires = []
for require in install_requires:
    if require[:2] == "-e":
        print("Warning: you need install {} separatly".format(require))
    else:
        filtred_install_requires.append(require)
install_requires = filtred_install_requires

setup(name='lieposenet',
      version='0.1',
      description='LiePoseNet',
      author='Mikhail Kurenkov',
      author_email='Mikhail.Kurenkov@skoltech.ru',
      package_dir={},
      packages=["lieposenet", "lieposenet.utils", "lieposenet.models",
                "lieposenet.data", "lieposenet.criterions", "lieposenet.factories"],
      install_requires=install_requires
      )