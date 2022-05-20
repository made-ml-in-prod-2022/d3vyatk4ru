""" Installer necessary lib """
from setuptools import find_packages, setup

REQUIREMENTS_TXT = 'requirements.txt'

with open(REQUIREMENTS_TXT, 'r', encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name='Production ready project hw-1',
    packages=find_packages(),
    version="0.1.0",
    description='''
            'Production ready' project 'for ML in prodation' course
            in VK education by Devyatkin Daniil, ML-21, 2022 y.
    ''',
    author='Devyatkin Daniil',
    license='MIT',
    install_requires=required,
)
