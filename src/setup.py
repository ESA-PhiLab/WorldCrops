from setuptools import find_packages, setup

with open('../requirements.txt') as f:
    reqs = f.read().splitlines()

setup(
    name='selfsupervised',
    version='0.0.3',
    description='Lib for self supervised learning',
    license='MIT',
    packages=find_packages(),
    install_requires=reqs,
    zip_safe=False)
