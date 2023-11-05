from setuptools import find_packages, setup

with open('../requirements.txt') as f:
    reqs = f.read().splitlines()

setup(
    name='selfsupervised',
    version='0.0.2',
    description='Lib for self supervised learning',
    license='MIT',
    packages=[
        'selfsupervised', 'selfsupervised.model', 'selfsupervised.processing',
        'selfsupervised.data', 'selfsupervised.data.croptypes',
        'selfsupervised.data.yields', 'selfsupervised.data.images'
    ],
    # packages=find_packages(,'PackageName.SubModule'),
    install_requires=reqs,
    zip_safe=False)
