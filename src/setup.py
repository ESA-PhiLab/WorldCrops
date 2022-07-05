from setuptools import setup, find_packages

with open('../requirements.txt') as f:
    reqs = f.read().splitlines()

setup(name='WorldCrops',
      version='0.0.1',
      description='Lib for self supervised learning',
      license='MIT',
      packages=['selfsupervised','selfsupervised.model', 'selfsupervised.processing',
      'selfsupervised.data','selfsupervised.data.croptypes','selfsupervised.data.yields','selfsupervised.data.images'],
      #packages=find_packages(,'PackageName.SubModule'),
      install_requires=reqs,
      zip_safe=False)