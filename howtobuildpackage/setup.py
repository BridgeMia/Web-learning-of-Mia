from setuptools import setup

setup(name='sample_package',
      packages=['sample_package', 'sample_package.utils'],
      package_data = {'sample_package': ['package_data/*.txt']})

