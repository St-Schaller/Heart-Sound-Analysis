

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='heartsound',
    version='0.1.0',
    description='Framework for Heart Sound Analysis',
    long_description=readme,
    author='Stefanie Schaller',
    author_email='stefanie.schaller@informatik.uni-augsburg.de',
    url='https://github.com/St-Schaller/Heart-Sound-Analysis',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
   python_requires='>=3.7',
   install_requires=[],
)



