from setuptools import setup, find_packages

package_name = 'nrp'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    install_requires=['setuptools', "numpy", "matplotlib", "scikit-learn", "typing_extensions", "networkx"],
    maintainer='Yunfan Lu',
    maintainer_email='luyunfan44@gmail.com',
    description= 'nrp',
    tests_require=['pytest'],
)
