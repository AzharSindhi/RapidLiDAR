from setuptools import setup, find_packages

setup(
    name='rapidlidar',
    version='0.1.0',
    description='RapidLiDAR: single-pass LiDAR scene completion via adaptive initialization and multi-scale deformable reconstruction.',
    packages=find_packages(include=['rapidlidar', 'rapidlidar.*']),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)