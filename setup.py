"""Setup script for sleep stage classification package."""

from setuptools import find_packages, setup

setup(
    name='eeg_random_forest',
    packages=find_packages(),
    version='0.1.0',
    description='A machine learning model using Random Forests to classify sleep stages from EEG recordings.',
    author='David Rivas',
    license='MIT',
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'mne>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'click>=8.0.0',
        'joblib>=1.0.0',
        'python-dotenv>=0.19.0'
    ],
    extras_require={
        'cloud': ['google-cloud-storage>=2.0.0', 'google-cloud-aiplatform>=1.0.0'],
        'dev': ['pytest>=6.0.0', 'black>=21.0.0', 'flake8>=4.0.0']
    }
)