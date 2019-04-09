from setuptools import find_packages, setup

setup(
    name='isic_challenge_scoring',
    version='0.1',
    description='ISIC: Skin Lesion Analysis Towards Melanoma Detection Scoring',
    url='https://github.com/ImageMarkup/isic-challenge-scoring',
    license='Apache 2.0',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.7.0',
    install_requires=['click', 'numpy', 'pandas', 'pillow', 'scipy', 'scikit-learn'],
)
