from pathlib import Path

from setuptools import find_packages, setup

readme_file = Path(__file__).parent / 'README.md'
with readme_file.open() as f:
    long_description = f.read()

setup(
    name='isic-challenge-scoring',
    description='Submission scoring for the ISIC Challenge',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    url='https://github.com/ImageMarkup/isic-challenge-scoring',
    project_urls={
        'Bug Reports': 'https://github.com/ImageMarkup/isic-challenge-scoring/issues',
        'Source': 'https://github.com/ImageMarkup/isic-challenge-scoring',
    },
    maintainer='ISIC Archive',
    maintainer_email='support@isic-archive.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8',
    install_requires=[
        'click',
        'click-pathlib',
        'numpy',
        'pandas>=1.1',
        'pillow>=7',
        'rdp',
        'scipy',
        'scikit-learn',
    ],
    packages=find_packages(),
    entry_points="""
        [console_scripts]
        isic-challenge-scoring=isic_challenge_scoring.__main__:cli
    """,
)
