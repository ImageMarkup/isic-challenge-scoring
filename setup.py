import os
import pathlib

from setuptools import find_packages, setup


def prerelease_local_scheme(version) -> str:
    """
    Return local scheme version unless building a tag or master in CircleCI.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on CircleCI for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    circleci_tag = os.getenv('CIRCLE_TAG')
    circleci_branch = os.getenv('CIRCLE_BRANCH')
    if circleci_tag or (circleci_branch == 'master'):
        return ''
    else:
        return get_local_node_and_date(version)


with (pathlib.Path(__file__).parent / 'README.md').open() as description_stream:
    long_description = description_stream.read()


setup(
    name='isic-challenge-scoring',
    description='Submission scoring for the ISIC Challenge',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ImageMarkup/isic-challenge-scoring',
    license='Apache 2.0',
    maintainer='ISIC Archive',
    maintainer_email='admin@isic-archive.com',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.7.0',
    install_requires=[
        'click',
        'click-pathlib',
        'numpy',
        'pandas',
        'pillow>=7',
        'rdp',
        'scipy',
        'scikit-learn',
    ],
    use_scm_version={'local_scheme': prerelease_local_scheme},
    entry_points="""
        [console_scripts]
        isic-challenge-scoring=isic_challenge_scoring.__main__:cli
    """,
)
