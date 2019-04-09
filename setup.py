import os

from setuptools import find_packages, setup


def prerelease_local_scheme(version):
    """
    Return local scheme version unless building on master in CircleCI.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on CircleCI for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if os.getenv('CIRCLE_BRANCH') == 'master':
        return ''
    else:
        return get_local_node_and_date(version)


setup(
    name='isic_challenge_scoring',
    version='0.1',
    description='ISIC: Skin Lesion Analysis Towards Melanoma Detection Scoring',
    url='https://github.com/ImageMarkup/isic-challenge-scoring',
    license='Apache 2.0',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.7.0',
    install_requires=['click', 'numpy', 'pandas', 'pillow', 'scipy', 'scikit-learn'],
    use_scm_version={'local_scheme': prerelease_local_scheme},
    setup_requires=['setuptools_scm'],
)
