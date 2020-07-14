from setuptools import find_packages, setup

setup(
    name='noddlrm',
    version='0.0.1',
    packages=find_packages(exclude=("tutorials",)),
    description="NOD DLRM - Adapted from OpenRec(https://openrec.ai/)",
    url="https://nod.ai/",
    license='Apache 2.0',
    author='Chi Liu',
    author_email='chi@nod-labs.com',
    install_requires=[
        'tqdm>=4.15.0',
        'numpy>=1.13.0',
        'termcolor>=1.1.0'
          ],
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: Apache Software License',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
