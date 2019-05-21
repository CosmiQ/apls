from setuptools import setup, find_packages

readme = ''

version = '0.1.0'

# Runtime requirements.
inst_reqs = ['matplotlib>=3.0.0',
             'shapely>=1.6.4',
             'pandas>=0.23.4',
             'geopandas>=0.4.0',
             'opencv-python>=4.0',
             'numpy>=1.15.4',
             'tqdm>=4.28.1',
             'GDAL==2.4.0',
             'rtree>=0.8.3',
             'networkx>=2.2',
             'scipy>=1.2.0',
             'scikit-image>=0.14.0',
             'affine>=2.2.1',
             'utm>=0.4.0',
             ]

extra_reqs = {
    'test': ['mock', 'pytest', 'pytest-cov', 'codecov']}

setup(name='apls',
      version=version,
      description=u"""CosmiQ Works APLS Metric Implementation""",
      long_description=readme,
      classifiers=[
                   'Intended Audience :: Information Technology',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: GIS'],
      keywords='spacenet machinelearning iou aws',
      author=u"Adam Van Etten",
      author_email='avanetten@iqt.org',
      url='https://github.com/CosmiQ/apls',
      license='Apache-2.0',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      zip_safe=False,
      include_package_data=True,
      install_requires=inst_reqs,
      extras_require=extra_reqs,
      )
