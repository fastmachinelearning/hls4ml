from setuptools import setup
from setuptools import find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='hls4ml',
      version='0.4.0',
      description='Machine learning in FPGAs using HLS',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='HLS4ML Team',
      author_email='hls4ml.help@gmail.com',
      url='https://github.com/hls-fpga-machine-learning/hls4ml',
      license='Apache 2.0',
      install_requires=['numpy',
                        'six',
                        'pyyaml',
                        'h5py',
                        'onnx>=1.4.0'],
      extras_require={
        'profiling': [
            'pandas',
            'seaborn',
            'matplotlib',
            'tensorflow'
        ]
      },
      scripts=['scripts/hls4ml'],
      include_package_data=True,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: C++',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
