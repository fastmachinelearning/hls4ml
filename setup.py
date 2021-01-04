from setuptools import setup
from setuptools import find_packages

long_description = '''
A package for machine learning inference in FPGAs. We create firmware
implementations of machine learning algorithms using high level synthesis
language (HLS). We translate traditional open-source machine learning package
models into HLS that can be configured for your use-case!

For more information visit the webpage:
https://hls-fpga-machine-learning.github.io/hls4ml/
'''

setup(name='hls4ml',
      version='0.2.0',
      description='Machine learning in FPGAs using HLS',
      long_description=long_description,
      author='HLS4ML Team',
      author_email='hls4ml.help@gmail.com',
      url='https://github.com/hls-fpga-machine-learning/hls4ml',
      license='Apache 2.0',
      install_requires=['numpy',
                        'six',
                        'pyyaml',
                        'h5py',
                        'calmjs.parse',
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
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
