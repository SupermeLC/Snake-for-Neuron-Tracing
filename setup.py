from setuptools import setup

setup(name='neurontracing',
      version='1.0.2',
      description='neuron_tracing first edition',
      author='LiuChao',
      author_email='supermeliu@zju.edu.cn',
      url='https://github.com/SupermeLC/Snake-for-Neuron-Tracing',
      packages=['neuron_tracing','neuron_tracing.cli','neuron_tracing.lib','neuron_tracing.lib.swclib',
                'neuron_tracing.lib.klib','neuron_tracing.lib.klib.glib','neuron_tracing.model','neuron_tracing.tools'],
      py_modules=['neuron_tracing','neuron_tracing.cli','neuron_tracing.lib','neuron_tracing.lib.swclib',
                  'neuron_tracing.lib.klib','neuron_tracing.lib.klib.glib','neuron_tracing.model','neuron_tracing.tools'],
      install_requires=[
            'numpy==1.20.3',
            'anytree>=2.7.2',
            'kdtree>=0.16',
            'rtree>=0.8',
            'jsonschema>=3.2.0',
            'pandas==1.3.2',
            'matplotlib>=3.0.0',
            'scikit-image==0.18.3',
            'scipy==1.7.3',
            'Pillow==8.4.0',
            'opencv-python==4.5.5.64'
      ],
      entry_points={
          'console_scripts': [
              'neurontracing3D=neuron_tracing.cli.tracing_cli_3D:run',
              'neurontracing2D=neuron_tracing.cli.tracing_cli_2D:run'
          ]
      }
)

