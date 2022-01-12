from distutils.core import setup

setup(
    name='dlfs',
    packages=['dlfs',
              'dlfs.activation_functions',
              'dlfs.convolutions',
              'dlfs.layers',
              'dlfs.losses',
              'dlfs.metrics',
              'dlfs.models',
              'dlfs.optimizers',
              'dlfs.preprocessing'],

    version='0.1.5',
    license='apache-2.0',
    description='implement from scratch (using numpy arrays) a package based on tensorflow architecture which '
                'allows to build and train Fully Connected Networks and Convolutional Neural Networks (CNNs).',
    author='Pablo',
    author_email='pablete.arino@gmail.com',
    url='https://github.com/Pabloo22/Deep-Learning-from-Scratch',
    download_url='https://github.com/Pabloo22/dlfs/archive/refs/tags/v0.1.5.tar.gz',
    keywords=['machine-learning', 'deep-learning', 'backpropagation'],  # Keywords that define your package best
    install_requires=[
        'numpy',
        'tqdm',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
