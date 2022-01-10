from distutils.core import setup

setup(
    name='dlfs',  # How you named your package folder (MyLib)
    packages=['dlfs'],  # Chose the same as "name"
    version='0.1',  # Start with a small number and increase it with every change you make
    license='Apache-2.0',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='implement from scratch (using numpy arrays) a package based on tensorflow architecture which '
                'allows to build and train Fully Connected Networks and Convolutional Neural Networks (CNNs).',
    author='Pablo',  # Type in your name
    author_email='pablete.arino@gmail.com',  # Type in your E-Mail
    url='https://github.com/Pabloo22/Deep-Learning-from-Scratch',
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',  # I explain this later on
    keywords=['machine-learning', 'deep-learning', 'backpropagation'],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        'numpy',
        'sklearn',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache-2.0 License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which python versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
)
