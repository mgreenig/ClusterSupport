import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as req_file:
    reqs = req_file.read().splitlines()

setuptools.setup(
    name='clustersupport', # Replace with your own username
    version='0.0.6',
    author='Matthew Greenig',
    author_email='matt.greenig@gmail.com',
    description='A small package for enhancing scikit-learn\'s clustering capabilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mgreenig/ClusterSupport',
    packages=setuptools.find_packages(exclude = ['*.tests', '*.tests.*', 'tests.*', 'tests']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires = reqs
)