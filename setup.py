from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ilearn-memory',
    version='1.0.0',
    author='broadfield-dev',
    description='A library for a self-improving AI memory system with dual-memory (Memories & Rules) and a reflective learning loop.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/broadfield-dev/ilearn-memory',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
