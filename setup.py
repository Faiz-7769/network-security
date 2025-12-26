'''
Docstring for setup
The setup.py file is essential part of packaging and 
distributing Python projects. It is used by setuptools
(or distutils in older python versions) to define the configurations
of your project, such as its metadata,dependencies, and more
'''

from setuptools import find_packages,setup
from typing import List

def get_requirements()-> List[str]:
    """
    Docstring for get_requirements
    this function return list of requirements
    :return: Description
    :rtype: List[str]
    """
    require_list:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            lines = file.readlines()
            # process each line
            for line in lines:
                requirement = line.strip()
                #ignore empty lines and -e .
                if requirement and requirement!='-e .':
                    require_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return require_list

print(get_requirements())

setup(
    name = "Network-Security",
    version = "0.0.1",
    author = "Faiz Ahmed",
    author_email="ahmedfaiz4699@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements()
)