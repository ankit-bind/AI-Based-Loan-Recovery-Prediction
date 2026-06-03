from setuptools import setup, find_packages 
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:

    '''This function will return the list of requirements'''
    requirements = []
    with open(file_path) as file_obj:
         requirements = file_obj.readlines()
         requirements = [req.replace('\n', '').strip() for req in requirements]

         if HYPEN_E_DOT in requirements:
             requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='AI-Based Loan Recovery Probability Prediction System',
    version='0.0.1',
    author='Ankit',
    author_email='itz.ankitbind01@gmail.com',     
    description='A machine learning System to predict the probability of loan recovery.',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)
