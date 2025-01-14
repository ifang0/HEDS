from setuptools import setup, find_packages

setup(
    name='HEDS_image_processing',
    version='0.1.0',
    author='Irene Fang',
    author_email='ifang@uiowa.edu',
    description='An image processing package for HEDS',
    packages=find_packages(),
    install_requires=[
    	'numpy', 
    	'scikit-learn', 
    	'matplotlib',
    	'pillow',
    	'scikit-image',
    	'opencv-python',
    	'SimpleITK',

    ],
)

