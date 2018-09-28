from setuptools import setup

setup(

    name='spatial-evo', 

    version='0.1.0',  

    description='2D Spatial evolution', 

    long_description_content_type='text/markdown',  
    
    url='https://github.com/atarashansky/spatial_evolution',  

    author='Alexander J. Tarashansky',  

    author_email='tarashan@stanford.edu',  

    py_modules=["evolution"],
    
    install_requires=['numpy','scipy','matplotlib','Pillow']
)
