# For development mode
import setuptools

setuptools.setup(
    name = "vision6D",
    packages=setuptools.find_packages() + ['vision6D.data'],
    include_package_data=True,
    entry_points = {
        'console_scripts':[
            'vision6D = vision6D.entry.main:main',
        ]
    },
    url='https://github.com/ykzzyk/vision6D',
    project_urls={
        'Documentation': 'https://vision6d.readthedocs.io/en/latest/',
    },
)
