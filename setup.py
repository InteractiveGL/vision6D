# For development mode
import setuptools

setuptools.setup(
    name = "vision6D",
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points = {
        'console_scripts':[
            'vision6D = vision6D.run_gui:exe',
        ]
    }
)
