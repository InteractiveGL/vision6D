# For development mode
import setuptools

setuptools.setup(
    name = "vision6D",
    packages=setuptools.find_packages(),
    include_package_data=True,
    extras_require={
        ':sys_platform == "win32"': [
            'opencv-python',
        ],
        ':sys_platform == "linux"': [
            'opencv-python-headless',
        ],
    },
)
