from setuptools import setup, find_packages

setup(
    name="gpv",
    version="0.1.1",  # Initial version
    author="Haoran Ye",
    author_email="hrye@stu.pku.edu.cn",
    description="Generative Psychometrics for Values",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Value4AI/gpv",  # GitHub or other repo URL
    packages=find_packages(),  # Automatically finds packages in your directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # Python version requirement
)
