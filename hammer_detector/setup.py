#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAMMER 伪造检测工具安装脚本
"""

import os
import sys
from setuptools import setup, find_packages

# 读取版本信息
with open(os.path.join("__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.strip().split("=")[1].strip(" '\"")
            break
    else:
        version = "0.1.0"

# 读取README文件
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# 读取依赖项
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="hammer_detector",
    version=version,
    author="HAMMER Team",
    author_email="hammer@example.com",
    description="HAMMER多模态伪造检测工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hammer/hammer_detector",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "hammer=run:main",
        ],
    },
) 