from glob import glob

from setuptools import find_packages, setup

package_name = "alohamini_imu"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md", "SOURCE.json"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="AlohaMini",
    maintainer_email="anncatto@users.noreply.github.com",
    description="Hiwonder ten-axis USB IMU reader and data verification tools.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "imu_node = alohamini_imu.imu_node:main",
            "imu_data_test = alohamini_imu.imu_data_test:main",
        ]
    },
)
