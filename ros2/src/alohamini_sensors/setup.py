from glob import glob

from setuptools import find_packages, setup

setup(
    name="alohamini_sensors",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/alohamini_sensors"]),
        ("share/alohamini_sensors", ["package.xml"]),
        ("share/alohamini_sensors/launch", glob("launch/*.launch.py")),
        ("share/alohamini_sensors/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anncatto",
    maintainer_email="anncatto@users.noreply.github.com",
    description="AlohaMini sensor drivers and acquisition-time scan filtering",
    license="Apache-2.0",
    entry_points={"console_scripts": ["scan_filter = alohamini_sensors.scan_filter:main"]},
)
