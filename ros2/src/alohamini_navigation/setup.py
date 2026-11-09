from glob import glob

from setuptools import find_packages, setup

setup(
    name="alohamini_navigation",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/alohamini_navigation"]),
        ("share/alohamini_navigation", ["package.xml"]),
        ("share/alohamini_navigation/launch", glob("launch/*.launch.py")),
        (
            "share/alohamini_navigation/config",
            glob("config/*.yaml")
            + glob("config/*.xml")
            + glob("config/*.rviz")
            + glob("config/*.json"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anncatto",
    maintainer_email="anncatto@users.noreply.github.com",
    description="AlohaMini wheel/IMU localization, SLAM, maps and Nav2",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "map_session = alohamini_navigation.map_session:main",
            "base_teleop = alohamini_navigation.base_teleop:main",
            "edu_session = alohamini_navigation.session:main",
            "edu_check = alohamini_navigation.check:main",
            "scan_gate = alohamini_navigation.scan_gate:main",
        ]
    },
)
