from glob import glob

from setuptools import find_packages, setup

setup(
    name="alohamini_nav_sim",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/alohamini_nav_sim"]),
        ("share/alohamini_nav_sim", ["package.xml"]),
        ("share/alohamini_nav_sim/launch", glob("launch/*.launch.py")),
        ("share/alohamini_nav_sim/worlds", glob("worlds/*.yaml")),
    ],
    entry_points={"console_scripts": ["simulator = alohamini_nav_sim.node:main"]},
    maintainer="AlohaMini",
    maintainer_email="anncatto@users.noreply.github.com",
    description="Planar EDU navigation simulation",
    license="Apache-2.0",
)
