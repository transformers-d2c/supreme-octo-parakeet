#!/bin/bash
. /ros2_foxy/ros2-linux/setup.bash
mv src/robot_software_transformers /newtemp
cd src
ros2 pkg create --build-type ament_python robot_software_transformers
cd ..
colcon build
. install/local_setup.bash
cd src
rm -rf robot_software_transformers
mv /newtemp robot_software_transformers
cd ..
colcon build
. install/local_setup.bash
