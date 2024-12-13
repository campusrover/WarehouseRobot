
# Precision Agriculture Robot

This research project focuses on the development of an autonomous WarehouseRobot platform equipped with a precision object-detection and item-handling system. It integrates onboard and remote systems to enable efficient, automated item retrieval and delivery for warehouse operations.

## Features  
- **Autonomous Navigation**: Uses fiducial markers for warehouse mapping.  
- **Precision Item Handling**: Features a precision object-detection and item-handling system controlled via a ROS publisher.  
- **Real-Time Control**: Integrated terminal interface for real-time control and monitoring of warehouse operations.  

## Prerequisites
- Turtlebot platform with onboard Raspberry Pi.
- ROS (Robot Operating System) installed on both the onboard Raspberry Pi and the local machine.
- Python 3.x installed with required dependencies.

## Installation

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/your-repo/precision-agriculture-robot.git
    ```

2. Bring up the Turtlebot:
    ```bash
    roslaunch turtlebot3_bringup right2.launch
    roslaunch WarehouseRobot robot_bringup.launch
    ```
    
## LINKS

FAQ:
- [Partial Line Follow](https://github.com/campusrover/labnotebook2/blob/main/docs/faq/camera/follow_partial_line.md)

[Video](https://drive.google.com/drive/folders/14H7OUzs3a8utZB8V8RG8v7ATSxsY7lle?usp=sharing)

[Full Report](https://github.com/campusrover/labnotebook2/blob/main/docs/reports/2024/WarehouseRobot.md)
