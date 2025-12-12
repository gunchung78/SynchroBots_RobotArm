from setuptools import find_packages, setup

package_name = 'ARM_PKG'

setup(
    name=package_name,
    version='0.0.0',
     packages=[package_name, package_name + '.nodes', package_name + '.config'],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/arm_system.launch.py'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kgkim',
    maintainer_email='dbstjq34a2@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 형식: 실행이름 = 패키지.파일명:main함수
            'read_opcua_node = ARM_PKG.nodes.read_opcua_node:main',
            'arm_main_node   = ARM_PKG.nodes.main_node:main',
            'go_move_node = ARM_PKG.nodes.go_move_node:main',
            'camera_vision_node = ARM_PKG.nodes.camera_vision_node:main',
            'arm_driver_node = ARM_PKG.nodes.arm_driver_node:main',
            'mycobot_joint_state_publisher = ARM_PKG.nodes.joint_state_publisher_node:main',
        ],
    },
)
