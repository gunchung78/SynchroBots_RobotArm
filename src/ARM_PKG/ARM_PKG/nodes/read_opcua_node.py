#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import threading

from asyncua import ua
from asyncua import Client
from asyncua.common.subscription import DataChangeNotificationHandler
from asyncua.common.subscription import EventNotificationHandler
from asyncua.common.subscription import StatusChangeNotificationHandler

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ARM_PKG.config.opcua_config import OPCUA_SERVER_URL, SUBSCRIBE_NODES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("read_opcua_node")


class OpcuaSubHandler(
    DataChangeNotificationHandler,
    EventNotificationHandler,
    StatusChangeNotificationHandler
):
    """
    OPCUA datachange 이벤트를 받아서
    각각의 이벤트를 공통 토픽(/arm/opcua_event)으로 전달한다.
    """

    def __init__(self, ros_node: Node, node_info_map):
        super().__init__()
        self.node = ros_node
        self.node_info_map = node_info_map  # nodeid → config

    def datachange_notification(self, node, val, data):
        info = self.node_info_map.get(node.nodeid)
        if not info:
            self.node.get_logger().warn(f"[READ] unknown node: {node}")
            return

        name = info["name"]
        self.node.get_logger().info(f"[READ] {name} changed → {val}")

        # 모든 결과를 공통 토픽으로 publish
        msg = {
            "name": name,
            "value": val
        }

        ros_msg = String()
        ros_msg.data = json.dumps(msg)
        self.node.event_pub.publish(ros_msg)


class ReadOpcuaNode(Node):
    """
    config.py에 정의된 모든 OPCUA read 노드를 구독하고
    데이터를 단일 topic(/arm/opcua_event)에 publish하는 노드
    """

    def __init__(self):
        super().__init__("read_opcua_node")

        # 하나의 공통 topic
        self.event_pub = self.create_publisher(String, "/opcua_read_event", 10)

        self.get_logger().info("read_opcua_node 초기화 완료")

        # OPCUA loop 분리
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._opcua_main())

    async def _opcua_main(self):
        while rclpy.ok():
            client = Client(OPCUA_SERVER_URL)

            try:
                await client.connect()
                self.get_logger().info("OPCUA 서버 연결 성공")

                node_info_map = {}
                nodes = []

                # config 기반 node 생성
                for conf in SUBSCRIBE_NODES:
                    node = client.get_node(conf["node_id"])
                    nodes.append(node)
                    node_info_map[node.nodeid] = conf

                    self.get_logger().info(
                        f"[OPCUA] subscribe: {conf['name']} -> {conf['node_id']}"
                    )

                handler = OpcuaSubHandler(self, node_info_map)
                sub = await client.create_subscription(200, handler)
                await sub.subscribe_data_change(nodes)

                # 연결 유지
                while rclpy.ok():
                    await asyncio.sleep(1)

            except Exception as e:
                self.get_logger().error(f"OPCUA 오류: {e}")

            finally:
                try:
                    await client.disconnect()
                except:
                    pass
                self.get_logger().info("OPCUA 연결 종료 → 3초 후 재접속")
                await asyncio.sleep(5)


def main(args=None):
    rclpy.init(args=args)
    node = ReadOpcuaNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
