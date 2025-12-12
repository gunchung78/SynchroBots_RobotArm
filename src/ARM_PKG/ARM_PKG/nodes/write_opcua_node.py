#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import threading

from asyncua import Client
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ARM_PKG.config.opcua_config import OPCUA_SERVER_URL
# TODO: 나중에 opcua_config.py 안에 아래 상수들도 옮기면 좋다.
WRITE_OBJECT_NODE_ID = "ns=2;i=3"          # 예시: ARM 객체 노드
WRITE_METHOD_NODE_ID = "ns=2;s=write_send_arm_json"  # 예시: JSON 보내는 메서드


class WriteOpcuaNode(Node):
    """
    go_move_node / camera_node 에서 넘어온 결과를 받아서
    OPC UA 서버로 write하는 전용 노드.

    - /opcua_write_mission_state  : Place 완료, 전체 완료 등 상태 전송
    - /opcua_write_vision_result  : AI 비전 체크 결과 전송

    실제 OPCUA write는 내부 asyncio 루프에서 처리한다.
    """

    def __init__(self):
        super().__init__("write_opcua_node")

        # 1) ROS 구독 설정
        self.mission_sub = self.create_subscription(
            String,
            "/opcua_write_mission_state",     # go_move_node 에서 publish 예정
            self._on_mission_state,
            10
        )

        self.vision_sub = self.create_subscription(
            String,
            "/opcua_write_vision_result",     # camera_node 에서 publish 예정
            self._on_vision_result,
            10
        )

        # 2) OPCUA write 요청을 처리할 큐 + 별도 asyncio 루프
        self.loop = asyncio.new_event_loop()
        self.queue: asyncio.Queue = asyncio.Queue()

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        self.get_logger().info("write_opcua_node 초기화 완료")

    # ----------------------------- ROS 콜백 -----------------------------

    def _on_mission_state(self, msg: String):
        """
        go_move_node 에서 올라온 미션 상태(JSON 문자열)를 OPCUA write 큐에 적재.
        """
        self.get_logger().info(f"[WRITE] mission_state 수신: {msg.data}")
        payload = {
            "kind": "mission_state",
            "json": msg.data,
        }
        # asyncio 루프에서 처리하도록 큐에 넣기
        self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)

    def _on_vision_result(self, msg: String):
        """
        camera_node 에서 올라온 비전 결과(JSON 문자열)를 OPCUA write 큐에 적재.
        """
        self.get_logger().info(f"[WRITE] vision_result 수신: {msg.data}")
        payload = {
            "kind": "vision_result",
            "json": msg.data,
        }
        self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)

    # --------------------------- OPCUA 비동기 루프 ---------------------------

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._writer_loop())

    async def _writer_loop(self):
        """
        큐에서 write 요청을 하나씩 꺼내 OPC UA 서버에 전송하는 메인 루프.
        """
        while rclpy.ok():
            payload = await self.queue.get()
            try:
                await self._send_to_opcua(payload)
            except Exception as e:
                self.get_logger().error(f"OPCUA write 중 오류: {e}")

    async def _send_to_opcua(self, payload: dict):
        """
        실제 OPCUA write를 수행하는 부분.
        지금은 예시로 METHOD 한 개만 호출하고,
        나중에 kind 에 따라 다른 메서드를 쓰도록 확장할 수 있다.
        """
        kind = payload.get("kind")
        json_str = payload.get("json")

        # JSON 유효성 간단 체크
        try:
            _ = json.loads(json_str)
        except json.JSONDecodeError:
            self.get_logger().error(f"잘못된 JSON, 전송 스킵: {json_str}")
            return

        self.get_logger().info(
            f"[OPCUA] send kind={kind} → URL={OPCUA_SERVER_URL}, "
            f"Object={WRITE_OBJECT_NODE_ID}, Method={WRITE_METHOD_NODE_ID}"
        )

        # OPCUA 서버에 접속해서 Method 호출
        async with Client(OPCUA_SERVER_URL) as client:
            obj_node = client.get_node(WRITE_OBJECT_NODE_ID)
            method_node = client.get_node(WRITE_METHOD_NODE_ID)

            # 서버 측 메서드 시그니처: (json_string) 하나 받는 것으로 가정
            result = await obj_node.call_method(method_node.nodeid, json_str)

            self.get_logger().info(f"[OPCUA] write 완료, result={result}")

    # ------------------------------ 종료 처리 ------------------------------

    def destroy_node(self):
        # asyncio 루프 정리
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WriteOpcuaNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("write_opcua_node 종료(Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
