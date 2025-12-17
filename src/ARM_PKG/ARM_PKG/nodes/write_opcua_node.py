#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import threading

from asyncua import Client, ua  # ✅ ua 추가
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ARM_PKG.config.opcua_config import OPCUA_SERVER_URL

WRITE_OBJECT_NODE_ID = "ns=2;i=3"
WRITE_METHOD_NODE_ID = "ns=2;s=write_send_arm_json"


class WriteOpcuaNode(Node):
    """
    camera_vision_node 등이 publish한 결과를 받아
    OPC UA 서버의 write_send_arm_json 메서드를 호출하는 노드.

    - 구독 토픽:
      /arm/ai_result (String, JSON 문자열)

    - enable_opcua=True 일 때만 실제 통신 수행
    """

    def __init__(self):
        super().__init__("write_opcua_node")

        # ✅ 파라미터
        self.declare_parameter("enable_opcua", False)
        self.declare_parameter("opcua_url", OPCUA_SERVER_URL)
        self.declare_parameter("write_object_node_id", WRITE_OBJECT_NODE_ID)
        self.declare_parameter("write_method_node_id", WRITE_METHOD_NODE_ID)
        self.declare_parameter("opcua_call_timeout_sec", 5.0)
        self.declare_parameter("opcua_retry", 2)

        # ✅ AI 결과 구독 (지금 네가 이미 쓰는 흐름)
        self.ai_sub = self.create_subscription(
            String,
            "/arm/ai_result",
            self._on_ai_result,
            10
        )

        # ✅ 내부 asyncio 루프 + 큐
        self.loop = asyncio.new_event_loop()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        self.get_logger().info("write_opcua_node 초기화 완료")

    # ----------------------------- ROS 콜백 -----------------------------

    def _on_ai_result(self, msg: String):
        json_str = msg.data
        enable_opcua = bool(self.get_parameter("enable_opcua").value)

        # JSON 유효성 체크(여기서 한번)
        try:
            data = json.loads(json_str)
        except Exception as e:
            self.get_logger().error(f"[WRITE] invalid json → drop: {repr(e)}")
            return

        # 로그는 너무 길어지지 않게 요약
        status = str(data.get("status", "-"))
        module_type = str(data.get("module_type", "-"))
        conf = float(data.get("classification_confidence") or 0.0)
        pick_coord = data.get("pick_coord") or []
        elapsed_ms = data.get("elapsed_ms", None)

        self.get_logger().info(
            f"[WRITE]   status={status}, module_type={module_type}, conf={conf}, "
            f"pick_coord_len={len(pick_coord)}, elapsed_ms={elapsed_ms}"
        )

        payload = {
            "kind": "ai_result",
            "json": json_str,
        }
        self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)

        # enable 상태도 같이 로그로 확인
        self.get_logger().info(f"[WRITE] enqueue kind=ai_result (enable_opcua={enable_opcua})")

    # --------------------------- OPCUA 비동기 루프 ---------------------------

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._writer_loop())

    async def _writer_loop(self):
        while rclpy.ok():
            payload = await self.queue.get()
            try:
                enable_opcua = bool(self.get_parameter("enable_opcua").value)
                self.get_logger().info(f"[WRITE] queue pop kind={payload.get('kind')} (enable_opcua={enable_opcua})")

                if not enable_opcua:
                    await self._send_stub(payload)
                else:
                    await self._send_to_opcua(payload)

            except Exception as e:
                self.get_logger().error(f"[WRITE] writer_loop error: {repr(e)}")

    async def _send_stub(self, payload: dict):
        url = str(self.get_parameter("opcua_url").value)
        obj_id = str(self.get_parameter("write_object_node_id").value)
        mtd_id = str(self.get_parameter("write_method_node_id").value)
        json_str = payload.get("json", "")
        self.get_logger().info(
            f"[OPCUA_STUB] would call write_send_arm_json: URL={url}, Object={obj_id}, Method={mtd_id}, json_len={len(json_str)}"
        )

    async def _send_to_opcua(self, payload: dict):
        kind = payload.get("kind")
        json_str = payload.get("json", "")

        # JSON 재검증(방어)
        try:
            _ = json.loads(json_str)
        except Exception:
            self.get_logger().error("[OPCUA] invalid json → drop")
            return

        url = str(self.get_parameter("opcua_url").value)
        obj_id = str(self.get_parameter("write_object_node_id").value)
        mtd_id = str(self.get_parameter("write_method_node_id").value)
        timeout_sec = float(self.get_parameter("opcua_call_timeout_sec").value)
        retry = int(self.get_parameter("opcua_retry").value)

        self.get_logger().info(
            f"[OPCUA] send kind={kind} → URL={url}, Object={obj_id}, Method={mtd_id}, json_len={len(json_str)}"
        )

        last_err = None
        for attempt in range(1, retry + 2):  # retry=2면 총 3번 시도
            try:
                async with Client(url) as client:
                    obj_node = client.get_node(obj_id)
                    method_node = client.get_node(mtd_id)

                    # ✅ 서버 메서드가 (String) 하나 받는 형태라면 Variant로 보내는 게 가장 안전함
                    arg = ua.Variant(json_str, ua.VariantType.String)

                    # ✅ timeout 적용
                    result = await asyncio.wait_for(
                        obj_node.call_method(method_node.nodeid, arg),
                        timeout=timeout_sec
                    )

                self.get_logger().info(f"[OPCUA] ✅ write 완료, result={result}")
                return

            except Exception as e:
                last_err = e
                self.get_logger().error(f"[OPCUA] attempt {attempt} failed: {repr(e)}")
                await asyncio.sleep(0.3)

        self.get_logger().error(f"[OPCUA] ❌ all attempts failed: {repr(last_err)}")

    # ------------------------------ 종료 처리 ------------------------------

    def destroy_node(self):
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        finally:
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
