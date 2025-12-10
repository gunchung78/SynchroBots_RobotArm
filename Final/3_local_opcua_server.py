import asyncio
import json
import logging
import sys
from asyncua import Server, ua
import base64
import numpy as np
import cv2
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- 서버 설정 ---
LOCAL_SERVER_URL = "opc.tcp://127.0.0.1:4840/freeopcua/server/"
URI = "http://examples.freeopcua.github.io"
OBJECT_NODE_ID = 3
METHOD_NODE_ID = 25
# 전역 변수로 cmd_var 선언 (main 함수와 key_input_task에서 접근)
cmd_var = None
# ===============================================
# :메모: OPC UA Method 콜백 함수 (로봇 -> 서버 통신)
# ===============================================
async def method_callback(parent, json_string):
    """
    로봇 클라이언트(5_OPCUA_Pick.py)로부터 JSON 데이터를 수신하는 콜백.
    (로봇 코드의 send_mission_state / send_vision_result 에 대응)
    """
    logger.info(f"\n[SERVER] :로켓: Method Call 수신: {parent.Identifier}")
    logger.info(f"[SERVER] :종: 수신된 JSON 데이터: {json_string}")
    # :압정: OPC UA 통신 결과 반환 (기본값)
    # ResultCode: 1 (Success) 또는 2 (JSON Error)
    result_code = ua.Variant(1, ua.VariantType.Int32)
    result_message = ua.Variant("Success", ua.VariantType.String)
    try:
        # :압정: 1. Variant 객체에서 문자열 값 추출
        if hasattr(json_string, 'Value'):
            data_string = json_string.Value
        else:
            data_string = json_string
        # :압정: 2. JSON 파싱
        data = json.loads(data_string)
        # :압정: 3. 이미지 데이터 처리 (Base64 포함된 경우)
        base64_img_str = data.get("img")
        if base64_img_str:
            try:
                # 1. Base64 디코딩 (ASCII 문자열 -> 바이트)
                img_bytes = base64.b64decode(base64_img_str)
                # 2. 바이트 -> Numpy 배열 (JPEG 디코딩)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                # 3. 이미지 사용 (예: 파일로 저장)
                # 파일 이름을 유니크하게 저장하고 싶다면 time.time() 등을 사용
                cv2.imwrite("received_object_image.jpg", decoded_img)
                print(":액자에_담긴_그림: 서버에서 이미지 복원 및 저장 완료: received_object_image.jpg")
            except Exception as e:
                logger.error(f"서버에서 이미지 복원 중 오류: {e}")
                # 이미지 복원 오류는 전체 JSON 오류로 처리하지 않고 로깅만 함
        # :압정: 4. 미션 상태 데이터 처리 (가장 간단한 데이터 먼저 확인)
        if 'status' in data:
            # 미션 상태 업데이트 (예: arm_mission_success)
            print(f"[SERVER] :반짝임: 미션 상태 보고: {data['status']}")
        # :압정: 5. 비전 분류 결과 데이터 처리 (status에 없으면 비전 결과로 간주)
        elif 'module_type' in data:
            print("[SERVER] :한_쪽_눈: 비전 결과 보고:")
            print(f" - Module Type: {data.get('module_type')}")
            print(f" - Confidence: {data.get('classification_confidence')}")
            print(f" - Pick Coord: {data.get('pick_coord')}")
            print(f" - Coord Conf: {data.get('pick_coord_confidence')}")
            print(f" - Img: {data.get('img')}")
    except json.JSONDecodeError:
        logger.error("[SERVER] :x: JSON 디코딩 오류 발생. 수신된 데이터: " + data_string[:100] + "...")
        result_code = ua.Variant(2, ua.VariantType.Int32)
        result_message = ua.Variant("JSON Decode Error", ua.VariantType.String)
    except Exception as e:
        logger.error(f"[SERVER] :x: 알 수 없는 오류 발생: {e}")
        result_code = ua.Variant(3, ua.VariantType.Int32)
        result_message = ua.Variant(f"Unknown Error: {e}", ua.VariantType.String)
    return [result_code, result_message]
# ===============================================
# :키보드: 사용자 키 입력 처리 태스크 (서버 -> 로봇 통신)
# ===============================================
def blocking_input():
    """블로킹(동기) 함수: 사용자 입력을 기다립니다."""
    return sys.stdin.read(1)
async def key_input_task():
    """키 입력을 비동기적으로 처리하고 OPC UA 변수에 값을 쓰는 태스크."""
    global cmd_var
    # cmd_var가 설정될 때까지 기다림
    while cmd_var is None:
        await asyncio.sleep(0.1)
    print("\n[SERVER CMD] :세로형_신호등: 명령 대기 중: '1' (go_home) 또는 '2' (mission_start)를 누르세요.")
    while True:
        try:
            # 비동기적으로 블로킹 입력 함수 실행
            key = await asyncio.to_thread(blocking_input)
            key = key.strip()
            if key == '1':
                command_value = "go_home"
                logger.info(f"[SERVER CMD] :오른쪽_화살표: '1' 감지. {command_value} 명령 전송 중...")
                await cmd_var.write_value(command_value)
                logger.info(f"[SERVER CMD] :흰색_확인_표시: go_home 명령 전송 완료.")
            elif key == '2':
                # 복잡한 명령은 JSON 문자열로 전송
                command_value = '{"move_command": "mission_start"}'
                logger.info(f"[SERVER CMD] :오른쪽_화살표: '2' 감지. mission_start 명령 전송 중...")
                await cmd_var.write_value(command_value)
                logger.info(f"[SERVER CMD] :흰색_확인_표시: mission_start 명령 전송 완료.")
            elif key == '':
                # 엔터 키는 무시
                continue
            else:
                print(f"[SERVER CMD] :경고: 알 수 없는 키 입력 '{key}' 무시.")
        except EOFError:
            # 입력 스트림 종료 시 (예: 파이프)
            logger.warning("[SERVER CMD] 입력 스트림 종료 감지.")
            break
        except Exception as e:
            logger.error(f"[SERVER CMD] 키 입력 처리 중 오류 발생: {e}")
            await asyncio.sleep(1)
# ===============================================
# :데스크톱_컴퓨터: 메인 서버 실행 함수
# ===============================================
async def main():
    global cmd_var
    logger.info("OPC UA 로컬 테스트 서버 초기화 시작...")
    server = Server()
    server.set_endpoint(LOCAL_SERVER_URL)
    # AddressSpace 초기화 및 시작
    await server.init()
    await server.start()
    # 네임스페이스 등록
    idx = await server.register_namespace(URI)
    try:
        logger.info(f"OPC UA 서버 시작 완료. 엔드포인트: {LOCAL_SERVER_URL}")
        # 1. ARM Object 및 Method 생성 (로봇 -> 서버 통신용)
        arm_object = await server.nodes.objects.add_object(
            ua.NodeId(OBJECT_NODE_ID, idx), "ARM_Mission_Comm"
        )
        inarg = ua.Argument()
        inarg.Name = "json_input"
        inarg.DataType = ua.NodeId(ua.ObjectIds.String)
        inarg.ValueRank = -1
        outarg1 = ua.Argument()
        outarg1.Name = "result_code"
        outarg1.DataType = ua.NodeId(ua.ObjectIds.Int32)
        outarg1.ValueRank = -1
        outarg2 = ua.Argument()
        outarg2.Name = "result_message"
        outarg2.DataType = ua.NodeId(ua.ObjectIds.String)
        outarg2.ValueRank = -1
        await arm_object.add_method(
            ua.NodeId(METHOD_NODE_ID, idx),
            "write_arm_mission_state",
            method_callback,
            [inarg],
            [outarg1, outarg2]
        )
        # 2. Command Variable 생성 (서버 -> 로봇 통신용)
        arm_folder = await server.nodes.objects.add_folder(idx, "ARM")
        # cmd_var 전역 변수에 할당
        cmd_var = await arm_folder.add_variable(
            idx,
            "read_arm_go_move",
            ua.Variant("", ua.VariantType.String)
        )
        await cmd_var.set_writable()
        logger.info("OPC UA 노드 구조 설정 완료.")
        logger.info("서버가 시작되었습니다. 로봇 클라이언트 연결을 기다립니다...")
        # 3. 키 입력 처리 태스크 시작
        input_task = asyncio.create_task(key_input_task())
        # 서버 유지 루프
        await asyncio.gather(input_task, keep_server_alive())
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {e}")
        await server.stop()
    finally:
        await server.stop()
async def keep_server_alive():
    """서버가 종료되지 않도록 무한 루프를 유지합니다."""
    while True:
        await asyncio.sleep(5)
if __name__ == "__main__":
    try:
        # sys.stdin이 Non-blocking이 아닐 경우 Windows 등에서 에러 발생 방지
        # asyncio.run()이 메인 루프를 실행하고 서버를 유지합니다.
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n사용자 중단 (Ctrl+C). 서버 종료.")