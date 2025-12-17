# insert_arm_place_completed.py
import pymysql
from db_config import DB_CONFIG

def report_arm_completed(mission_id):
    """
    모든 물체 적재 완료 시 호출 (예: mission_logs 테이블의 상태 업데이트 등)
    현재는 robotarm_logs에 최종 완료 메시지를 남기는 것으로 구성
    """
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO mission_robotarm_logs 
            (mission_id, action_type, result_status, result_message, description)
            VALUES (%s, 'WAIT', 'SUCCESS', 'ALL_COMPLETED', '모든 물체 적재 완료 및 컨베이어 종료')
            """
            cursor.execute(sql, (mission_id,))
            conn.commit()
            print(f"Mission {mission_id} Completed reported to DB.")
    except Exception as e:
        print(f"Database Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()