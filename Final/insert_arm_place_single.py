# insert_arm_place_single.py
import pymysql
from db_config import DB_CONFIG

def insert_arm_log(mission_id, action_type, target_pose, result_status, result_message, module_type, description):
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO mission_robotarm_logs 
            (mission_id, action_type, target_pose, result_status, result_message, module_type, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (mission_id, action_type, str(target_pose), result_status, result_message, module_type, description))
            conn.commit()
    except Exception as e:
        print(f"Database Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()