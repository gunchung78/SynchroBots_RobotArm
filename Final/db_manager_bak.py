import aiomysql
import logging

class DBManager:
    def __init__(self):
        self.config = {
            'host': '172.30.1.29',
            'user': 'root',
            'password': '1234',
            'db': 'SynchroBots',
            'port': 3306,
            'autocommit': True
        }
        self.logger = logging.getLogger("DBManager")

    async def get_pool(self):
        return await aiomysql.create_pool(**self.config)

    # mission_logs 삽입 및 ID 반환
    async def insert_mission_start(self):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                sql = "INSERT INTO mission_logs (equipment_id, status) VALUES (%s, %s)"
                await cur.execute(sql, ('ARM01', 'RUNNING'))
                mission_id = cur.lastrowid
                return mission_id

    # mission_logs 상태 업데이트 (RUNNING -> DONE/ERROR)
    async def update_mission_status(self, mission_id, status='DONE'):
        print(f"update_mission_status {status} {mission_id}")
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                sql = "UPDATE mission_logs SET status = %s WHERE mission_id = %s"
                await cur.execute(sql, (status, mission_id))

    # 상세 로그 삽입 (mission_robotarm_logs)
    async def insert_arm_log(self, mission_id, action_type, target_pose=None, 
                            result_status='SUCCESS', result_message=None, 
                            module_type=None, description=None):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                sql = """
                INSERT INTO mission_robotarm_logs 
                (mission_id, action_type, target_pose, result_status, result_message, module_type, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                # 리스트 형태의 pose를 문자열로 변환
                pose_str = str(target_pose) if target_pose else None
                await cur.execute(sql, (mission_id, action_type, pose_str, 
                                        result_status, result_message, module_type, description))