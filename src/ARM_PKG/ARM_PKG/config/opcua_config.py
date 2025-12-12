OPCUA_SERVER_URL = "opc.tcp://172.30.1.61:0630/freeopcua/server/"

SUBSCRIBE_NODES = [
    {
        "name": "arm_go_move",
        "node_id": "ns=2;s=read_arm_go_move"
    },
    # 필요시 아래처럼 추가하면 된다.
    # {
    #     "name": "amr_mission_state",
    #     "node_id": "ns=2;s=read_amr_mission_state"
    # },
]
