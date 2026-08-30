import socket
import json
import uuid

SOCKET_PATH = "/tmp/learningos_worker.sock"

class WorkerClient:
    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path

    def execute(self, code: str, parameters: dict) -> dict:
        req_id = f"req_{uuid.uuid4().hex}"
        request_payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "execute_task",
            "params": {
                "code": code,
                "parameters": parameters,
                "limits": {"timeout_sec": 30, "memory_mb": 2048}
            }
        }
        
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.socket_path)
            
            client.sendall(json.dumps(request_payload).encode("utf-8"))
            
            response_data = client.recv(65536)
            client.close()
            
            if not response_data:
                return {"error": "No response from worker"}
                
            response = json.loads(response_data.decode("utf-8"))
            if "error" in response:
                return {"error": response["error"]}
                
            return {"result": response.get("result", {})}
            
        except FileNotFoundError:
            return {"error": f"Worker socket not found at {self.socket_path}. Is the daemon running?"}
        except Exception as e:
            return {"error": f"Worker communication error: {str(e)}"}
