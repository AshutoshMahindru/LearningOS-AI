import socket
import json
import traceback
import sys
import os

SOCKET_PATH = "/tmp/learningos_worker.sock"

def execute_task(params):
    # Mock implementation of dynamic code execution for the foundation phase.
    # In full implementation, this will exec() or import the user code safely.
    code = params.get("code", "")
    return {
        "output": f"Mock executed {len(code)} bytes of code.",
        "artifacts": []
    }

def handle_request(conn):
    try:
        data = conn.recv(8192)
        if not data:
            return
            
        req = json.loads(data.decode("utf-8"))
        if req.get("jsonrpc") != "2.0":
            return
            
        method = req.get("method")
        params = req.get("params", {})
        req_id = req.get("id")
        
        result = None
        error = None
        
        if method == "execute_task":
            try:
                result = execute_task(params)
            except Exception as e:
                error = {"code": -32000, "message": str(e), "data": traceback.format_exc()}
        else:
            error = {"code": -32601, "message": "Method not found"}
            
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
        }
        if error:
            response["error"] = error
        else:
            response["result"] = result
            
        conn.sendall(json.dumps(response).encode("utf-8"))
        
    except Exception as e:
        print(f"Error handling request: {e}", file=sys.stderr)

def main():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
        
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    
    print(f"Worker daemon listening on {SOCKET_PATH}")
    
    try:
        while True:
            conn, _ = server.accept()
            handle_request(conn)
            conn.close()
    except KeyboardInterrupt:
        print("Worker shutting down.")
    finally:
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

if __name__ == "__main__":
    main()
