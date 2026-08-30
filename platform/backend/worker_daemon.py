import socket
import os
import json
import traceback

SOCKET_PATH = "/tmp/learningos_worker.sock"

def execute_code(code: str, parameters: dict) -> dict:
    # Setup the execution namespace
    # We provide the parameters in the local scope, and expect the user code
    # to define a variable named `result` containing the output data.
    local_vars = parameters.copy()
    local_vars["result"] = None
    
    try:
        # For this local-first prototype, we use exec().
        # In a real SaaS, this would be inside a Firecracker VM or gVisor sandbox.
        exec(code, local_vars)
        return {"success": True, "result": local_vars.get("result")}
    except Exception as e:
        error_info = traceback.format_exc()
        return {"success": False, "error": str(e), "traceback": error_info}

def handle_client(conn):
    try:
        data = conn.recv(8192)
        if not data:
            return
            
        payload = json.loads(data.decode('utf-8'))
        
        # Verify JSON-RPC format
        if payload.get("jsonrpc") != "2.0" or payload.get("method") != "execute_task":
            response = {"jsonrpc": "2.0", "error": "Invalid request", "id": payload.get("id")}
            conn.sendall(json.dumps(response).encode('utf-8'))
            return
            
        params = payload.get("params", {})
        code = params.get("code", "")
        parameters = params.get("parameters", {})
        
        exec_result = execute_code(code, parameters)
        
        if exec_result["success"]:
            response = {
                "jsonrpc": "2.0",
                "result": exec_result["result"],
                "id": payload.get("id")
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "error": exec_result["error"],
                "traceback": exec_result["traceback"],
                "id": payload.get("id")
            }
            
        conn.sendall(json.dumps(response).encode('utf-8'))
        
    except json.JSONDecodeError:
        response = {"jsonrpc": "2.0", "error": "Parse error", "id": None}
        conn.sendall(json.dumps(response).encode('utf-8'))
    except Exception as e:
        print(f"Worker Error: {e}")
    finally:
        conn.close()

def main():
    # Make sure the socket does not already exist
    try:
        os.unlink(SOCKET_PATH)
    except OSError:
        if os.path.exists(SOCKET_PATH):
            raise

    # Create a Unix domain socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    
    # Bind the socket to the port
    print(f"Starting up on {SOCKET_PATH}")
    sock.bind(SOCKET_PATH)
    
    # Listen for incoming connections
    sock.listen(5)
    print("Waiting for a connection...")
    
    try:
        while True:
            conn, client_address = sock.accept()
            handle_client(conn)
    except KeyboardInterrupt:
        print("Shutting down worker daemon.")
    finally:
        sock.close()
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass

if __name__ == "__main__":
    main()
