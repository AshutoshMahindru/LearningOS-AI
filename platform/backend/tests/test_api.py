from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["version"] == "3.0.0"
    print("Health check endpoint OK.")

if __name__ == "__main__":
    try:
        import httpx
    except ImportError:
        print("Please install httpx to run TestClient.")
    else:
        test_health_check()
        print("All tests passed.")
