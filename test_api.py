import requests

PAYLOAD = {
    "id": 1,
    "sttl": 64,
    "ct_state_ttl": 0,
    "dload": 0.0,
    "ct_dst_sport_ltm": 0,
    "dmean": 0.0,
    "rate": 0.1,
    "swin": 0.0,
    "dwin": 0.0,
    "ct_src_dport_ltm": 0,
    "ct_dst_src_ltm": 0
}


def test_flask_predict():
    res = requests.post("http://127.0.0.1:5000/predict", json=PAYLOAD, timeout=5)
    assert res.status_code == 200
    assert "result" in res.json()


def test_fastapi_predict():
    res = requests.post("http://127.0.0.1:8001/predict", json=PAYLOAD, timeout=5)
    assert res.status_code == 200
    assert "prediction" in res.json()
