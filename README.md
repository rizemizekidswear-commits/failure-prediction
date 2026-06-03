import requests

response = requests.post('http://127.0.0.1:5000/predict', json={
    'air_temp': 304.0,
    'process_temp': 314.0,
    'rot_speed': 1200,
    'torque': 68.0,
    'tool_wear': 200
})

print(response.json())
