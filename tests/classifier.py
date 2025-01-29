import requests
import uuid
import json


classifier_host = "http://84.201.168.6"
classifier_port = 8080

r = requests.post(f'{classifier_host}:{classifier_port}/predict', 
                  json={"text": "Я - конкретный бот", 
                        "dialog_id": str(uuid.uuid4()), 
                        "id": str(uuid.uuid4()), 
                        "participant_index": 0})
print(r) # 200 is ok
json.loads(r.content)