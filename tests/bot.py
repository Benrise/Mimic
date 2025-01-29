import requests
import uuid
import json


bot_host = "http://84.201.168.6"
bot_port = 443

r = requests.post(f'{bot_host}:{bot_port}/get_message', 
                  json={"dialog_id": str(uuid.uuid4()),
                        "last_msg_text": "привет, как дела?",
                        "last_message_id": str(uuid.uuid4())})
print(r) # 200 is ok
json.loads(r.content)