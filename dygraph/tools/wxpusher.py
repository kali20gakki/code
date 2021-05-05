import requests
import json

class WxPusher():
    def __init__(self):
        super().__init__()
        self.post_api = "http://wxpusher.zjiecode.com/api/send/message"
        self.app_token = "AT_BewdK3Hm1F4l2AFhLSrpJvrsXWU9BWZA"
        self.uids = ["UID_EB2XOY3Ys4w93s34pWjVXReL76jq"]
        self.headers={
            "Content-Type": "application/json"
        }

    def send_msg(self, content):
        data = {
            "appToken":self.app_token,
            "content":content,
            "contentType":1,
            "uids":self.uids,
        }
        try:
            r = requests.post(self.post_api, data=json.dumps(data), headers=self.headers)
        except:
            print("发送失败")



pusher = WxPusher()
 
if __name__ == "__main__":
    test = WxPusher()
    test.send_msg("hhhhhhhhhhhh")