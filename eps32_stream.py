#Display esp32 stream

import cv2
import numpy as np
import requests

#url = "http://192.168.1.154/capture?_cb=1651821223485"
url = "http://192.168.1.154:81/stream"
resp = requests.get(url, stream=True).raw
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)

cv2.imshow('ESP32 Cam image', image)
cv2.waitKey(1)
cv2.destroyAllWindows()
