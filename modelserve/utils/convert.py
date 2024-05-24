import json
import numpy as np

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)  # 将 int64 类型转换为普通的整数类型
        return json.JSONEncoder.default(self, obj)