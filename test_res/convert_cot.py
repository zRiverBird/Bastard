import re
import sys
import json


data = json.load(open(sys.argv[1]))
res = []
for i in data:
    try:
        i["answer"] = re.findall(r'<answer>(.*?)</answer>', i["answer"], re.DOTALL)[0].strip()
        res.append(i)
    except:
        print(i["answer"])

json.dump(res, open("cot_results.json", 'w'), indent=2, ensure_ascii=False)





