import os
import base64
import pandas as pd
from zhipuai import ZhipuAI
import time
from prompt_config import PROMPT
API_KEY = "0282c89d46a14b7ab76c67479890a1a3.y2HSvz5NUVjha1u4"

client = ZhipuAI(api_key=API_KEY)

def get_description(img_path):
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    content = [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": img_base64}},
        {"type": "text", "text": "请根据上述角色和流程，分析该图像。"}
    ]
    try:
        response = client.chat.completions.create(
            model="glm-4v-plus-0111",
            messages=[{"role": "user", "content": content}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return ""

def process_folder(folder, label):
    data = []
    count = 0
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            img_path = os.path.join(folder, filename)
            print(f"Processing {img_path} ...")
            desc = get_description(img_path)
            if desc:
                data.append({"text": desc, "label": label})
                count += 1
            time.sleep(1.5)  # 防止API限流
            # 每10条写入一次
            if count % 10 == 0 and data:
                df = pd.DataFrame(data)
                df.to_csv("data.csv", mode='a', index=False, encoding="utf-8", header=not os.path.exists("data.csv"))
                data.clear()
    # 剩余不足10条的也要写入
    if data:
        df = pd.DataFrame(data)
        df.to_csv("data.csv", mode='a', index=False, encoding="utf-8", header=not os.path.exists("data.csv"))

if __name__ == "__main__":
    # 运行前建议先删除旧的data.csv，避免重复追加
    if os.path.exists("data.csv"):
        os.remove("data.csv")
    malicious_folder = "data/cyberbullying"
    normal_folder = "data/non_cyberbullying"
    process_folder(malicious_folder, 1)
    process_folder(normal_folder, 0)
    print("已生成 data.csv")