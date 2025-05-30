from flask import Flask, request, jsonify, send_from_directory
import base64
from zhipuai import ZhipuAI
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from prompt_config import PROMPT
app = Flask(__name__)

API_KEY = "0282c89d46a14b7ab76c67479890a1a3.y2HSvz5NUVjha1u4"

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# 静态文件路由，支持background.mp4等资源访问
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


# 加载BERT模型和分词器（只需加载一次，放在全局）
bert_tokenizer = BertTokenizer.from_pretrained('./bert_cls_model')
bert_model = BertForSequenceClassification.from_pretrained('./bert_cls_model')
bert_model.eval()

def classify_text(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        confidence = float(probs[pred])
    # 假设标签0为正常，1为恶意
    label = "恶意" if pred == 1 else "正常"
    return label, confidence

@app.route('/api/detect', methods=['POST'])
def detect():
    file = request.files['file']
    file_type = file.content_type
    file_bytes = file.read()
    file_base64 = base64.b64encode(file_bytes).decode('utf-8')
    client = ZhipuAI(api_key=API_KEY)
    if file_type.startswith('image/'):
        content = [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": file_base64}},
            {"type": "text", "text": "请根据上述角色和流程，分析该图像。"}
        ]
    elif file_type.startswith('video/'):
        content = [
            {"type": "text", "text": PROMPT},
            {"type": "video_url", "video_url": {"url": file_base64}},
            {"type": "text", "text": "请根据上述角色和流程，分析该视频。"}
        ]
    elif file_type == 'video/x-msvideo' or file.filename.lower().endswith('.avi'):
        return jsonify({"error": "暂不支持 avi 格式，请上传 mp4、webm 等主流视频格式"}), 400
    else:
        return jsonify({"error": "不支持的文件类型"}), 400

    try:
        response = client.chat.completions.create(
            model="glm-4v-plus-0111",
            messages=[{"role": "user", "content": content}]
        )
        reply = response.choices[0].message.content

        # 用BERT对大模型描述文本进行分类
        label, confidence = classify_text(reply)
        return jsonify({
            "chain": reply,
            "result": f"分析完成<br>BERT判定：{label}<br>置信度：{confidence:.2f}"
        })
    except Exception as e:
        return jsonify({"chain": "模型调用失败", "result": "错误"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True )  # 设置debug=True以便于调试
