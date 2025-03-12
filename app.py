# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bootstrap import Bootstrap
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import module
from datetime import datetime
import logging
import signal
import sys


# 获取当前可执行文件所在目录
base_path = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(base_path, "templates"),
    static_folder=os.path.join(base_path, "static")
)

app.secret_key = 'supersecretkey'  # 设置密钥用于会话管理

bootstrap = Bootstrap(app)
# 配置 Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'



# JSON 数据存储路径
DATA_FILE = "static/data/data.json"
now = datetime.now().strftime('%Y-%m-%d')
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志文件路径、格式及日志级别
logging.basicConfig(
    filename=f"{log_dir}/app.log",  # 指定日志文件路径
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def signal_handler(sig, frame):
    logging.info('Flask server is shutting down...')
    sys.exit(0)  # 确保进程退出

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# 加载 JSON 数据
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # 如果文件内容为空或格式错误，返回默认数据
                return {"users": {}, "results": []}
    return {"users": {}, "results": []}


# 保存 JSON 数据
def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)


# Flask-Login 用户加载器
@login_manager.user_loader
def load_user(user_id):
    user_data = module.USERS.get(user_id)
    if user_data:
        return module.User(user_id, user_data["password"], user_data["role"])
    return None


# 登录页面
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = module.User.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password")
            return render_template('login.html')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# 配置参数
class WebConfig:
    model_path = "./garbage_classifier_best_EfficientNet-B4.pth"
    label_mapping = "./static/data/class_indices.json"
    upload_folder = './static/uploads'
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224  # 需要与训练时保持一致


# 确保上传目录存在
os.makedirs(WebConfig.upload_folder, exist_ok=True)

# 数据预处理（与验证集相同）
transform = transforms.Compose([
    transforms.Resize(int(WebConfig.img_size * 1.1)),
    transforms.CenterCrop(WebConfig.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 加载模型
def load_model():
    # 加载类别映射
    with open(WebConfig.label_mapping, 'r') as f:
        label_data = json.load(f)
        class_names = label_data['classes']
        class_to_idx = label_data['class_to_idx']

    # 创建模型（需要与训练时的结构一致）
    model = models.efficientnet_b4(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, len(class_names))
    )

    # 加载权重
    checkpoint = torch.load(WebConfig.model_path, map_location=WebConfig.device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(WebConfig.device)
    model.eval()

    return model, class_names


model, class_names = load_model()


# 辅助函数
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in WebConfig.allowed_extensions


def predict_image(image_path):
    try:
        # 打开并预处理图像
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(WebConfig.device)

        # 推理
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # 获取top3结果
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        results = []
        for i in range(3):
            results.append({
                "class": class_names[top3_indices[i].item()],
                "confidence": f"{top3_probs[i].item() * 100:.2f}%"
            })
        return {"success": True, "predictions": results}

    except Exception as e:
        return {"success": False, "error": str(e)}


# 主页
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if current_user.role == 'admin':
        data = load_data()
        user_stats = {user: len(results) for user, results in data["users"].items()}
        total_correct = sum(1 for result in data["results"] if result["correct_result"])
        total_results = len(data["results"])
        accuracy = (total_correct / total_results * 100) if total_results > 0 else 0  # 计算准确率

        # 模型生成总数与准确率
        model_stats = {'total_results': total_results, 'success_num': total_correct}
        return render_template('dashboard.html', user_stats=user_stats, model_stats=model_stats)
    return redirect(url_for('upload_file'))


# 路由定义
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # 保存上传的文件
            filepath = os.path.join(WebConfig.upload_folder, file.filename)
            file.save(filepath)

            # 进行预测
            result = predict_image(filepath)

            # # 删除临时文件
            # if os.path.exists(filepath):
            #     os.remove(filepath)

            if result['success']:
                return render_template('result.html',
                                       predictions=result['predictions'],
                                       filename=file.filename,
                                       file=file)
            else:
                return jsonify(result), 500

    return render_template('upload.html')


# 保存用户选择的结果
@app.route('/save_result', methods=['POST'])
@login_required
def save_result():
    data = request.json
    username = current_user.id
    model_results = data.get("model_results")
    correct_result = data.get("correct_result")
    filename = data.get("filename")

    if correct_result is None or correct_result == "":
        correct_result = "无正确结果"
    saved_data = load_data()
    saved_data["results"].append({
        "username": username,
        "model_results": model_results,
        "correct_result": correct_result,
        "created_at": now,
        "filename":  filename
    })
    if username not in saved_data["users"]:
        saved_data["users"][username] = []

    saved_data["users"][username].append({
        "username": username,
        "model_results": model_results,
        "correct_result": correct_result,
        "created_at": now,
        "filename": filename
    })
    save_data(saved_data)
    flash("success")
    return redirect(url_for('upload_file'))


# 管理用户页面
@app.route('/admin/users', methods=['GET'])
@login_required
def manage_users():
    if current_user.role != 'admin':
        return redirect(url_for('index'))
    return render_template('manage_users.html', users=module.USERS)


@app.route('/add-users', methods=['POST'])
@login_required
def add_user():
    if current_user.role != 'admin':
        return redirect(url_for('index'))
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')
    result = module.User.add_user(username, password, role)
    if result:
        flash('用户添加成功')
    else:
        flash('该用户已存在')

    return redirect(url_for('manage_users'))


@app.route('/edit-users', methods=['POST'])
@login_required
def edit_user():
    if current_user.role != 'admin':
        return redirect(url_for('index'))
    data = request.json
    edit_username = data.get('editUsername')
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')
    result = module.User.edit_user(edit_username, username, password, role)
    if result:
        flash('编辑成功')
    else:
        flash('编辑失败')

    return redirect(url_for('manage_users'))


@app.route('/delete-users', methods=['POST'])
@login_required
def delete_user():
    if current_user.role != 'admin':
        return redirect(url_for('index'))
    data = request.json
    username = data.get('username')
    result = module.User.delete_user(username)
    if result:
        flash('删除成功')
    else:
        flash('删除失败')

    return redirect(url_for('manage_users'))


# 用户页面
@app.route('/user/<username>')
@login_required
def user_profile(username):
    if current_user.role != 'admin' and current_user.id != username:
        flash("You do not have permission to view this page.")
        return redirect(url_for('index'))

    # 加载数据
    data = load_data()
    user_results = data["users"].get(username, [])

    # 统计信息
    total_uploads = len(user_results)
    correct_results = sum(1 for result in user_results if result["correct_result"])
    accuracy = (correct_results / total_uploads * 100) if total_uploads > 0 else 0

    return render_template('user_profile.html', username=username, results=user_results, total_uploads=total_uploads,
                           accuracy=accuracy)


if __name__ == '__main__':
    print('Flask server is ready')
    app.run(host='0.0.0.0', port=9000, debug=True)
