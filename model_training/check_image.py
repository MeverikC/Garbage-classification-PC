import os
from PIL import Image

def check_images(data_dir):
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            try:
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path)
                img.verify()  # 检查图像是否损坏
            except (IOError, SyntaxError) as e:
                print(f"删除损坏的图像文件：{img_path}")
                os.remove(img_path)
