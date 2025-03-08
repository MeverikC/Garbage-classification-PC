# 垃圾分类桌面应用

## 介绍

1. 使用python+pytorch+efficientnet_b3实现
2. web服务使用flask
3. 使用electron打包成桌面应用

## 下载依赖
// TODO 说明文档

### 后端服务依赖
// TODO 
### electron依赖
// TODO 
## 运行
// TODO 说明文档

## 打包后端
```bash
.\.venv\Scripts\pyinstaller.exe -F -w --add-data "templates;templates" --add-data "static;static" --add-data "log;log" --add-data ".\.venv\Lib\site-packages\flask_bootstrap\templates;flask_bootstrap/templates" --icon="static/favicon.png" app.py
```

## 打包应用
```bash
npm config set registry https://registry.npmmirror.com
npm install --save-dev @electron-forge/cli
npx electron-forge import
npm run make // npm run make --offline
```
