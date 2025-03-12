# 垃圾分类桌面应用

## 速览
<table>
  <tr>
    <td align="center"><img src="/static/images/login.png" width="300px" alt="login" /><br>login</td>
    <td align="center"><img src="/static/images/home.png" width="300px" alt="home" /><br>home</td>
    <td align="center"><img src="/static/images/use_profile.png" width="300px" alt="home" />usage</td>
  </tr>
  <tr>
    <td align="center"><img src="/static/images/uploads.png" width="300px" alt="login" /><br>upload</td>
    <td align="center"><img src="/static/images/result.png" width="300px" alt="home" /><br>result</td>
    <td align="center"><img src="/static/images/user_manage.png" width="300px" alt="home" />user</td>
  </tr>
</table>

## 1. 介绍
1. 使用python+pytorch+efficientnet_b3实现
2. web服务使用flask
3. 使用electron打包成桌面应用
> ps: 
> 1. python版本不能低于 3.12
> 2. nodejs 版本不低于 20
> 

## 2. 下载依赖
python: 3.12.3
node: 20.9.0

### 2.1 后端服务依赖
```bash
python -m venv .venv
.vnev\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r .\requirements.txt
```

### 2.2 electron依赖
```bash
npm install
# 若报错执行以下
npm config set registry https://registry.npmmirror.com
npm install
```

### 2.3 下载模型到项目根目录
[点此下载](http://s7sy8cpwf.hb-bkt.clouddn.com/garbage_classifier_best_EfficientNet-B4.pth)

## 3. 运行 
1. `npm run dev` => 开发环境
2. `npm run start` => 后端服务打成`.exe`文件后
3. `npm build` => 打包

## 4. 打包
使用 `electron-builder` 及 `pyinstaller` 工具实现打包

### 4.1 打包后端
1. 运行以下命令, 使用pyinstaller打包成exe文件
    ```bash
    pyinstaller -F -w --add-data "templates;templates" --add-data "static;static" --add-data "log;log" --add-data ".\.venv\Lib\site-packages\flask_bootstrap\templates;flask_bootstrap/templates" --icon="static/favicon.png" app.py
    ```
2. 把 `templates` `static` `log` `garbage_classifier_best_EfficientNet-B4.pth` 复制到exe的同级目录下
3. 双击运行 `app.exe`
4. 打开浏览器访问: `localhost:9000`, 此时即打包成功

### 4.2 打包应用
1. 安装 `electron-builder`
   ```bash
   npm install --save-dev electron-builder
   ```
2. 执行 `npm run build`
> ps: 一定先打包后端服务

### 4.3 electron-builder 详解
package.json
```bash
{
   "build": {
       "productName": "GarbageSort", # 应用名
       "appId": "com.example.Garbage", # 应用id
       "copyright": "MeverikC", # 版权
       "directories": {
         "output": "out" # 打包文件的输出目录
       },
       "files": [
         "out/electron/**/*",
         "node_modules/",
         "package.json",
         "main.js"
       ],
       "win": { # 指定windows系统
         "icon": "static/favicon.ico", # ico, 最小256*256
         "target": [
           {
             "target": "nsis" # 使用 nsis
           }
         ],
         "verifyUpdateCodeSignature": false # 禁用链接验证
       },
       "nsis": { # nsis配置
         "oneClick": false, # 是否一键安装
         "allowElevation": true, # 允许请求提升。 如果为false，则用户必须使用提升的权限重新启动安装程序。
         "allowToChangeInstallationDirectory": true, # 允许修改安装目录
         "installerIcon": "./static/favicon.ico", # 安装图标
         "uninstallerIcon": "./static/uninstall.ico", # 卸载图标
         "installerHeaderIcon": "./static/favicon.ico", # 安装时头部图标
         "createDesktopShortcut": true, # 创建桌面图标
         "createStartMenuShortcut": true, # 创建开始菜单图标
         "shortcutName": "GarbageSort" # 图标名称
       },
       "extraResources": [ # 添加后端程序
         {
           "from": "dist", # 后端运行目录
           "to": "app/dist" # 指定目录, 与 `main.js` 中的 `return path.join(process.resourcesPath, 'app', 'dist', 'app.exe');` 目录保持一致
         }
       ]
    }
}
```

## 5. 快速体验
[点此下载](https://gitee.com/MeverikC/electron-flask/releases)