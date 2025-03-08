const { app, BrowserWindow } = require("electron");
const path = require("path");
const isDev = process.env.NODE_ENV === 'development';

let mainWindow = null;
let subpy = null;

const PY_DIST_FOLDER = "dist"; // python 可分发文件夹
const PY_SRC_FOLDER = ""; // Python 源代码的路径
const PY_MODULE = "app.py"; // 主模块的名称
const PYTHON_EXECUTABLE_NAME = "python.exe";
const APP_ICON_PATH = './static/favicon.png';

const getPythonScriptPath = () => {
  if (PY_DIST_FOLDER === '') {
    if (PY_SRC_FOLDER === ''){
      return path.join(__dirname, PY_MODULE);
    }else {
      return path.join(__dirname, PY_SRC_FOLDER, PY_MODULE);
    }
  }
  if (process.platform === "win32") {
    return path.join(
      __dirname,
      PY_DIST_FOLDER,
      PY_MODULE.slice(0, -3) + ".exe"
    );
  }
  return '';
};

const getPythonScriptPath2 = () => {
  if (isDev) {
    // 开发环境直接使用项目目录
    return path.join(__dirname, PY_DIST_FOLDER, 'app.exe');
  }
  // 生产环境使用解压后的资源目录
  return path.join(process.resourcesPath, 'dist', 'app.exe'); 
};

const startPythonSubprocess = () => {
  console.log(getPythonScriptPath2());
  if (PY_DIST_FOLDER === ''){
    subpy = require("child_process").spawn('python', [getPythonScriptPath2()]);
  }else {
    subpy = require("child_process").execFile(getPythonScriptPath2(), []);
  }


  //监听 Python 进程的标准输出
  subpy.stdout.on("data", (data) => {
    const output = data.toString();
    if (output.includes("Flask server is ready")) {
      console.log("Flask server is ready, creating main window...");
      createMainWindow(); // 在flask准备好后创建窗口
    }
  });

  // 监听 Python 进程中的错误
  subpy.stderr.on("error", (err) => {
    console.error(`Python subprocess error: ${err}`);
  });

  // 处理 Python 进程退出
  subpy.on("close", (code) => {
    console.log(`Python subprocess exited with code ${code}`);
    subpy = null;
  });
};

const killPythonSubprocesses = (main_pid) => {
  let python_processes_name = '';
  if (PY_DIST_FOLDER === ''){
    python_processes_name = PYTHON_EXECUTABLE_NAME
  }else {
    python_processes_name = PY_MODULE.slice(0, -3) + ".exe"
  }
  let cleanup_completed = false;
  const psTree = require("ps-tree");
  psTree(main_pid, function (err, children) {
    let python_pids = children
      .filter(function (el) {
        return el.COMMAND === python_processes_name;
      })
      .map(function (p) {
        return p.PID;
      });
    // 终止所有生成的 python 进程
    python_pids.forEach(function (pid) {
      process.kill(pid);
    });
    subpy = null;
    cleanup_completed = true;
  });
  return new Promise(function (resolve, reject) {
    (function waitForSubProcessCleanup() {
      if (cleanup_completed) return resolve();
      setTimeout(waitForSubProcessCleanup, 30);
    })();
  });
};

const createMainWindow = () => {
  // 创建浏览器窗口
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    resizable: true,
    icon: APP_ICON_PATH,
  });

  mainWindow.loadURL("http://localhost:9000/");

  // 打开发开着工具
  // mainWindow.webContents.openDevTools();

  // 当窗口关闭时
  mainWindow.on("closed", function () {
    // 取消引用窗口对象
    mainWindow = null;
  });
};

app.on("ready", function () {
  // 开启后端服务
  startPythonSubprocess();
});

// 禁用菜单
app.on("browser-window-created", function (e, window) {
  window.setMenu(null);
});

// 当全部窗口关闭时调用
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    let main_process_pid = process.pid;
    killPythonSubprocesses(main_process_pid).then(() => {
      app.quit();
    });
  }
});

app.on("activate", () => {
  // 在 macOS 上，当单击停靠图标且没有打开其他窗口时，通常会在应用程序中重新创建一个窗口。
  if (subpy == null) {
    startPythonSubprocess();
  }
  if (mainWindow === null) {
    createMainWindow();
  }
});

app.on("quit", function () {
  // 做一些额外的清理
});