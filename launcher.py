import subprocess
import webbrowser
import time
import os
import sys
from pathlib import Path

def is_frozen():
    return getattr(sys, 'frozen', False)

def main():
    print("正在启动知识追踪系统...")
    
    # 获取正确的工作目录
    if is_frozen():
        # 在打包环境中，使用临时目录
        script_dir = Path(sys._MEIPASS)
    else:
        # 在开发环境中，使用当前目录
        script_dir = Path(__file__).parent
    
    os.chdir(script_dir)
    print(f"当前工作目录: {script_dir}")
    
    # 确保logs目录存在
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    try:
        # 检查必要的文件是否存在
        app_py = script_dir / "app.py"
        if not app_py.exists():
            raise FileNotFoundError(f"app.py文件不存在: {app_py}")
        
        print(f"找到app.py文件: {app_py}")
        
        # 检查data和save目录
        data_dir = script_dir / "data"
        save_dir = script_dir / "save"
        print(f"data目录存在: {data_dir.exists()}")
        print(f"save目录存在: {save_dir.exists()}")
        
        # 直接导入并运行Streamlit，避免递归调用
        print("启动Streamlit服务器...")
        
        if is_frozen():
            # 在打包环境中，直接导入运行
            print("在打包环境中运行Streamlit...")
            import streamlit
            print("Streamlit模块导入成功")
            streamlit.run(str(app_py), server_port=8501, server_address="localhost")
        else:
            # 在开发环境中，正常运行
            print("在开发环境中运行Streamlit...")
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--browser.gatherUsageStats",
                "false"
            ]
            subprocess.run(cmd)
        
    except Exception as e:
        print(f"启动失败: {e}")
        print("请检查系统是否安装了所有必要的依赖")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")

if __name__ == "__main__":
    main()