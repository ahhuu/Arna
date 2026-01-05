#!/usr/bin/env python3
"""
主启动脚本
启动GNSS数据分析器GUI
"""

import sys
import os
import traceback

def main():
    """主函数"""
    try:
        # 添加当前目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # 导入GUI模块
        from rinex_analysis_modules.gui.main_gui import MainGUI
        
        print("正在启动GNSS数据分析器...")
        
        # 创建并运行GUI
        gui = MainGUI()
        gui.run()
        
    except Exception as e:
        print(f"启动应用程序时发生错误: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        input("按任意键退出...")

if __name__ == '__main__':
    main()