"""
增强版一键入口：
1. problem→exercise→course→concept知识点回退映射；
2. Time2Vec、相对位置、时间衰减注意力；
3. 消费/门禁小时、星期、节假日Embedding；
4. 双塔召回→PLE精排。

直接运行：python main_enhanced.py
其他参数与main.py完全一致，例如：python main_enhanced.py --quick
"""
from __future__ import annotations

import os
from pathlib import Path

# 让默认 ../my_output 与输出目录始终相对本脚本，而不是用户当前终端目录。
os.chdir(Path(__file__).resolve().parent)

from main import main


if __name__ == "__main__":
    main()

