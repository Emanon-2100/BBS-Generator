# configurator.py (终极魔改版)

import sys
from ast import literal_eval

print("--- Running Magic-Modified Configurator ---")

config_file = None
# 优先寻找配置文件
for arg in sys.argv[1:]:
    if not arg.startswith('--'):
        config_file = arg
        break

# 如果找到了配置文件，就用它来设置全局变量
if config_file:
    print(f"Loading base config from: {config_file}")
    with open(config_file, 'r', encoding='utf-8') as f:
        code = f.read()
        # 直接执行文件内容，这会创建或覆盖全局变量
        exec(code, globals())

# 然后，再用命令行参数来覆盖
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        key, val = arg.split('=', 1)
        key = key[2:]
        
        try:
            # 尝试将值转换为Python对象 (int, float, bool, etc.)
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            # 如果失败，就直接用字符串
            attempt = val
        
        # 直接覆盖或创建全局变量，不管它之前是否存在或类型是什么
        print(f"Command-line override: {key} = {repr(attempt)}")
        globals()[key] = attempt

print("--- Configurator finished ---")