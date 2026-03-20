#!/usr/bin/env python3
"""
调试代码提取逻辑
"""

import re

# 模拟 LLM 返回的内容（包含代码块）
sample_response = '''以下是生成的代码：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试代码
"""

import pandas as pd

def test():
    print("Hello")
```

一些说明文字...

```python
def another_function():
    pass
```
'''

print("原始内容:")
print("=" * 60)
print(sample_response[:200])
print("...")
print("=" * 60)
print()

# 测试正则表达式
code_blocks = re.findall(r'```python\n(.*?)\n```', sample_response, re.DOTALL)
print(f"匹配到的代码块数量: {len(code_blocks)}")
for i, block in enumerate(code_blocks):
    print(f"\n代码块 {i+1}:")
    print(block[:100])
    print("...")

# 合并代码块
if code_blocks:
    extracted_code = '\n\n'.join(code_blocks)
    print("\n" + "=" * 60)
    print("提取后的代码:")
    print("=" * 60)
    print(extracted_code[:200])
