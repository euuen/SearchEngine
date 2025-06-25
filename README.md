# 环境准备
本代码用到了chromadb、sentence_tranformer库，python版本1.11，文本嵌入模型bge-small-zh-v1.5

文本嵌套模型用这段python代码下载比较快在国区。

```python
# -*- coding:utf-8 -*-
# @Author: 喵酱
# @time: 2025 - 04 -05
# @File: miao_test.py
# desc:
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

model_name = "BAAI/_bge-small-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 保存模型到本地目录（例如 ./_bge-small-zh-v1.5）
save_path = "./_bge-small-zh-v1.5"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

model = SentenceTransformer("./_bge-small-zh-v1.5")
model.save("./bge-small-zh-v1.5")

if __name__ == '__main__':
    print(f"模型已保存到 ./bge-small-zh-v1.5")

```

# 如何使用？

请打开`engine.config.json`文件，然后修改`paths`，此参数用于指定要读取的哪些文件夹里的文件（支持递归），如

```json
"paths": [
        "/path/to/your/documents"
]
```

本库默认读取MarkDown文件作为文档，如果要添加其他文件如`txt`，请在`endswith`中添加文件后缀名，如

```json
"endswith":[
        ".md"
]
```

配置好了这些之后，运行文件`main.py`

输入指令`\loaddoc`，接下来等待chromadb自己加载好。加载完毕后，随便输入什么就可以搜索了。示例：

```sh
search in <default> ~> 什么是共产主义
=========(1～/run/media/euuen/dir/PKM/RU/社会.md+2)=========:
 # 共产主义&社会主义社会
===============================

=========(2～/run/media/euuen/dir/PKM/RU/社会.md+1)=========:
 **我要大力推进共产主义！！！**
===============================

=========(3～/run/media/euuen/dir/PKM/RU/社会.md+34)=========:
 我坚信，共产主义是可以实现的！人类的本性就是趋利避害，如果参与群体活动能获得更多的利益，那么我想应该人人都愿意参与群体活动吧。
===============================

=========(4～/run/media/euuen//PKM/RU/社会.md+3)=========:
 1. **“共产” ≠ “全民发钱躺平”**。消灭的是**生产资料私有垄断**，而非个人生活资料（你的手机、衣服仍属私人）
===============================

=========(5～/run/media/euuen/dir/PKM/RU/社会.md+36)=========:
 当每个新生儿睁开眼睛时，看到的不是家族信托基金的厚度，而是社会为他/她准备的无限可能——这才是“共产”二字的真谛。
===============================
```

# 其余参数
1. database 指定数据库的路径
2. model 指定文本嵌入模型路径
3. result_num 返回结果个数
