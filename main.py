import json
import os

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BertTokenizerFast

# 改变工作路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class SearchEngine:
    client: chromadb.ClientAPI
    collection: chromadb.Collection
    collection_name: str = "default"
    model: SentenceTransformer
    # 注意注意，这个类型不一定是Bert
    tokenizer: BertTokenizerFast
    result_num: int
    paths: list
    endswith: list

    def __init__(self):
        with open("engine.config.json") as configFile:
            print("初始化中......", end="  ")
            config = json.load(configFile)

            self.client = chromadb.PersistentClient(config["database"],
                                                    settings=chromadb.Settings(allow_reset=True))
            self.collection = self.client.get_or_create_collection(name="default")

            self.model = SentenceTransformer(config["model"])

            self.tokenizer = AutoTokenizer.from_pretrained(config["model"])

            self.result_num = config["result_num"]

            self.paths = config["paths"]

            self.endswith = config["endswith"]
            print("初始化完成")


    def start(self) -> None:
        print(r"==============")
        print(r"MADE BY EUUEN")
        print(r"VERSION 1.0")
        print(r"输入\help获得帮助")
        print(r"输入(q)退出")
        print(r"==============")
        while True:
            inputs = input(f"search in <{self.collection_name}> ~> ")

            inputs = inputs.strip()

            if inputs == "q":
                break

            elif inputs == r"\reload":
                with open("engine.config.json") as configFile:
                    print("重新加载部分配置中......", end="  ")
                    config = json.load(configFile)

                    self.result_num = config["result_num"]

                    self.paths = config["paths"]

                    self.endswith = config["endswith"]

                    print("重新加载完成")


            elif inputs == r"\loaddoc":
                for path in self.paths:
                    self.loaddoc(path)

            elif inputs.startswith("\cc"):
                inputs_splits = inputs.split(" ")
                for token in inputs_splits:
                    if token == "\cc":
                        continue

                    if token.isspace():
                        continue

                    self.collection_name = token
                    self.collection = self.client.get_or_create_collection(token)

            elif inputs == r"\reset":
                self.client.reset()
                print("请重启重新加载数据库")
                break

            elif inputs == r"\lc":
                print(self.client.list_collections())

            elif inputs == r"\clear":
                self.client.delete_collection(name=self.collection_name)
                print("已经删除此集合")
                self.collection_name = "default"
                self.collection = self.client.get_or_create_collection("default")

            elif inputs == r"\help":
                print(r"\loaddoc 加载文档到当前集合中")
                print(r"\clear 删除当前集合")
                print(r"\lc 列出所有集合")
                print(r"\cc 改变所选当前集合 示例：\cc another")
                print(r"\reload 重新加载部分config")
                print(r"\reset 重置数据库（不知道怎么回事没用，请用\clear指令）")

            else:
                self.query(inputs)


    def loaddoc(self, dir_path: str):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)

            if os.path.isfile(file_path):
                for ends in self.endswith:
                    if file_path.endswith(ends):
                        print(f"处理{file_path}中", end="  ")
                        self.add(file_path)
                        print("处理完毕")
            else:
                self.loaddoc(file_path)


    def add(self, file_path: str):
        with open(file_path) as file:
            content = file.read().strip()
            tokens = self.tokenizer.tokenize(content, verbose=False)

            i = 0
            j = 512 if len(tokens) >= 512 else len(tokens)
            texts = []

            while True:
                texts.append(''.join(tokens[i:j]))

                i += 256
                j += 256

                if j > len(tokens):
                    j = len(tokens)

                if i > len(tokens):
                    break

            embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_tensor=False)

            self.collection.add(
                ids = [file_path],
                embeddings = embeddings.mean(axis=0).reshape(1, 512),
                documents = [content]
            )

    def query(self, query_text: str):
        query_embeddings = self.model.encode(query_text,
                                             normalize_embeddings=True,
                                             convert_to_tensor=False)

        results = self.collection.query(
            query_embeddings=query_embeddings,
            query_texts=[query_text],
            n_results=self.result_num
        )

        for i in range(len(results["documents"][0])):
            end = "\n"
            if results["documents"][0][i].endswith("\n"):
                end = ""
            print(f"=========({i+1}～{results['ids'][0][i]})=========:\n", results["documents"][0][i], end=end)
            print(f"===============================\n")

        print()

if __name__ == "__main__":
    search_engine = SearchEngine()
    search_engine.start()























