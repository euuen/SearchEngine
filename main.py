import json
import os

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BertTokenizerFast
from chromadb.errors import NotFoundError

from hashlib import md5

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
    outputForm: str
    includes: list
    endswith: list
    isRunning: bool = False
    lastResult: list = []

    def __init__(self):
        with open("engine.config.json") as configFile:
            print("初始化中搜索引擎......", end="  ")
            config = json.load(configFile)

            self.client = chromadb.PersistentClient(config["database"],
                                                    settings=chromadb.Settings(allow_reset=True))
            self.collection = self.client.get_or_create_collection(name="default")

            self.model = SentenceTransformer(config["model"])

            self.tokenizer = AutoTokenizer.from_pretrained(config["model"])

            self.outputForm = config["outputForm"]

            self.result_num = config["result_num"]

            self.includes = config["includes"]

            self.endswith = config["endswith"]
            print("初始化完成")


    def process(self, command):
        if command == "q":
            self.isRunning = False

        elif command == r"\reload":
            with open("engine.config.json") as configFile:
                print("重新加载部分配置中......", end="  ")
                config = json.load(configFile)

                self.result_num = config["result_num"]

                self.includes = config["includes"]

                self.endswith = config["endswith"]

                print("重新加载完成")


        elif command == r"\loaddoc":
            for path in self.includes:
                self.load_doc(path)

        elif command.startswith(r"\cc"):
            inputs_splits = command.split(" ")
            for token in inputs_splits:
                if token == r"\cc":
                    continue

                if token.isspace():
                    continue

                try:
                    collection = self.client.get_collection(token)
                    self.collection = collection
                    self.collection_name = token
                except NotFoundError:
                    print("This collection don't exist")

        elif command.startswith(r"\mc"):
            inputs_splits = command.split(" ")
            for token in inputs_splits:
                if token == r"\mc":
                    continue

                if token.isspace():
                    continue

                self.collection = self.client.get_or_create_collection(token)
                self.collection_name = token

        elif command == r"\reset":
            self.client.reset()
            print("请重启重新加载数据库")
            self.isRunning = False

        elif command == r"\lc":
            list_collections = self.client.list_collections()
            for i in range(len(list_collections)):
                print(f"[ {i + 1} ] -- ", list_collections[i].name)

        elif command == r"\clear":
            self.client.delete_collection(name=self.collection_name)
            print("已经删除此集合")
            self.collection_name = "default"
            self.collection = self.client.get_or_create_collection("default")

        elif command == r"\clearall":
            for collection in self.client.list_collections():
                self.client.delete_collection(collection.name)

            print("已经删除全部集合")
            self.collection_name = "default"
            self.collection = self.client.get_or_create_collection("default")

        elif command == r"\help":
            print(r"\loaddoc 加载文档到当前集合中")
            print(r"\clear 删除当前集合")
            print(r"\lc 列出所有集合")
            print(r"\clearall 删除全部集合")
            print(r"\cc 改变所选当前集合 示例：\cc another")
            print(r"\reload 重新加载部分config")
            print(r"\reset 重置数据库（不知道怎么回事没用，请用\clear指令）")

        elif command.startswith(r"\look"):
            inputs_splits = command.split(" ")
            for token in inputs_splits:
                if token == r"\look":
                    continue

                if token.isspace():
                    continue

                self.look(token)

        elif command.startswith("\\"):
            print("Unknown Command")

        else:
            self.query(command)


    def start(self) -> None:
        print(r"==============")
        print(r"MADE BY EUUEN")
        print(r"VERSION 1.0")
        print(r"输入\help获得帮助")
        print(r"输入(q)退出")
        print(r"==============")
        self.isRunning = True
        while self.isRunning:
            inputs = input(f"search in <{self.collection_name}> ~> ")

            inputs = inputs.strip()

            self.process(inputs)

            if not self.isRunning:
                break


    def load_doc(self, dir_path: str):
        default_collection = self.client.get_or_create_collection(name="default")
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)


            if os.path.isfile(file_path):
                self.collection = default_collection
                for end in self.endswith:
                    if file_path.endswith(end):
                        self.add(file_path)
            else:
                if filename.startswith(r"."):
                    continue
                self.collection = self.client.get_or_create_collection(filename.lower())
                self.load_collection(file_path)

        # 调回本来的collection
        self.collection = self.client.get_or_create_collection(self.collection_name)


    def load_collection(self, path: str):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)

            if os.path.isfile(file_path):
                for end in self.endswith:
                    if file_path.endswith(end):
                        self.add(file_path)
            else:
                self.load_collection(file_path)


    def add(self, file_path: str):
        with open(file_path) as file:
            content = file.read().strip()
            obj = md5()
            obj.update(content.encode("utf-8"))
            hash_value = obj.hexdigest()

            result = self.collection.get(ids=[file_path])
            if file_path in result["ids"] and hash_value == result["metadatas"][0]["hash"]:
                return

            print(f"处理{file_path}中", end="  ")

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

            if not file_path in result["ids"]:
                self.collection.add(
                    ids = [file_path],
                    metadatas = {"hash": hash_value},
                    embeddings = embeddings.mean(axis=0).reshape(1, 512),
                    documents = [content]
                )
            else:
                self.collection.update(
                    ids = [file_path],
                    metadatas = {"hash": hash_value},
                    embeddings = embeddings.mean(axis=0).reshape(1, 512),
                    documents = [content]
                )
            print("处理完成")

    def query(self, query_text: str):
        query_embeddings = self.model.encode(query_text,
                                             normalize_embeddings=True,
                                             convert_to_tensor=False)

        results = self.collection.query(
            query_embeddings=query_embeddings,
            query_texts=[query_text],
            n_results=self.result_num
        )

        if self.outputForm == "simple":
            self.lastResult.clear()
            for i in range(len(results["documents"][0])):
                print(f"[ {i + 1} ]～{results['ids'][0][i]}")
                self.lastResult.append(results['ids'][0][i])
            print()

        elif self.outputForm == "full":
            self.lastResult.clear()
            for i in range(len(results["documents"][0])):
                end = "\n"
                if results["documents"][0][i].endswith("\n"):
                    end = ""
                print(f"=========({i + 1}～{results['ids'][0][i]})=========:\n", results["documents"][0][i], end=end)
                print(f"===============================\n")
                self.lastResult.append(results['ids'][0][i])

            print()

        else:
            print("Unknow output form.Please set a valid value")

    def look(self, id: str):
        if id.isnumeric():
            id = int(id)
            if id > len(self.lastResult):
                print("This number is too big")
            else:
                id = self.lastResult[id-1]
                print(self.collection.get(ids=[id])["documents"][0])

        else:
            result = self.collection.get(ids=[id])
            if len(result["documents"]) == 0:
                print("This doc don't exist!Please check your id")
            else:
                print(result["documents"][0])

if __name__ == "__main__":
    search_engine = SearchEngine()
    search_engine.start()

























