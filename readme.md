## 一、安装Ollama
[https://ollama.com/download/windows](https://ollama.com/download/windows)
## 二、拉取模型
命令行执行
~~~shell
  # 根据硬件自己选择主语言模型 本地环境建议选择7b以下的模型
  ollama pull llama3.1:latest # 主语言模型(推荐)
  ollama pull deepseek-r1:latest # 主语言模型
  
  ollama pull nomic-embed-text:latest # 矢量嵌入模型
~~~
## 三、安装依赖
创建虚拟环境
~~~shell
  pip install pipenv
  pipenv install
  pipenv shell
~~~
进入根目录，执行命令
~~~shell
  pip install -r requirements.txt
~~~
等待依赖安装结束，遇见其他的错误自行百度～
## 四、运行项目
1. 运行[doc_loader.py](bmos/document_loader/doc_loader.py)的main函数，将测试md文档导入到矢量库（矢量库位置：[chroma_data](bmos/document_loader/chroma_data)）
2. 运行[chat_rag.py](bmos/chat_rag.py)的main函数，启动知识库对话工具
3. 浏览器访问[http://127.0.0.1:7861](http://127.0.0.1:7861)
4. 基于graph构建的agent智能体应用[langgraph_test.py](bmos/langgraph_test.py)（学习中）