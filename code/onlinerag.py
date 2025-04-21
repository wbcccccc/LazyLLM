import lazyllm
from lazyllm import bind
import sys
sys.path.append("D:/Code/LazyLLM/LazyLLM")
# 配置API_KEY 和 SECRET_KEY
API_KEY = ""
SECRET_KEY = ""

# 文档路径
dataset_path = 'D:/Source/ragdata/ds'


# 文档加载
documents = lazyllm.Document(dataset_path=dataset_path)
prompt = 'You will act as an AI question-answering assistant and complete a dialogue task. \
      In this task, you need to provide your answers based on the given context and questions.'

# RAG 数据流
with lazyllm.pipeline() as ppl:
    ppl.retriever = lazyllm.Retriever(doc=documents,
                group_name="CoarseChunk", similarity="bm25_chinese", topk=3)
    ppl.formatter = (
        lambda nodes,query: dict(
            context_str = "".join([node.get_content()
                                   for node in nodes]),
            query = query
        )) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(source="sensenova", model="DeepSeek-R1", api_key=API_KEY, secret_key=SECRET_KEY, temperature=0.3).prompt(
        lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str'])
    )

lazyllm.WebModule(ppl, port=23464).start().wait()