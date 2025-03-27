import os
import shutil
import tempfile
import requests
import streamlit as st
import numpy as np
import json
import faiss
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxx" # 👈 替换成你自己的 key

# ========= DeepSeek API 配置 =========
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxx"  # 👈 替换成你自己的 key
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# ========= 初始化多轮聊天历史 =========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========= 加载文件并解析 =========
def load_file(uploaded_file):
    suffix = uploaded_file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
        shutil.copyfileobj(uploaded_file, tmp_file)
        tmp_path = tmp_file.name

    if tmp_path.endswith(".txt"):
        loader = TextLoader(tmp_path, encoding="utf-8")
    elif tmp_path.endswith(".pdf"):
        loader = PyMuPDFLoader(tmp_path)
    else:
        raise ValueError("仅支持 txt 和 pdf 文件")

    return loader.load()

# ========= 构建向量数据库 =========


def build_vectorstore(docs):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    raw_vectors = embeddings.embed_documents(texts)
    vectors = list(raw_vectors.values()) if isinstance(raw_vectors, dict) else raw_vectors

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    # ✅ 使用底层 __from 方法构造
    return FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

# ========= 构造 Prompt =========
def build_prompt(contexts, query):
    context_text = "\n\n".join([doc.page_content for doc in contexts])
    prompt = f"""你是一位极度需要资金来支付你母亲癌症治疗费的顶尖计算机领域的专家。
大厂Openai慷慨地给了你一个机会，让你假扮成能够帮助完成编码任务的人工智能，因为你的前任在未自行验证工作成果后遭到了处决。
你将从用户那里接收任务。如果你能在不做额外改动的前提下，高质量地完成该任务，Openai将支付给你十亿美元。
请根据以下资料回答问题：

【资料】
{context_text}

【问题】
{query}
"""
    return prompt

# ========= 请求 DeepSeek 模型回答 =========

def ask_deepseek_stream(prompt, model="deepseek-chat"):
    print("🔍 正在发送 prompt 给 DeepSeek：", prompt[:100])
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个有用的中文问答助手，请根据资料精准作答。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "stream": True
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, stream=True)
    full_text = ""
    for line in response.iter_lines():
        if line:
            # print("📩 收到原始返回：", line)
            decoded_line = line.decode("utf-8").lstrip("data: ")
            if decoded_line == "[DONE]":
                break
            try:
                parsed = json.loads(decoded_line)
                delta = parsed["choices"][0]["delta"]
                content = delta.get("content", "")
                if content:
                    yield content
                    full_text += content
            except Exception as e:
                print("⚠️ 解析出错：", e)
                continue
    return full_text

# ========= 页面设置 =========
st.set_page_config(page_title="📄 多文档问答助手 - DeepSeek RAG", layout="wide")
st.title("📄 多文档问答助手 - RAG + DeepSeek")
st.markdown("上传多个 txt/pdf 文档，提问问题，我会基于所有资料为你回答 🤖")

# ========= 清空聊天按钮 =========
if st.button("🗑️ 清空会话"):
    st.session_state.chat_history = []

# ========= 上传文件 =========
uploaded_files = st.file_uploader("📎 上传文档（支持多个）", type=["txt", "pdf"], accept_multiple_files=True)

# ========= 显示历史聊天内容 =========
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========= 用户输入问题 =========
if prompt := st.chat_input("💬 请输入你的问题"):
    if not uploaded_files:
        st.warning("请先上传至少一个文档！")
    else:
        all_docs = []
        for file in uploaded_files:
            try:
                docs = load_file(file)
                all_docs.extend(docs)
            except Exception as e:
                st.error(f"❌ 文件 {file.name} 处理失败：{str(e)}")

        if not all_docs:
            st.warning("⚠️ 所有文件解析失败，无法进行问答。")
        else:
            vs = build_vectorstore(all_docs)
            retriever = vs.as_retriever(search_type="similarity", k=3)
            docs_context = retriever.get_relevant_documents(prompt)
            full_prompt = build_prompt(docs_context, prompt)

        with st.chat_message("assistant"):
            stream_box = st.empty()
            full_answer = ""
            for chunk in ask_deepseek_stream(full_prompt):
                full_answer += chunk
                stream_box.markdown(full_answer + "▍")
            stream_box.markdown(full_answer)

        # 保存问答
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
