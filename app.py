import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

class OfflineSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name, cache_folder=None, device="cpu"):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        self.model = None
    
    def _find_local_model_path(self):
        if self.cache_folder is None:
            self.cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        
        possible_names = [self.model_name]
        if not self.model_name.startswith("sentence-transformers/"):
            possible_names.append(f"sentence-transformers/{self.model_name}")
        
        for name in possible_names:
            model_cache_path = os.path.join(
                self.cache_folder,
                f"models--{name.replace('/', '--')}"
            )
            
            if os.path.exists(model_cache_path):
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        return os.path.join(snapshots_dir, snapshots[0])
        
        return None
    
    def _load_model(self):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_ENDPOINT"] = ""
        
        local_path = self._find_local_model_path()
        
        if local_path is not None and os.path.exists(local_path):
            st.write(f"使用本地模型路径: {local_path}")
            self.model = SentenceTransformer(
                local_path,
                device=self.device
            )
        else:
            st.write(f"使用模型名称: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder,
                device=self.device
            )
        
        st.write("✅ 模型加载成功")
    
    def embed_documents(self, texts):
        if self.model is None:
            st.write("正在加载SentenceTransformer模型...")
            self._load_model()
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_query(self, text):
        if self.model is None:
            st.write("正在加载SentenceTransformer模型...")
            self._load_model()
        
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()

PDF_PATH = os.path.join(os.path.dirname(__file__), "test.pdf")
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "vectorstore")
DOCUMENT_NAME = "数据结构_南京审计大学.pdf"

def init_vectorstore():
    if "db" not in st.session_state or "retriever" not in st.session_state or "rag_chain" not in st.session_state:
        with st.spinner("正在初始化向量库..."):
            api_key = os.getenv("SILICONFLOW_API_KEY")
            if not api_key:
                st.error("未设置 SILICONFLOW_API_KEY 环境变量，请在 .env 文件中配置")
                return False
            
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_ENDPOINT"] = ""
            
            cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            
            embeddings = OfflineSentenceTransformerEmbeddings(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                cache_folder=cache_folder,
                device="cpu"
            )
            
            if os.path.exists(VECTORSTORE_PATH):
                try:
                    db = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
                    st.success(f"成功加载向量库，共 {db.index.ntotal} 条记录")
                except Exception as e:
                    st.error(f"加载向量库失败，重新创建: {e}")
                    db = None
            else:
                db = None
            
            if db is None:
                if not os.path.exists(PDF_PATH):
                    st.error(f"未找到PDF文件: {PDF_PATH}")
                    return False
                
                loader = PyPDFLoader(PDF_PATH)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512,
                    chunk_overlap=50,
                    length_function=len,
                    is_separator_regex=False,
                )
                splits = text_splitter.split_documents(documents)
                
                db = FAISS.from_documents(splits, embeddings)
                db.save_local(VECTORSTORE_PATH)
                st.success(f"向量库创建完成，共 {db.index.ntotal} 条记录")
            
            retriever = db.as_retriever(search_kwargs={"k": 4})
            
            template = """基于以下参考资料回答问题。如果不知道就回答不知道。
参考资料：{context}
问题：{question}"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            llm = ChatOpenAI(
                model="Qwen/Qwen2.5-7B-Instruct",
                api_key=api_key,
                base_url="https://api.siliconflow.cn/v1",
                temperature=0.1
            )
            
            def format_docs(docs):
                return "\n\n".join([f"[页码{doc.metadata.get('page', '未知')}]{doc.page_content}" for doc in docs])
            
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser
            
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            st.session_state.db = db
            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
        
        return True
    return True

def reload_vectorstore():
    if "db" in st.session_state:
        del st.session_state.db
    if "retriever" in st.session_state:
        del st.session_state.retriever
    if "rag_chain" in st.session_state:
        del st.session_state.rag_chain
    init_vectorstore()

def main():
    st.set_page_config(page_title="📚 数据结构课程智能问答助手", page_icon="📚", layout="wide")
    
    st.title("📚 数据结构课程智能问答助手")
    
    with st.sidebar:
        st.header("📄 当前文档")
        st.write(DOCUMENT_NAME)
        
        if st.button("🔄 重新加载向量库"):
            reload_vectorstore()
        
        st.divider()
        st.markdown("---")
    
    if not init_vectorstore():
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("🔍 参考来源"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] 页码: {source['page']}**")
                        st.markdown(source["content"])
    
    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("正在检索并生成回答..."):
                retrieved_docs = st.session_state.retriever.invoke(prompt)
                
                sources = []
                for doc in retrieved_docs:
                    sources.append({
                        "page": doc.metadata.get("page", "未知"),
                        "content": doc.page_content
                    })
                
                answer = st.session_state.rag_chain.invoke(prompt)
                
                st.markdown(answer)
                
                with st.expander("🔍 参考来源"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**[{i}] 页码: {source['page']}**")
                        st.markdown(source["content"])
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
    
    st.divider()
    st.markdown("⚡ 使用技术：LangChain + FAISS + Qwen2.5")

if __name__ == "__main__":
    main()
