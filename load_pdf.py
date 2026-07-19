from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os

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
            print(f"使用本地模型路径: {local_path}")
            self.model = SentenceTransformer(
                local_path,
                device=self.device
            )
        else:
            print(f"使用模型名称: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder,
                device=self.device
            )
        
        print("✅ 模型加载成功")
    
    def embed_documents(self, texts):
        if self.model is None:
            print("正在加载SentenceTransformer模型...")
            self._load_model()
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_query(self, text):
        if self.model is None:
            print("正在加载SentenceTransformer模型...")
            self._load_model()
        
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()

def load_and_split_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"错误：未找到文件 {pdf_path}")
        return None
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"成功加载 {len(documents)} 页文档")
    print("=" * 60)
    
    print("\n[加载后的文档元数据]")
    print("-" * 60)
    for i, doc in enumerate(documents):
        page_num = doc.metadata.get('page', '无')
        print(f"文档 {i}: page={page_num}, 内容长度={len(doc.page_content)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    splits = text_splitter.split_documents(documents)
    
    print(f"\n文档切分完成，共生成 {len(splits)} 个文本块")
    print("=" * 60)
    
    print("\n[所有文本块的页码分布]")
    print("-" * 60)
    page_counts = {}
    for i, split in enumerate(splits):
        page = split.metadata.get('page', '未知')
        page_counts[page] = page_counts.get(page, 0) + 1
        print(f"文本块 {i}: 页码={page}")
    
    print("\n[页码统计]")
    print("-" * 60)
    for page in sorted(page_counts.keys()):
        print(f"页码 {page}: {page_counts[page]} 个文本块")
    
    if len(splits) >= 1:
        print("\n[第一块内容]")
        print("-" * 60)
        print(f"来源: {splits[0].metadata.get('source', '未知')}")
        print(f"页码: {splits[0].metadata.get('page', '未知')}")
        print(f"内容:\n{splits[0].page_content}")
        print("-" * 60)
    
    if len(splits) >= 5:
        print("\n[第五块内容]")
        print("-" * 60)
        print(f"来源: {splits[4].metadata.get('source', '未知')}")
        print(f"页码: {splits[4].metadata.get('page', '未知')}")
        print(f"内容:\n{splits[4].page_content}")
        print("-" * 60)
    
    return splits

def create_vectorstore(splits, vectorstore_path):
    try:
        print("\n正在初始化向量化模型...")
        
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_ENDPOINT"] = ""
        
        cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        
        embeddings = OfflineSentenceTransformerEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=cache_folder,
            device="cpu"
        )
        
        print("正在创建FAISS向量库...")
        db = FAISS.from_documents(splits, embeddings)
        
        print(f"向量库创建完成，共存储 {db.index.ntotal} 条记录")
        print("正在保存向量库到本地...")
        db.save_local(vectorstore_path)
        print(f"向量库已保存到: {vectorstore_path}")
        
        return db, embeddings
    except Exception as e:
        print(f"\n向量化过程出错: {e}")
        return None, None

def create_rag_chain(db, api_key):
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
    
    return rag_chain, retriever

def main():
    pdf_path = os.path.join(os.path.dirname(__file__), "test.pdf")
    vectorstore_path = os.path.join(os.path.dirname(__file__), "vectorstore")
    
    load_dotenv()
    api_key = os.getenv("SILICONFLOW_API_KEY")
    
    if not api_key:
        print("错误：未设置 SILICONFLOW_API_KEY 环境变量")
        print("请在 .env 文件中添加：SILICONFLOW_API_KEY=你的API密钥")
        return
    
    splits = load_and_split_pdf(pdf_path)
    if splits is None:
        return
    
    db, embeddings = create_vectorstore(splits, vectorstore_path)
    if db is None:
        print("无法创建向量库，尝试从本地加载...")
        try:
            db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            print(f"成功从本地加载向量库，共 {db.index.ntotal} 条记录")
        except Exception as e:
            print(f"加载失败: {e}")
            return
    
    rag_chain, retriever = create_rag_chain(db, api_key)
    
    print("\n" + "=" * 60)
    print("RAG问答系统已就绪！")
    print("输入问题进行提问，输入 'exit' 退出")
    print("=" * 60)
    
    while True:
        query = input("\n请输入问题：")
        if query.lower() == "exit":
            print("退出程序")
            break
        
        print("\n正在检索相关片段...")
        retrieved_docs = retriever.invoke(query)
        
        print("\n[检索到的参考片段]")
        print("-" * 60)
        for i, doc in enumerate(retrieved_docs, 1):
            page = doc.metadata.get('page', '未知')
            print(f"\n[{i}] 页码: {page}")
            print(f"内容:\n{doc.page_content}")
        
        print("\n[生成回答]")
        print("-" * 60)
        try:
            answer = rag_chain.invoke(query)
            print(answer)
        except Exception as e:
            print(f"生成回答失败: {e}")

if __name__ == "__main__":
    main()