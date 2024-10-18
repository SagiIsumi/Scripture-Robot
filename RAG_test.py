import torch
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# 定義自定義嵌入模型類
class HuggingfaceEmbeddingModel(Embeddings):
    def __init__(self, model_name: str, device):
        # 初始化 Huggingface 模型和 Tokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)  # 將模型放到指定設備
    
    def embed_query(self, text: str) -> list[float]:
        # 將文本轉為嵌入
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)  # 將輸入移動到相同設備
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 取最後一層的平均值作為句子嵌入
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 對多個文檔生成嵌入
        return [self.embed_query(text) for text in texts]

print(torch.version.cuda)
if torch.cuda.is_available():
    print("有可用的 GPU！")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"模型將運行在: {device}")

# 初始化 Huggingface 的自定義嵌入模型
embedding_model = HuggingfaceEmbeddingModel(model_name="intfloat/multilt", device=device)
print("加載完成")

# 初始化向量資料庫
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
print("vector db建置完成")

record = []

def load_text(path=".\scripts") -> None:  # 建立vector database，進行後續RAG撈資料
    global record
    texts = []
    raw_documents = None
    for content in Path(path).glob("*.txt"):
        splitter = RecursiveCharacterTextSplitter(separators=[" ", "\n"], chunk_size=256, chunk_overlap=48)
        raw_documents = TextLoader(str(content), encoding='utf-8').load_and_split(splitter)
        if path == ".\scripts":  # 讀取經文資料時對metadata進行處理並儲存
            for name in raw_documents:
                title = name.metadata["source"].split("_")[0]
                name.metadata["source"] = title.split("\\")[1]
                texts.append(name)
        else:
            texts = texts + raw_documents

    filter_texts = [item for item in texts if item not in record]
    record = texts
    if raw_documents != None:
        vector_db.add_documents(documents=filter_texts)

load_text()
print('完成建檔')
# 構建檢索器
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# 查詢問題
query = "你好跟我說說無受想息事是什麼意思"

# 檢索相關文檔
documents = retriever.invoke(query)

# 打印檢索結果
for doc in documents:
    print(f"檢索到的文檔: {doc.page_content} source: {doc.metadata}")