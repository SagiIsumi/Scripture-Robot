from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
import torch
import configparser

config=configparser.ConfigParser()
config.read('config.ini')
API_KEY=config.get('openai','key1')
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

class VectorDB():
    def __init__(self,chunk_size=512,chunk_overlap=64,persist_directory=None)->None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模型將運行在: {device}")
        self.splitter=RecursiveCharacterTextSplitter(separators=[" ", "\n"],chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        self.embeddings=OpenAIEmbeddings(model='text-embedding-3-large')
        self.record=[]
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
    def load_text(self,path=".\conversation_history")->None:     #建立vector database，進行後續RAG撈資料
        texts=[]
        raw_documents=None
        for content in Path(path).glob("*.txt"):
            raw_documents = TextLoader(str(content), encoding='utf-8').load_and_split(self.splitter)
            if path==".\scripts":#讀取經文資料時對metadata進行處理並儲存
                for name in raw_documents:
                    # print(name)
                    title=name.metadata["source"].split("_")[0]
                    name.metadata["source"]= title.split("\\")[1]
                    texts.append(name)

            else:
                texts=texts+raw_documents
        filter_texts=[item for item in texts if item not in self.record]
        self.record=texts
        if raw_documents!=None:
            self.vectorstore.add_documents(documents=filter_texts) 
        print('文檔加載完成')
            
    def save_text(self,response:list)->None:     #將對話儲存為txt檔
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%Y-%m-%d")
        currentMoment= currentDateAndTime.strftime("%Y-%m-%d,%H:%M:%S")
        title='./conversation_history/response_'+currentTime+".txt"
        path=Path(title)
        with open(path,mode="a",encoding="utf-8") as f:
            f.write(f'(Time: {currentMoment}), Human: '+response[0])
            f.write("\n")
            f.write(f'(Time: {currentMoment}), 莫比: '+response[1]+"\n")
    def retrive_text(self,text,keyword=None)->str:       #RAG，取出相關對話、資料
        retriver=self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        results=retriver.invoke(text)
        output=''
        #print(results)
        for i,content in enumerate(results):
            output=output+f"Document {i+1}: Text:{content.page_content}\n\
                            Metadata: {content.metadata}\n"
        return output
    def get_retriver(self,keyword=None): #返回vector database的retriver，功能跟上面重複了，但我懶得優化
        if keyword!=None:
            return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3,"filter":{"source":keyword}})
        else:
            return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})


    