from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datetime import datetime
import os
API_KEY=os.getenv('OPENAI_API_KEY')

class VectorDB():
    def __init__(self,chunk_size=200,chunk_overlap=20,persist_directory=None)->None:
        self.splitter=RecursiveCharacterTextSplitter(separators=[" ", "\n"],chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        self.embeddings=OpenAIEmbeddings(openai_api_key=API_KEY)
        self.record=[]
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
    def load_text(self,path=".\conversation_history")->None:     #建立vector database，進行後續RAG撈資料
        texts=[]
        raw_documents=None
        for i,content in enumerate(Path(path).glob("*.txt")):
            if content in self.record:#若已經讀過該檔，就不要再讀一次
                continue
            self.record.append(content)
            raw_documents = TextLoader(str(content), encoding='utf-8').load_and_split(self.splitter)
            if path==".\scripts":#讀取經文資料時對metadata進行處理並儲存
                for name in raw_documents:
                    title=name.metadata["source"].split("_")[0]
                    name.metadata["source"]= title.split("\\")[1]
                    texts.append(name)

            else:
                texts=texts+raw_documents
        if raw_documents!=None:
            self.vectorstore.add_documents(documents=texts) 
            
    def save_text(self,response:list)->None:     #將對話儲存為txt檔
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%H-%M-%S")
        title='./conversation_history/response_'+currentTime+".txt"
        path=Path(title)
        with open(path,mode="w",encoding="utf-8") as f:
            f.write(response[0])
            f.write("\n")
            f.write(response[1])
    def retrive_text(self,text,keyword=None)->str:       #RAG，取出相關對話、資料
        retriver=self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        results=retriver.invoke(text)
        output=''
        #print(results)
        for i in results:
            output=output+"\n"+f"{i.page_content}"
        return output
    def get_retriver(self,keyword=None): #返回vector database的retriver，功能跟上面重複了，但我懶得優化
        if keyword!=None:
            return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3,"filter":{"source":keyword}})
        else:
            return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})


    