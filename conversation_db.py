from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datetime import datetime
class Conversation_DB():
    def __init__(self,chunk_size=200,chunk_overlap=0,persist_directory="./conversation_history_db")->None:
        self.splitter=RecursiveCharacterTextSplitter(separators=[" ", "\n"],chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        self.text=[]
        self.embeddings=OpenAIEmbeddings(openai_api_key='sk-proj-ebjcEZYloUZnX0nLGaXdT3BlbkFJAeGVT80bSncVYWOqmaFZ')
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    def load_text(self,path="./conversation_history"):
        self.path=Path(path)
        exist=[]
        for content in self.path.glob('*.txt'):
            if content in exist: continue
            else: exist.append(content)
            raw_documents = TextLoader(str(content), encoding='utf-8').load_and_split(self.splitter)
            self.text=self.text+raw_documents
        if self.text!=[]:
            self.vectorstore.add_documents(documents=self.text)
    def save_text(self,response:str):
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%H-%M-%S")
        title='./conversation_history/response_'+currentTime+".txt"
        path=Path(title)
        with open(path,mode="w",encoding="utf-8") as f:
            f.write(response)
    def retrive_text(self,text):
        retriver=self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        results=retriver.invoke(text)
        output=''
        for i in results:
            output=output+"\n"+f"{i.page_content}"
        return output
