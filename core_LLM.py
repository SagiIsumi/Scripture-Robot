import os 
from GPTpackages.GPTopenai import GPTopenai
from GPTpackages.PromptTemplate import PromptTemplate
from GPTpackages.ImageBufferMemory import ImageBufferMemory, encode_image
from GPTpackages.TextBuffer import TextBuffer
from pathlib import Path
from vir_db import VectorDB
import json
from opencc import OpenCC

ser_key=os.getenv('SERPAPI_API_KEY')
openai_key=os.getenv('OPENAI_API_KEY')
MODEL='gpt-4o'
class Chatmodel():
    def __init__(self,promptpath:str,openai_key=openai_key,longmemory_db=None,local_db=None):
        self.key = openai_key
        self.title=['心經原文','法鼓山','耕雲先生','悉曇學會','淨空法師','開山祖師']
        self.stm=TextBuffer(buffer_size=5)
        self.ltm=longmemory_db
        self.local_db=local_db
        self.trans=OpenCC('s2twp')
        self.model=GPTopenai(
            openai_api_key=openai_key,
            prompt=PromptTemplate(Path(promptpath)),
            temperature=0.2,
            model=MODEL,
            img_memory=ImageBufferMemory())
    def text_retrival(self,text,keyword) -> str:
        if keyword in self.title:
            dataretiver=self.local_db.get_retriver(keyword=keyword)
        else:
            dataretiver=self.local_db.get_retriver()
        results=dataretiver.invoke(text)
        output=''
        for i,content in enumerate(results):
            output=output+f"Document {i+1}: Text:{content.page_content}\n\
                            Metadata: {content.metadata}\n"

        #print(output)

        return output
    def run_intention(self,texts:dict)->dict:
        texts['what']=self.stm.get()+texts['what']
        output=self.model.run(text_dict=texts)
        result=json.loads(output)
        #result=[input,output]
        return result
    def run(self,texts:dict,intention:dict)->list: 
        text=self.stm.get()+texts['what']
        texts['intention']=intention['intention']
        texts["local_data"]=self.text_retrival(text,intention['keyword'])
        conversation=self.stm.get()+"\n"+self.ltm.retrive_text(text)
        texts["conversation"]=conversation
        results=self.model.run(text_dict=texts)
        input=self.trans.convert(texts['what'])
        output=self.trans.convert(results)
        results=[input,output]
        self.stm.set(results)
        return results