import os 
from GPTpackages.GPTopenai import GPTopenai
from GPTpackages.PromptTemplate import PromptTemplate
from GPTpackages.ImageBufferMemory import ImageBufferMemory, encode_image
from GPTpackages.TextBuffer import TextBuffer
from pathlib import Path
from vir_db import VectorDB
import json
from opencc import OpenCC
import configparser

config=configparser.ConfigParser()
config.read('config.ini')
openai_key=config.get('openai','key1')

MODEL='gpt-4o-mini'
class Chatmodel():#核心對話模型
    def __init__(self,promptpath:str,openai_key=openai_key,longmemory_db=None,local_db=None,temperature=0.2):
        self.key = openai_key
        self.title=['心經原文','法鼓山','耕雲先生','悉曇學會','淨空法師','開山祖師'] #關鍵字，聽到的話從metadata裡撈資料
        self.stm=TextBuffer(buffer_size=3)#短期記憶，學長設計的
        self.ltm=longmemory_db
        self.local_db=local_db
        self.trans=OpenCC('s2twp')#簡中轉繁中用，備用的
        self.model=GPTopenai(
            openai_api_key=openai_key,
            prompt=PromptTemplate(Path(promptpath)),
            temperature=temperature,
            model=MODEL,
            img_memory=ImageBufferMemory())
    def text_retrival(self,text,keyword) -> str:#取出收音的文字
        if keyword in self.title:#偵測到關鍵字啟用
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
    def run_intention(self,texts:dict)->dict:#intention model用
        texts['what']=self.stm.get()+texts['what']
        output=self.model.run(text_dict=texts)
        result=json.loads(output)
        #result=[input,output]
        return result
    def run(self,texts:dict,intention:dict,img_list=[])->list: #主對話model用
        text=self.stm.get()+texts['what']
        texts['intention']=intention['intention']
        texts['keyword']=intention['keyword']
        texts["local_data"]=self.text_retrival(text,intention['keyword'])
        conversation=self.stm.get()+"\n"+self.ltm.retrive_text(text)
        texts["conversation"]=conversation
        texts["paragraph"]=intention["paragraph"]
        #以上就撈本地資料、stm、ltm各種資料
        results=self.model.run(text_dict=texts,img_list=img_list,img_refresh=True)#丟給GPT
        if texts['language']=='ch':
            input=self.trans.convert(texts['what'])
        else:
            input=texts['what']
        output=self.trans.convert(results)
        results=[input,output]
        self.stm.set(results)
        return results