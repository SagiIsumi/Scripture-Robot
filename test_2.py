from core_LLM import Chatmodel
from GPTpackages.ImageBufferMemory import encode_image
from MOBIpackages import ControlInterface
from vir_db import VectorDB
import speech
import multiprocessing as mp
import threading
import time
from pathlib import Path
from trilingual_module import female_speak, minnan_speak2
import queue
import os

#以上為 input module
class thread_parameter_get():
    def __init__(self):
        self.thread=None
    def inner_getter(self,query,func,path,language):
        query.put(func(path,language))
    def independent_thread(self,*args):
        self.thread=threading.Thread(target=self.inner_getter,args=args,daemon=True)
        self.thread.start()
        
def language_judger(path):
    global language
    result=[]
    thread_2=thread_parameter_get()
    thread_2.independent_thread(chinese_trigger,MyAudio.speech_to_text,path,'ch')
    thread_3=thread_parameter_get()
    thread_3.independent_thread(english_trigger,MyAudio.speech_to_text,path,'en')
    thread_4=thread_parameter_get()
    thread_4.independent_thread(minnan_trigger,MyAudio.speech_to_text,path,'minnan')
    thread_2.thread.join()
    thread_3.thread.join()
    thread_4.thread.join()
    result.append(chinese_trigger.get())
    result.append(english_trigger.get())
    result.append(minnan_trigger.get())
    print(result[0])
    print(result[1])
    print(result[2])
    if result[0]=='我要說中文' or result[0]=='我要說中文。':
        language='ch'
        female_speak("好的，我知道了",1,'fast','normal')
        return None
    elif result[1]=='I am going to speak English' or result[1]=="I'm going to speak English" or\
            result[1]=='I am going to speak English.' or result[1]=="I'm going to speak English.":
        language='en'
        female_speak("OKay, I am ready, please go ahead",1,'fast','normal')
        return None
    elif result[2]=='我要說台語' or result[2]=='我要說臺語':
        language='minnan'
        minnan_speak2("好的，我知道了")
        return None
    if language=='minnan':
        return result[1]
    else: return result[0]

def myinterruptspeak(language,interface):
    interface.state='speak'
    if language=='ch':
        female_speak("抱歉，你想說什麼?",1,'fast','normal')
    if language=='en':
        female_speak("sorry, what do you say?",1,'fast','normal')
    if language=='minnan':
        minnan_speak2("抱歉，你想說什麼?")
    interface.state='idol'


if __name__=='__main__':
    conversation_history=VectorDB(persist_directory='.\conversation_db')
    script_data=VectorDB(persist_directory='.\script_db')
    script_data.load_text(path='.\scripts')
    #建立本地程式庫，建好後會有conversation_db和script_db兩個資料夾
    intention_answer=Chatmodel(promptpath='.\prompts\intention_prompt.txt')
    Main_model=Chatmodel(promptpath='.\prompts\chat_prompt.txt',
                            longmemory_db=conversation_history,local_db=script_data,temperature=0.5)
    MyAudio=speech.audio_procession()
    interface=ControlInterface.ControlInterface(enable_camera=True, show_img=True, enable_arm=False, enable_face=True, is_FullScreen=False)
    #建立對話模型，上為偵測意圖，下為對話用
    language='ch'
    chinese_trigger=queue.Queue()
    english_trigger=queue.Queue()
    minnan_trigger=queue.Queue()
    text_dict={'what':''}
    interrupt=False
    action="nothing"

    while True:
        myinput=input("Human: ")#收音
        interface.get_frame()
        fileList = os.listdir('input_img')
        if fileList != []:
            img_list = [encode_image('input_img/' + fileList[-1])]
        text_dict['what']=myinput
        if text_dict['what']==None:
            continue
        if text_dict['what']=='對話結束':#結束對話
            break
        text_dict['language']=language
        intention=intention_answer.run_intention(texts=text_dict)
        print(intention)
        result=Main_model.run(text_dict,intention=intention,img_list=img_list)#run GPT model
        conversation_history.save_text(result)#存本次對話
        conversation_history.load_text()#讀ltm
        #print(result)
        #interrupt=MyAudio.speaking(result[1],language=language)#機器人說話
        print(result[1])