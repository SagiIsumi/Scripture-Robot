from core_LLM import Chatmodel
from MOBIpackages import ControlInterface
from vir_db import VectorDB
import speech
import multiprocessing as mp
import threading
import time
from pathlib import Path
from trilingual_module import female_speak, minnan_speak2
import queue
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
    thread_2.independent_thread(normal_trigger,MyAudio.speech_to_text,path,'ch')
    thread_3=thread_parameter_get()
    thread_3.independent_thread(minnan_trigger,MyAudio.speech_to_text,path,'minnan')
    thread_2.thread.join()
    thread_3.thread.join()
    result.append(normal_trigger.get())
    result.append(minnan_trigger.get())
    
    if result[0]=='我要說中文':
        language='ch'
        female_speak("好的，我知道了",1,'normal','normal')
        return None
    elif result[0]=='I am going to speak English':
        language='en'
        female_speak("OKay, I am ready, please go ahead",1,'normal','normal')
        return None
    elif result[1]=='我要說台語':
        language='minnan'
        minnan_speak2("好的，我知道了")
        return None
    if language=='minnan':
        return result[1]
    else: return result[0]

if __name__=='__main__':
    conversation_history=VectorDB(persist_directory='.\conversation_db')
    script_data=VectorDB(persist_directory='.\script_db')
    script_data.load_text(path='.\scripts')
    #建立本地程式庫，建好後會有conversation_db和script_db兩個資料夾
    intention_answer=Chatmodel(promptpath='.\prompts\intention_prompt.txt')
    Main_model=Chatmodel(promptpath='.\prompts\chat_prompt.txt',
                            longmemory_db=conversation_history,local_db=script_data)
    MyAudio=speech.audio_procession()
    interface=ControlInterface.ControlInterface(enable_camera=True, show_img=True, enable_arm=False, enable_face=True, is_FullScreen=False)
    #建立對話模型，上為偵測意圖，下為對話用
    language='ch'
    normal_trigger=queue.Queue()
    minnan_trigger=queue.Queue()
    text_dict={'what':''}
    interrupt=False
    action="nothing"

    while True:
        if interrupt:#如果說話被打斷執行
            interrupt=False
            thread_1=threading.Thread(target=female_speak,args=("抱歉，你想說什麼?",1,'normal','normal',),daemon=True)
            thread_1.start()
            time.sleep(1)
        input=MyAudio.recording()#收音
        
        if input=="None":#無聲音檔不執行
            continue
        text_dict['what']=language_judger(input)
        if text_dict['what']==None:
            continue
        if text_dict['what']=='對話結束':#結束對話
            break
        intention=intention_answer.run_intention(texts=text_dict)
        #print(intention)
        result=Main_model.run(text_dict,intention=intention)#run GPT model
        conversation_history.save_text(result)#存本次對話
        conversation_history.load_text()#讀ltm
        #print(result)
        #interrupt=MyAudio.speaking(result[1],language=language)#機器人說話
        interrupt=interface.express(result[1],intention['emotion'],action,language)
