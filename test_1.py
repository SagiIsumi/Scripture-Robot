from core_LLM import Chatmodel
from vir_db import VectorDB
import speech
import multiprocessing as mp
import threading
import time
from trilingual_module import female_speak
#以上為 input module

if __name__=='__main__':
    conversation_history=VectorDB(persist_directory='.\conversation_db')
    script_data=VectorDB(persist_directory='.\script_db')
    script_data.load_text(path='.\scripts')
    #建立本地程式庫，建好後會有conversation_db和script_db兩個資料夾
    intention_answer=Chatmodel(promptpath='.\prompts\intention_prompt.txt')
    Main_model=Chatmodel(promptpath='.\prompts\chat_prompt.txt',
                            longmemory_db=conversation_history,local_db=script_data)
    MyAudio=speech.audio_procession()
    #建立對話模型，上為偵測意圖，下為對話用
    text_dict={'what':''}
    interrupt=False

    while True:
        if interrupt:#如果說話被打斷執行
            interrupt=False
            thread_1=threading.Thread(target=female_speak,args=("抱歉，你想說什麼?",1,'normal','normal',),daemon=True)
            thread_1.start()
            time.sleep(1)
        input=MyAudio.recording()#收音
        if input=="None":#無聲音檔不執行
            continue
        text_dict['what']=MyAudio.text_to_speech(input)
        if text_dict['what']=='對話結束':#結束對話
            break
        intention=intention_answer.run_intention(texts=text_dict)
        #print(intention)
        result=Main_model.run(text_dict,intention=intention)#run GPT model
        conversation_history.save_text(result)#存本次對話
        conversation_history.load_text()#讀ltm
        #print(result)
        interrupt=MyAudio.speaking(result[1])#機器人說話

