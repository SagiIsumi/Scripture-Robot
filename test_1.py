from core_LLM import Chatmodel
from vir_db import VectorDB
import speech
import multiprocessing as mp
import threading
import time
from trilingual_module import female_speak

if __name__=='__main__':
    conversation_history=VectorDB(persist_directory='.\conversation_db')
    script_data=VectorDB(persist_directory='.\script_db')
    script_data.load_text(path='.\scripts')
    intention_answer=Chatmodel(promptpath='.\prompts\intention_prompt.txt')
    Main_model=Chatmodel(promptpath='.\prompts\chat_prompt.txt',
                            longmemory_db=conversation_history,local_db=script_data)
    MyAudio=speech.audio_procession()
    text_dict={'what':''}
    interrupt=False

    while True:
        if interrupt:
            interrupt=False
            thread_1=threading.Thread(target=female_speak,args=("抱歉，你想說什麼?",1,'normal','normal',),daemon=True)
            thread_1.start()
            time.sleep(1)
        input=MyAudio.recording()
        if input=="None":
            continue
        text_dict['what']=MyAudio.text_to_speech(input)
        if text_dict['what']=='對話結束':
            break
        intention=intention_answer.run_intention(texts=text_dict)
        #print(intention)
        result=Main_model.run(text_dict,intention=intention)
        conversation_history.save_text(result)
        conversation_history.load_text()
        #print(result)
        thread_2=threading.Thread(target=MyAudio.speaking,args=(result[1],interrupt),daemon=True)
        thread_2.start()
        thread_2.join()
        print(interrupt)
