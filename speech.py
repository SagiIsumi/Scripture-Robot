from pathlib import Path
from datetime import datetime
import requests
import pyaudio
import wave
import threading
import pygame
from datetime import datetime
from openai import OpenAI
import numpy as np
from trilingual_module import female_speak,minnan_speak2
from opencc import OpenCC


class audio_procession():
    def __init__(self) -> None:
        self.audio_format=pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.triger=False

    def inner_female_speak(self,input_text):
        female_speak(input_text,volume=1,speed='fast',tone='normal')
        self.triger=True
    def inner_minnan_speak(self,input_text):
        minnan_speak2(input_text)
        self.triger=True 
    def speaking(self,text,language='ch')->None:#播音
        frames=[]
        interrupt=False
        self.triger=False
        try:
            if language == 'ch' or language=='en':
                speaker=threading.Thread(target=self.inner_female_speak, args=(text,), daemon=True)
                speaker.start()#播音執行緒
            else:
                speaker=threading.Thread(target=self.inner_minnan_speak, args=(text,), daemon=True)
                speaker.start()#播音執行緒
        except Exception as e:
            print(e)
        p=pyaudio.PyAudio()
        detecting_threashold=55#音量閾值
        stream=p.open(format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                #input_device_index=2,
                frames_per_buffer=self.chunk)
        try:
            while True:#開始收音
                for i in range(12):
                    data = stream.read(self.chunk)
                    frames.append(data)
                audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
                volume = np.abs(audio_data).mean()
                print(volume)
                if volume>detecting_threashold:#音量大於閾值結束播音
                    pygame.mixer.music.stop()
                    interrupt=True
                    print("over")
                    break
                elif self.triger:
                    interrupt=False
                    break
        except Exception as e:
            print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()#關閉串流
        return interrupt
            
    def recording(self)->str:
        p=pyaudio.PyAudio()
        frames=[]
        threashold=60 #音量閾值
        max_volume_threashold=45
        silent_chunk=0 #沉默時長
        silent_duration=3
        silent_chunks_threshold = int(silent_duration*self.rate/self.chunk)
        try:
            stream=p.open(format=self.audio_format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    #input_device_index=0,
                    frames_per_buffer=self.chunk)
            print("開始錄音")
        except Exception as e:
            print(e)
        try:#持續收音直到沉默時長<沉默閾值
            while True:
                data = stream.read(self.chunk)
                frames.append(data)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                if volume < threashold:
                    silent_chunk+=1
                else:
                    silent_chunk=0
                if silent_chunk>=silent_chunks_threshold:
                    break
        except Exception as e:
            print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()#關閉串流
        #檔案輸出_wav檔

        verify_data=np.frombuffer(b"".join(frames), dtype=np.int16)
        max_volume = np.abs(verify_data).mean()
        if max_volume<max_volume_threashold:#若收音為無聲音檔返回None
            return "None"

        time_object=datetime.now()
        currentTime = time_object.strftime("%d-%m-%y_%H-%M-%S")
        audio_path=Path("./audio_file")/Path(currentTime+".wav")
        audio_path=str(audio_path)
        with wave.open(audio_path,"wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))#音檔寫出
        return audio_path#返回檔案儲存路徑
    

    def speech_to_text(self,path, language='ch')->str:
        path=Path(path)
        try:
            if language=='ch':
                client=OpenAI()
                audio=open(path,"rb")
                response=client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    language="zh"
                )
                trans=OpenCC('s2twp')
                response=trans.convert(response.text)
                return response
            if language=='en':
                client=OpenAI()
                audio=open(path,"rb")
                response=client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    language="en"
                )
                return response.text
            if language=='minnan':
                url="http://119.3.22.24:3998/dotcasr"
                with open(path, 'rb') as file:
                    data = {'userid': '00001 ', 'token': '123356'}
                    response = requests.post(url, data=data, files={"file": file})
                return response.json()['result']
        except Exception as e:
            print(e)
            return "對話結束"


        
def Main(): #測試用
    test=audio_procession()
    mypath=audio_path=test.recording()
    human_word=test.text_to_speech(mypath)
    print(human_word)
    return human_word

if __name__=='__main__':
    Main()


















# time_object=datetime.now()
# currentTime = time_object.strftime("%d-%m-%y_%H-%M-%S")
# print(currentTime)
# audio_path=Path("./audio_file")/Path(currentTime+".mp3")
# print(audio_path)