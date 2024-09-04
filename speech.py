from pathlib import Path
from datetime import datetime
import pyaudio
import wave
import threading
import pygame
from datetime import datetime
from openai import OpenAI
import numpy as np
from trilingual_module import female_speak


class audio_procession():
    def __init__(self) -> None:
        self.audio_format=pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.triger=False

    def intra_female_speak(self,input_text):
        female_speak(input_text,volume=1,speed='fast',tone='normal')
        self.triger=True
    def speaking(self,text,interrupt)->None:
        frames=[]
        speaker=threading.Thread(target=self.intra_female_speak, args=(text,), daemon=True)
        speaker.start()
        p=pyaudio.PyAudio()
        detecting_threashold=70
        stream=p.open(format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                #input_device_index=2,
                frames_per_buffer=self.chunk)
        while True:#開始收音
                for i in range(12):
                    data = stream.read(self.chunk)
                    frames.append(data)
                audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
                volume = np.abs(audio_data).mean()
                print(volume)
                if volume>detecting_threashold:
                    pygame.mixer.music.stop()
                    interrupt=True
                    print("over")
                    break
                elif self.triger:
                    self.triger=False
                    interrupt=False
                    break
    def recording(self)->str:
        p=pyaudio.PyAudio()
        frames=[]
        threashold=70 #音量閾值
        max_volume_threashold=40
        silent_chunk=0 #沉默時長
        silent_duration=3
        silent_chunks_threshold = int(silent_duration*self.rate/self.chunk)

        stream=p.open(format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                #input_device_index=0,
                frames_per_buffer=self.chunk)
        print("開始錄音")
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
        if max_volume<max_volume_threashold:
            return "None"

        time_object=datetime.now()
        currentTime = time_object.strftime("%d-%m-%y_%H-%M-%S")
        audio_path=Path("./audio_file")/Path(currentTime+".wav")
        audio_path=str(audio_path)
        with wave.open(audio_path,"wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
        return audio_path#返回檔案儲存路徑
    

    def text_to_speech(self,path)->str:
        client=OpenAI()
        path=Path(path)
        audio=open(path,"rb")
        response=client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            language="zh"
        )
        return response.text

        
def Main():
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