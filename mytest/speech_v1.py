from pathlib import Path
from datetime import datetime
import pyaudio
import wave
import threading
from datetime import datetime
from openai import OpenAI
import numpy as np
import json

class audio_procession():
    def __init__(self) -> None:
        self.audio_format=pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024

    def recording(self)->str:
        p=pyaudio.PyAudio()
        frames=[]
        threashold=200 #音量閾值
        silent_chunk=0 #沉默時長
        silent_duration=5
        silent_chunks_threshold = int(silent_duration*self.rate/self.chunk)

        stream=p.open(format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                #input_device_index=0,
                frames_per_buffer=self.chunk)
        print("開始錄音")
        try:
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
                    print(silent_chunk)
                    break
        except Exception as e:
            print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        time_object=datetime.now()
        currentTime = time_object.strftime("%d-%m-%y_%H-%M-%S")
        audio_path=Path("./audio_file")/Path(currentTime+".wav")
        audio_path=str(audio_path)
        print(audio_path)
        with wave.open(audio_path,"wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
        return audio_path
    def text_to_speech(self,path):
        client=OpenAI()
        path=Path(path)
        audio=open(path,"rb")
        response=client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            language="zh"
        )
        print(response.text)
        with open("./call.json","w",encoding="utf-8")as f:
            a=json.loads(response.to_json())
            print(type(a))
            b=json.dumps(a,ensure_ascii=False)
            print(b)
            f.write(b)
        
def Main():
    test=audio_procession()
    # audio_path=test.recording()
    test.text_to_speech("./audio_file/02-09-24_13-14-32.mp3")

if __name__=='__main__':
    Main()