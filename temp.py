import pyaudio

def list_input_devices():
    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 获取系统上的所有音频设备
    device_count = p.get_device_count()
    
    print("Available input devices:")
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        # 只列出输入设备
        if device_info['maxInputChannels'] > 0:
            print(f"Device ID: {device_info['index']}")
            print(f"  Name: {device_info['name']}")
            print(f"  Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  Sample Rate: {device_info['defaultSampleRate']}")
            print("==========================================")
    
    # 关闭 PyAudio
    p.terminate()

# 列出所有输入设备
list_input_devices()
