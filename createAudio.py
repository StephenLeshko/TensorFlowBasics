import pyaudio
import wave
import audioop
import matplotlib.pyplot as plt #gives visual
import numpy as np #should always be imported for math/data vis

FRAMES_PER_BUFFER = 3200 #frames per second
FORMAT = pyaudio.paInt16 #if stereo, make 32
CHANNELS = 1 #if stereo, make 2
RATE = 16000 #recording frequency
print(FORMAT)

pa = pyaudio.PyAudio()

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

print('start recording')

seconds = 8
frames = []
second_tracking = 0
second_count = 0
#need it to be always listening...
#once it hears something that is a certain volume, it
#starts recording to a wave file
#while looping through wave file, it is looping while the 
#audio is a certain volume
#once it lowers enough, and after a certain amount of time,
#it creates the wav file and uses the model to check it
for i in range(0, int(RATE/FRAMES_PER_BUFFER*seconds)):
    data = stream.read(FRAMES_PER_BUFFER)
    
    frames.append(data) #this is what gets if/elsed
    second_tracking += 1
    if second_tracking == RATE/FRAMES_PER_BUFFER:
        # rms = audioop.rms(data, 2)
        # print('rms', rms) use an RMS of 500 to get it to work
        second_count += 1
        second_tracking = 0
        print(f'Time Left: {seconds - second_count} seconds')

#stop the stream
stream.stop_stream()
stream.close()
pa.terminate()

#write function to wave file; wb means write binary
obj = wave.open('left.wav', 'wb')
obj.setnchannels(CHANNELS)
obj.setsampwidth(pa.get_sample_size(FORMAT))
obj.setframerate(RATE)
obj.writeframes(b''.join(frames))
obj.close()

#opens the file to be read
file = wave.open('left.wav', 'rb')

sample_freq = file.getframerate()
frames = file.getnframes()
signal_wave = file.readframes(-1) #read everything

file.close() #close file since its now in local


time = frames / sample_freq


# if one channel use int16, if 2 use int32
audio_array = np.frombuffer(signal_wave, dtype=np.int16)

times = np.linspace(0, time, num=frames)

plt.figure(figsize=(15, 5))
plt.plot(times, audio_array)
plt.ylabel('Signal Wave')
plt.xlabel('Time (s)')
plt.xlim(0, time)
plt.title('The Thing I Just Recorded!!')
plt.show()

