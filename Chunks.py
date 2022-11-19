# chunk datset into 3-seconds pieces with a 50% overlap
from pydub import AudioSegment
import speech_recognition as sr
import os

inputdir = ''
outputdir = ''
# Input audio file to be sliced
for filename in os.listdir(inputdir):
    save_file_name = filename[:-4]
    audio = AudioSegment.from_file(inputdir + "/" + filename, "wav")
    n = len(audio)
    counter = 1
    interval = 3 * 1000
    overlap = 1.5 * 1000
    start = 0
    end = 0
    flag = 0
    for i in range(0, 2 * n, interval):
        if i == 0:
            start = 0
            end = interval
        else:
            start = end - overlap
            end = start + interval
        if end >= n:
            end = n
            flag = 1
            # Storing audio file from the defined start to end
        chunk = audio[start:end]
        # Filename / Path to store the sliced audio
        chunk_name = save_file_name + "_{0}.wav".format(i)

        # Store the sliced audio file to the defined path
        chunk.export(outputdir + "/" + chunk_name, format="wav")
        # Print information about the current chunk
        print("Processing chunk " + str(counter) + ". Start = "
              + str(start) + " end = " + str(end))
