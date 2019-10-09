from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
     # convert the sound with altered frame rate to a standard frame rate
     # so that regular playback programs will work right. They often only
     # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

### Name of the file

sound = AudioSegment.from_wav(file name, "wav")

# sound = speed_change(sound, 0.95)

chunks = split_on_silence(sound, 
    # must be silent for at least half a second
    min_silence_len=300,

    # consider it silent if quieter than -30 dBFS
    silence_thresh=-30)

denoised = AudioSegment.empty()

for i, chunk in enumerate(chunks):
    chunk.export("files/chunks/chunk{0}.wav".format(i), format="wav")

for i, chunk in enumerate(chunks):
    os.system("python main.py --init_noise_std 0. --save_path segan_160 --batch_size 100 --g_nl prelu --weights SEGAN-41700 --test_wav files/chunks/chunk{0}.wav".format(i))

for i, chunk in enumerate(chunks):
    # Create a silence chunk that's 0.25 seconds long for padding.
    silence = AudioSegment.silent(duration=250)
    clean = AudioSegment.from_wav("test_clean_results/chunk{0}.wav".format(i))
    audio = silence + clean + silence
    denoised = denoised.append(audio, crossfade=0)

denoised.export(clean file name, format="wav")

os.system("rm -rf files/chunks/*")
os.system("rm -rf test_clean_results/*")