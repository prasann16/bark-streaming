from bark.api import semantic_to_waveform
from bark.generation import (generate_text_semantic, preload_models)
from bark import SAMPLE_RATE
import numpy as np
import io
import base64
import soundfile as sf
import nltk
import time
import os

os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

class InferlessPythonModel:
    
    def initialize(self):
        preload_models(
            text_use_gpu=True,
            coarse_use_gpu=True,
            fine_use_gpu=True,
            codec_use_gpu=True
        )
        nltk.download('punkt')
        nltk.download('punkt_tab')
        
    def infer(self, inputs, stream_output_handler):
        start_time = time.time()
        prompt = inputs["prompt"]
        speaker = inputs["speaker"]
        sentences = nltk.sent_tokenize(prompt)
        silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence
        print("Time taken to tokenize the input text: ", time.time() - start_time)
        
        for i, sentence in enumerate(sentences):
            print(f"Processing chunk {i+1}: {sentence}")
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=speaker,
                temp=0.6,
                min_eos_p=0.05,
                silent=True
            )
            print("Time to generate semantic tokens: ", time.time() - start_time)

            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=speaker, temp=0.7, silent=True)
            print("Time to generate audio array: ", time.time() - start_time)

            # Combine the current sentence audio with silence
            chunk_with_silence = np.concatenate([audio_array, silence.copy()])
            
            # Write the audio data (including silence) to the bytes buffer
            buffer = io.BytesIO()
            sf.write(buffer, chunk_with_silence, SAMPLE_RATE, format='WAV')
            buffer.seek(0)
            base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
            stream_output_handler.send_streamed_output({"generated_audio": base64_audio})
            print("Time taken to send the audio chunk: ", time.time() - start_time)

        stream_output_handler.finalise_streamed_output()

    def finalize(self, args):
        self.pipe = None