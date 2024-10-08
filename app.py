from bark.api import semantic_to_waveform
from bark.generation import (generate_text_semantic, preload_models)
import numpy as np
import io
import base64
import soundfile as sf
import nltk
import gc

class InferlessPythonModel:
    
    def initialize(self):
        preload_models()
        nltk.download('punkt')
        nltk.download('punkt_tab')
        
    def infer(self, inputs, stream_output_handler):
        prompt = inputs["prompt"]
        speaker = inputs["speaker"]
        sentences = nltk.sent_tokenize(prompt)
        
        for i, sentence in enumerate(sentences):
            print(f"Processing chunk {i+1}: {sentence}")
            
            semantic_response = generate_text_semantic(
                text=sentence,
                history_prompt=speaker,
                temp=0.7,
                silent=True
            )
            
            audio_array = semantic_to_waveform(
                semantic_tokens=semantic_response,
                history_prompt=speaker,
                temp=0.7,
                silent=True
            )

            # Stream audio in smaller chunks
            sample_rate = 24000
            chunk_size = 4000  # Adjust this value as needed
            for start in range(0, len(audio_array), chunk_size):
                chunk = audio_array[start:start+chunk_size]
                buffer = io.BytesIO()
                sf.write(buffer, chunk, sample_rate, format='WAV')
                buffer.seek(0)
                base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
                stream_output_handler.send_streamed_output({"generated_audio": base64_audio})
            
            # Clear large objects from memory
            del semantic_response
            del audio_array
            gc.collect()

        stream_output_handler.finalise_streamed_output()

    def finalize(self, args):
        self.pipe = None
        gc.collect()