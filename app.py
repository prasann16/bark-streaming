from bark import SAMPLE_RATE, generate_audio, preload_models, generate_text_semantic, semantic_to_waveform
import numpy as np
import io
import base64
import soundfile as sf
import nltk

class InferlessPythonModel:
    
    def initialize(self):
        preload_models()
        nltk.download('punkt')
        
    def infer(self, inputs, stream_output_handler):
        prompt = inputs["prompt"]
        speaker = inputs["speaker"]
        sentences = nltk.sent_tokenize(prompt)
        for i, sentence in enumerate(sentences):
            
            print(f"Processing chunk {i+1}: {sentence}")
            semantic_response = generate_text_semantic(
                text=sentence,
                history_prompt=speaker,  # Example speaker preset
                temp=0.7,
                silent=True
            )
            audio_array = semantic_to_waveform(
                semantic_tokens=semantic_response,
                history_prompt=speaker,  # Use the same speaker preset
                temp=0.7,
                silent=True
            )

            # Write the audio data to the bytes buffer using soundfile
            sample_rate = 24000  # Bark outputs at 24kHz
            buffer = io.BytesIO()
            sf.write(byte_io, audio_array, sample_rate, format='WAV')
            buffer.seek(0)
            base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
            stream_output_handler.send_streamed_output({"generated_audio" : base64_audio})
        
        stream_output_handler.finalise_streamed_output()

    def finalize(self,args):
        self.pipe = None
