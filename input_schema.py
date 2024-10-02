INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Bark is a generative AI model that can turn text into audio. It works across different languages and can handle speech, music, and ambient noises. 
In this example, we are testing how to split a large paragraph into smaller chunks and process them using Bark for audio generation."]
    }, 
    "speaker": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["v2/en_speaker_1"]
    }
}
