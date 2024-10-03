INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Bark is a generative AI model that can turn text into audio It works across different languages and can handle speech music and ambient noises"]
    }, 
    "speaker": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["v2/en_speaker_1"]
    }
}
IS_STREAMING_OUTPUT = True

