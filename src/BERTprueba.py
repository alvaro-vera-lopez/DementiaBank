# Install the necessary libraries if not already installed
# !pip install SpeechRecognition transformers torch sentence-transformers

import speech_recognition as sr
from transformers import BertTokenizer, BertModel
import torch

# Step 1: Transcribe Audio to Text

# Initialize recognizer
recognizer = sr.Recognizer()

# Path to your audio file
audio_file = "C:/Users/alvar/PycharmProjects/DementiaBank/audiosBERTprueba/S002.wav"

# Open the audio file and recognize the speech
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)

# Use Google Speech Recognition to transcribe audio to text
try:
    text = recognizer.recognize_google(audio_data)
    print("Transcription: ", text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio.")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")

# Step 2: Generate BERT Embeddings

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize the text (convert to input format for BERT)
inputs = tokenizer(text, return_tensors='pt')

# Pass through the BERT model to get the embeddings
with torch.no_grad():
    outputs = model(**inputs)

# The last hidden state contains the embeddings for each token in the input
# To get the embedding of the [CLS] token, which is at index 0
cls_embedding = outputs.last_hidden_state[:, 0, :]

# Print the embedding for the [CLS] token
print("BERT Embedding for [CLS] token: ", cls_embedding)
print("Shape of the embedding: ", cls_embedding.shape)



# import os
# import torch
# from transformers import BertTokenizer, BertModel
# from speechbrain.inference.TTS import Tacotron2
# from speechbrain.inference.vocoders import HIFIGAN
# # Initialize the speech recognition model
# sb_model = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
# sb_vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
#
# # Initialize BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# def transcribe_audio(audio_path):
#     # Load the audio file and transcribe it
#     transcription = sb_model.encode_file(audio_path)
#     return transcription
#
# def get_bert_embeddings(text):
#     # Tokenize the text and get BERT embeddings
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Get the last hidden state (the embeddings)
#     embeddings = outputs.last_hidden_state
#     return embeddings
#
# def process_audio_files(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".wav"):
#             audio_path = os.path.join(directory, filename)
#             transcription = transcribe_audio(audio_path)
#             embeddings = get_bert_embeddings(transcription)
#             embedding_size = embeddings.size()
#             print(f"File: {filename}, BERT embedding size: {embedding_size}")
#
# # Replace 'path_to_audio_files' with the path to your directory containing .wav files
# process_audio_files('/audiosBERTprueba')



# import speechbrain as sr
# from transformers import BertTokenizer, BertModel
# import torch
#
# # Inicializa el reconocedor
# recognizer = sr.Recognizer()
#
# # Carga el archivo .wav
# with sr.AudioFile('C:/Users/alvar/PycharmProjects/DementiaBank/audiosBERTprueba/S001.wav') as source:
#     audio = recognizer.record(source)  # Lee el archivo de audio
#
# # Reconoce el texto
# try:
#     texto = recognizer.recognize_google(audio, language='es-ES')  # Cambia el idioma si es necesario
#     print("Texto reconocido:", texto)
# except sr.UnknownValueError:
#     print("No se pudo entender el audio")
# except sr.RequestError as e:
#     print(f"No se pudo solicitar resultados; {e}")
#
#
# # Carga el modelo y el tokenizador de BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Tokeniza el texto
# inputs = tokenizer(texto, return_tensors='pt', padding=True, truncation=True)
#
# # Genera los embeddings
# with torch.no_grad():
#     outputs = model(inputs)
#
# # Obtén los embeddings de la última capa
# embeddings = outputs.last_hidden_state
# print("Embeddings generados:", embeddings)


















# import os
# import torch
# from speechbrain.pretrained import EncoderDecoderASR
# from transformers import BertTokenizer, BertModel
#
# # Paths
# audio_folder = 'C:/Users/alvar/PycharmProjects/DementiaBank/audiosBERTprueba'
#
# # Directory where the ASR model will be saved locally
# local_model_dir = 'tmpdir_asr'
#
# # Initialize SpeechBrain ASR model
# asr = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=local_model_dir, force_reload=True)
#
# # Initialize BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# def get_text_from_audio(audio_path):
#     # Perform ASR
#     transcript = asr.transcribe_file(audio_path)
#     return transcript
#
# def get_bert_embeddings(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Return the last hidden state
#     return outputs.last_hidden_state
#
# def main():
#     for filename in os.listdir(audio_folder):
#         if filename.endswith('.wav'):
#             audio_path = os.path.join(audio_folder, filename)
#             text = get_text_from_audio(audio_path)
#             embeddings = get_bert_embeddings(text)
#             # Print the length of the BERT embeddings
#             print(f'File: {filename}')
#             print(f'Transcript: {text}')
#             print(f'Length of BERT embeddings: {embeddings.size()}')
#
# if __name__ == "__main__":
#     main()
#





