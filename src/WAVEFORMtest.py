import torchaudio
import torch
import speechbrain as sb
from transformers import Wav2Vec2Model, Wav2Vec2Tokenizer
from speechbrain.inference.speaker import EncoderClassifier

def get_xvector_embedding(audio_file):
    # # Load the pre-trained XVector model
    # xvector_model = sb.pretrained.XVector.from_hparams(source='speechbrain/embedding/xvector', savedir='tmpdir')
    #
    # # Extract the xvector embedding
    # embedding = xvector_model.encode_file(audio_file)
    # return embedding

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                savedir="pretrained_models/spkrec-xvect-voxceleb")
    signal, fs = torchaudio.load(audio_file)
    xvector = classifier.encode_batch(signal)

    return xvector


def get_wav2vec_embedding(audio_file):
    # Load pretrained Wav2Vec2 model and tokenizer
    model_name = "facebook/wav2vec2-base-960h"
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

    # Load your audio file
    waveform, sample_rate = torchaudio.load(audio_file)

    # Ensure the audio is in the correct format
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Tokenize and get embeddings
    inputs = tokenizer(waveform.squeeze().numpy(), return_tensors="pt", padding="longest")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings


def print_embeddings_info(audio_file1, audio_file2):
    # Get embeddings for both audio files
    xvector1 = get_xvector_embedding(audio_file1)
    xvector2 = get_xvector_embedding(audio_file2)

    wav2vec1 = get_wav2vec_embedding(audio_file1)
    wav2vec2 = get_wav2vec_embedding(audio_file2)

    # Print the lengths of embeddings
    print(f"Length of xvector embedding for {audio_file1}: {xvector1.shape}")
    print(f"Length of xvector embedding for {audio_file2}: {xvector2.shape}")

    print(f"Length of wav2vec embedding for {audio_file1}: {wav2vec1.shape}")
    print(f"Length of wav2vec embedding for {audio_file2}: {wav2vec2.shape}")



# Replace these with your actual file paths
audio_file1 = "C:/Users/alvar/PycharmProjects/DementiaBank/audiosBERTprueba/S001.wav"
audio_file2 = "C:/Users/alvar/PycharmProjects/DementiaBank/audiosBERTprueba/S002.wav"

print_embeddings_info(audio_file1, audio_file2)

