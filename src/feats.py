import os
import librosa
import numpy as np
import opensmile
import torch
import torchaudio
from spafe.features.bfcc import bfcc
from spafe.features.cqcc import cqcc
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc
from spafe.features.lpc import lpcc
from spafe.features.mfcc import mfcc, imfcc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.pncc import pncc
from spafe.features.psrcc import psrcc
from spafe.features.rplp import plp, rplp
import pandas as pd
from transformers import BertTokenizer, BertModel
import ast
import json
from transformers import Wav2Vec2Model, Wav2Vec2Tokenizer
from speechbrain.inference.speaker import EncoderClassifier
from scipy.stats import iqr

dir_bas = "/home/projects/avera/DementiaBank/dataset/Extracted_data"

db = "/home/projects/avera/DementiaBank/dataset/ADReSS_db.csv"
df = pd.read_csv(db)

# Initialize BERT tokenizer and model
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

classifier_xv = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                                savedir="pretrained_models/spkrec-xvect-voxceleb")

model_w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer_w2v = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")



class FeatureExtractor:
    """
    Class for feature extraction
    args: input arguments dictionary
    Mandatory arguments: resampling_rate, feature_type, window_size, hop_length
    For MFCC: f_max, n_mels, n_mfcc
    For MelSpec/logMelSpec: f_max, n_mels
    Optional arguments: compute_deltas, compute_delta_deltas
    """

    def __init__(self, arguments: dict):

        self.args = arguments
        self.audio_path = None
        self.resampling_rate = self.args['resampling_rate']
        assert (arguments['feature_type'] in ['MFCC', 'MelSpec', 'logMelSpec',
                                              'ComParE_2016_energy', 'ComParE_2016_voicing',
                                              'ComParE_2016_spectral', 'ComParE_2016_basic_spectral',
                                              'ComParE_2016_mfcc', 'ComParE_2016_rasta', 'ComParE_2016_llds',
                                              'Spafe_mfcc', 'Spafe_imfcc', 'Spafe_cqcc', 'Spafe_gfcc', 'Spafe_lfcc',
                                              'Spafe_lpc', 'Spafe_lpcc', 'Spafe_msrcc', 'Spafe_ngcc', 'Spafe_pncc',
                                              'Spafe_psrcc', 'Spafe_plp', 'Spafe_rplp', 'BERT_embedding', 'XVector', 'wav2vec',
                                              'embMFCC', 'embRasta', 'embBasicSpectral', 'embVoicing', 'embEnergy',
                                              'bert_MFCC', 'bert_wav2vec', 'MFCC_spectral']), \
            'Not expected feauture type' \

        nfft = int(float(self.args.get('window_size', 0) * 1e-3 * self.resampling_rate))
        hop_length = int(float(self.args.get('hop_length', 0)) * 1e-3 * self.resampling_rate)

        if self.args['feature_type'] == 'MFCC':
            self.feature_transform = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate,
                                                                n_mfcc=int(self.args['n_mfcc']),
                                                                melkwargs={'n_fft': nfft,
                                                                           'n_mels': int(self.args['n_mels']),
                                                                           'f_max': int(self.args['f_max']),
                                                                           'hop_length': hop_length})
        elif self.args['feature_type'] in ['MelSpec', 'logMelSpec']:
            self.feature_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.resampling_rate,
                                                                          n_fft=nfft,
                                                                          n_mels=int(self.args['n_mels']),
                                                                          f_max=int(self.args['f_max']),
                                                                          hop_length=hop_length)
        elif 'ComParE_2016' in self.args['feature_type']:
            self.feature_transform = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                                     feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                                     sampling_rate=self.resampling_rate)
        elif 'Spafe_' in self.args['feature_type']:
            spafe_feature_transformers = {'Spafe_mfcc': mfcc,
                                          'Spafe_imfcc': imfcc,
                                          'Spafe_bfcc': bfcc,
                                          'Spafe_cqcc': cqcc,
                                          'Spafe_gfcc': gfcc,
                                          'Spafe_lfcc': lfcc,
                                          'Spafe_lpc': lpc,
                                          'Spafe_lpcc': lpcc,
                                          'Spafe_msrcc': msrcc,
                                          'Spafe_ngcc': ngcc,
                                          'Spafe_pncc': pncc,
                                          'Spafe_psrcc': psrcc,
                                          'Spafe_plp': plp,
                                          'Spafe_rplp': rplp}
            self.feature_transform = spafe_feature_transformers[self.args['feature_type']]
        else:
            if not (self.args['feature_type'] == 'BERT_embedding' or self.args['feature_type'] == 'XVector'
                    or self.args['feature_type'] == 'wav2vec' or self.args['feature_type'] == 'embMFCC'
                    or self.args['feature_type'] == 'embRasta' or self.args['feature_type'] == 'embBasicSpectral'
                    or self.args['feature_type'] == 'embVoicing' or self.args['feature_type'] == 'embEnergy'
                    or self.args['feature_type'] == 'bert_MFCC' or self.args['feature_type'] == 'bert_wav2vec'
                    or self.args['feature_type'] == 'MFCC_spectral'):
                raise ValueError('Feature type not implemented')

    def _read_audio(self, audio_file_path):
        """
         The code above implements SAD using the librosa.effects.split() function with a threshold of top_db=30, which
         separates audio regions where the amplitude is lower than the threshold.
         The pre-emphasis filter is applied using the librosa.effects.preemphasis() function with a coefficient of 0.97.
         This filter emphasizes the high-frequency components of the audio signal,
         which can improve the quality of the speech signal.
         Finally, the code normalizes the audio signal to have maximum amplitude of 1
         :param audio_file_path: audio file path
         :return: audio signal and sampling rate
         """
        # load the audio file
        s, sr = librosa.load(audio_file_path, mono=True)
        # resample
        if (self.resampling_rate is not None) or (sr < self.resampling_rate):
            s = librosa.resample(y=s, orig_sr=sr, target_sr=self.resampling_rate)
            sr = self.resampling_rate
        # apply speech activity detection
        speech_indices = librosa.effects.split(s, top_db=30)
        s = np.concatenate([s[start:end] for start, end in speech_indices])
        # apply a pre-emphasis filter
        s = librosa.effects.preemphasis(s, coef=0.97)

        # normalize
        s /= np.max(np.abs(s))

        '''
        try:
            s /= np.max(np.abs(s))
        except ZeroDivisionError:
            # Handle the case where division by zero occurs.
            s = np.zeros_like(s)  # Or any other appropriate action.
    

        if np.isnan(s).any() or np.isinf(s).any():
            # Handle the case where NaN or infinity values are present.
            s = np.zeros_like(s)  # Or any other appropriate action.
        else:
            s /= np.max(np.abs(s))
        '''
        epsilon = 1e-9
        #s /= (np.max(np.abs(s)) + epsilon)

        return torch.from_numpy(s), sr

    @staticmethod
    def compute_sad(sig, fs, threshold=0.0001, sad_start_end_sil_length=100, sad_margin_length=50):
        """ Compute threshold based sound activity """
        # Leading/Trailing margin
        sad_start_end_sil_length = int(sad_start_end_sil_length * 1e-3 * fs)
        # Margin around active samples
        sad_margin_length = int(sad_margin_length * 1e-3 * fs)

        sample_activity = np.zeros(sig.shape)
        sample_activity[np.power(sig, 2) > threshold] = 1
        sad = np.zeros(sig.shape)
        for i in range(sample_activity.shape[1]):
            if sample_activity[0, i] == 1:
                sad[0, i - sad_margin_length:i + sad_margin_length] = 1
        sad[0, 0:sad_start_end_sil_length] = 0
        sad[0, -sad_start_end_sil_length:] = 0
        return sad

    def _do_feature_extraction(self, s, sr, audio_id):
        """ Feature preparation
        Steps:
        1. Apply feature extraction to waveform
        2. Convert amplitude to dB if required
        3. Append delta and delta-delta features
        """
        F = None
        F2 = None
        if self.args['feature_type'] == 'BERT_embedding':
            row = df.loc[df['root'] == audio_id]

            transcriptions_list = ast.literal_eval(row['transcriptions'].values[0])
            # print(f'Longitud: {len(transcriptions_list)}')

            for i, transcription in enumerate(transcriptions_list):
                try:
                    inputs = tokenizer_bert(transcription, return_tensors='pt', truncation=True, padding=True)

                    # Pass through the BERT model to get the embeddings
                    with torch.no_grad():
                        outputs = model_bert(**inputs)

                    # The last hidden state contains the embeddings for each token in the input
                    # To get the embedding of the [CLS] token, which is at index 0
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    # print(cls_embedding.squeeze(0))
                    # print(f'Longitud: {cls_embedding.squeeze(0).shape}')

                    if F is None:
                        F = []
                    #F.append(cls_embedding.squeeze(0))
                    F.append(np.array(cls_embedding.squeeze(0)))
                    #print(f'Tipo de F dentro del for {type(F)}')

                except Exception as e:
                    print("Error processing transcription. Error: {e}")

            if len(F) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)  # Stack along a new dimension if F is a list of arrays
            # F = torch.from_numpy(F).T
            
            #print(f'F en feats {F.shape}')
            #print(f'F en feats {F}')
            #print(f'Tipo de F fuera del for {type(F)}')
            return F

        if self.args['feature_type'] == 'XVector':
            row = df.loc[df['root'] == audio_id]

            roots_list = ast.literal_eval(row['audio_roots'].values[0])
            # print(f'Longitud: {len(transcriptions_list)}')

            for i, audio_root in enumerate(roots_list):
                try:
                    audio_root = os.path.join(dir_bas, audio_root)
                    signal, fs = torchaudio.load(audio_root)
                    xvector = classifier_xv.encode_batch(signal)
                    xvector = xvector.squeeze()
                    print(f'XVector shape: {xvector.shape}')
                    print(f'XVector type: {type(xvector)}')
                    if F is None:
                        F = []
                    F.append(np.array(xvector))
                    # print(f'Tipo de F dentro del for {type(F)}')

                except Exception as e:
                    print("Error processing audio. Error: {e}")

            if len(F) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)  # Stack along a new dimension if F is a list of arrays
            # F = torch.from_numpy(F).T

            print(f'F en feats {F.shape}')
            print(f'F en feats {F}')
            print(f'Tipo de F en feats.py {type(F)}')
            return F

        if self.args['feature_type'] == 'wav2vec':
            row = df.loc[df['root'] == audio_id]

            roots_list = ast.literal_eval(row['audio_roots'].values[0])
            # print(f'Longitud: {len(transcriptions_list)}')

            for i, audio_root in enumerate(roots_list):
                try:
                    audio_root = os.path.join(dir_bas, audio_root)
                    waveform, sr = torchaudio.load(audio_root)
                    if sr != 16000:
                        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

                    inputs = tokenizer_w2v(waveform.squeeze().numpy(), return_tensors="pt", padding="longest")
                    with torch.no_grad():
                        outputs = model_w2v(**inputs)
                    #w2v = outputs.last_hidden_state[:, 0, :]
                    # w2v = outputs.last_hidden_state.mean(dim=1)
                    # w2v = w2v.squeeze()

                    w2v = outputs.last_hidden_state
                    w2v = w2v.squeeze()

                    w2v_mean = w2v.mean(dim=0)
                    w2v_std = w2v.std(dim=0)
                    w2v_iqr = iqr(w2v.numpy(), axis=0)

                    w2v = np.concatenate((w2v_mean.numpy(), w2v_std.numpy(), w2v_iqr), axis=0)

                    print(f'wav2vec shape: {w2v.shape}')
                    print(f'wav2vec type: {type(w2v)}')

                    if F is None:
                        F = []
                    F.append(np.array(w2v))
                    # print(f'Tipo de F dentro del for {type(F)}')

                except Exception as e:
                    print("Error processing audio. Error: {e}")

            if len(F) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)  # Stack along a new dimension if F is a list of arrays
            # F = torch.from_numpy(F).T

            print(f'F en feats {F.shape}')
            print(f'F en feats {F}')
            print(f'Tipo de F en feats.py {type(F)}')
            return F

        if self.args['feature_type'] == 'embMFCC':
            nfft = int(float(self.args.get('window_size', 0) * 1e-3 * self.resampling_rate))
            hop_length = int(float(self.args.get('hop_length', 0)) * 1e-3 * self.resampling_rate)

            mfcc = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate,
                                              n_mfcc=int(self.args['n_mfcc']),
                                              melkwargs={'n_fft': nfft,
                                                         'n_mels': int(self.args['n_mels']),
                                                         'f_max': int(self.args['f_max']),
                                                         'hop_length': hop_length})

            row = df.loc[df['root'] == audio_id]

            roots_list = ast.literal_eval(row['audio_roots'].values[0])
            # print(f'Longitud: {len(transcriptions_list)}')

            for i, audio_root in enumerate(roots_list):
                try:
                    audio_root = os.path.join(dir_bas, audio_root)
                    s, sr = torchaudio.load(audio_root)

                    mfccs = mfcc(s)
                    if self.args.get('extra_features', False):
                        deltas = torchaudio.functional.compute_deltas(mfccs)
                        delta_deltas = torchaudio.functional.compute_deltas(deltas)
                        mfccs = torch.cat((mfccs, deltas, delta_deltas), dim=1)

                    mfccs = mfccs.squeeze(0)

                    mfcc_mean = mfccs.mean(dim=1)
                    mfcc_std = mfccs.std(dim=1)
                    mfcc_iqr = iqr(mfccs.numpy(), axis=1)

                    embMFCC = np.concatenate((mfcc_mean.numpy(), mfcc_std.numpy(), mfcc_iqr), axis=0)

                    print(f'embMFCC shape: {embMFCC.shape}')
                    print(f'embMFCC type: {type(embMFCC)}')

                    if F is None:
                        F = []
                    F.append(np.array(embMFCC))
                    # print(f'Tipo de F dentro del for {type(F)}')

                except Exception as e:
                    print("Error processing audio. Error: {e}")

            if len(F) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)  # Stack along a new dimension if F is a list of arrays
            # F = torch.from_numpy(F).T

            print(f'F en feats {F.shape}')
            print(f'F en feats {F}')
            print(f'Tipo de F en feats.py {type(F)}')
            return F

        if self.args['feature_type'] == 'embRasta':
            compare = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                            sampling_rate=self.resampling_rate)

            row = df.loc[df['root'] == audio_id]

            roots_list = ast.literal_eval(row['audio_roots'].values[0])
            # print(f'Longitud: {len(transcriptions_list)}')

            for i, audio_root in enumerate(roots_list):
                try:
                    audio_root = os.path.join(dir_bas, audio_root)
                    s, fs = self._read_audio(audio_root)
                    s = s[None, :]

                    rastas = compare.process_signal(s, sr)

                    # Definir el subconjunto de características Rasta
                    feature_subset = [
                        'audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 'audSpec_Rfilt_sma[2]',
                        'audSpec_Rfilt_sma[3]', 'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]',
                        'audSpec_Rfilt_sma[6]', 'audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]',
                        'audSpec_Rfilt_sma[9]', 'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]',
                        'audSpec_Rfilt_sma[12]', 'audSpec_Rfilt_sma[13]', 'audSpec_Rfilt_sma[14]',
                        'audSpec_Rfilt_sma[15]', 'audSpec_Rfilt_sma[16]', 'audSpec_Rfilt_sma[17]',
                        'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]',
                        'audSpec_Rfilt_sma[21]', 'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]',
                        'audSpec_Rfilt_sma[24]', 'audSpec_Rfilt_sma[25]'
                    ]

                    # Extraer el subconjunto de características y convertir a NumPy
                    rastas = rastas[feature_subset].to_numpy()
                    rastas = np.nan_to_num(rastas)
                    rastas = torch.from_numpy(rastas).T

                    if self.args.get('extra_features', False):
                        deltas = torchaudio.functional.compute_deltas(rastas)
                        delta_deltas = torchaudio.functional.compute_deltas(deltas)
                        rastas = torch.cat((rastas, deltas, delta_deltas), dim=0)

                    rastas = rastas.squeeze(0)

                    rastas_mean = rastas.mean(dim=1)
                    rastas_std = rastas.std(dim=1)
                    rastas_iqr = iqr(rastas.numpy(), axis=1)

                    embRasta = np.concatenate((rastas_mean.numpy(), rastas_std.numpy(), rastas_iqr), axis=0)

                    print(f'embRasta shape: {embRasta.shape}')
                    print(f'embRasta type: {type(embRasta)}')

                    if F is None:
                        F = []
                    F.append(np.array(embRasta))
                    # print(f'Tipo de F dentro del for {type(F)}')

                except Exception as e:
                    print("Error processing audio. Error: {e}")

            if len(F) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)  # Stack along a new dimension if F is a list of arrays
            # F = torch.from_numpy(F).T

            print(f'F en feats {F.shape}')
            print(f'F en feats {F}')
            print(f'Tipo de F en feats.py {type(F)}')
            return F

        if self.args['feature_type'] == 'embBasicSpectral':
            compare = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                      feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                      sampling_rate=self.resampling_rate)

            row = df.loc[df['root'] == audio_id]

            roots_list = ast.literal_eval(row['audio_roots'].values[0])
            # print(f'Longitud: {len(transcriptions_list)}')

            for i, audio_root in enumerate(roots_list):
                try:
                    audio_root = os.path.join(dir_bas, audio_root)
                    s, fs = self._read_audio(audio_root)
                    s = s[None, :]

                    basic_spectral = compare.process_signal(s, sr)

                    # Definir el subconjunto de características Rasta
                    feature_subset = ['pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                                                'pcm_fftMag_spectralRollOff25.0_sma',
                                                'pcm_fftMag_spectralRollOff50.0_sma',
                                                'pcm_fftMag_spectralRollOff75.0_sma',
                                                'pcm_fftMag_spectralRollOff90.0_sma',
                                                'pcm_fftMag_spectralFlux_sma',
                                                'pcm_fftMag_spectralCentroid_sma',
                                                'pcm_fftMag_spectralEntropy_sma',
                                                'pcm_fftMag_spectralVariance_sma',
                                                'pcm_fftMag_spectralSkewness_sma',
                                                'pcm_fftMag_spectralKurtosis_sma',
                                                'pcm_fftMag_spectralSlope_sma',
                                                'pcm_fftMag_psySharpness_sma',
                                                'pcm_fftMag_spectralHarmonicity_sma']

                    # Extraer el subconjunto de características y convertir a NumPy
                    basic_spectral = basic_spectral[feature_subset].to_numpy()
                    basic_spectral = np.nan_to_num(basic_spectral)
                    basic_spectral = torch.from_numpy(basic_spectral).T

                    # Mostrar la forma de las características
                    print(f'Características BE shape: {basic_spectral.shape}')

                    if self.args.get('extra_features', False):
                        deltas = torchaudio.functional.compute_deltas(basic_spectral)
                        delta_deltas = torchaudio.functional.compute_deltas(deltas)
                        basic_spectral = torch.cat((basic_spectral, deltas, delta_deltas), dim=0)

                    basic_spectral = basic_spectral.squeeze(0)

                    basic_spectral_mean = basic_spectral.mean(dim=1)
                    basic_spectral_std = basic_spectral.std(dim=1)
                    basic_spectral_iqr = iqr(basic_spectral.numpy(), axis=1)

                    embBasicSpectral = np.concatenate((basic_spectral_mean.numpy(), basic_spectral_std.numpy(), basic_spectral_iqr), axis=0)

                    print(f'embBasicSpectral shape: {embBasicSpectral.shape}')
                    print(f'embBasicSpectral type: {type(embBasicSpectral)}')

                    if F is None:
                        F = []
                    F.append(np.array(embBasicSpectral))
                    # print(f'Tipo de F dentro del for {type(F)}')

                except Exception as e:
                    print("Error processing audio. Error: {e}")

            if len(F) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)  # Stack along a new dimension if F is a list of arrays
            # F = torch.from_numpy(F).T

            print(f'F en feats {F.shape}')
            print(f'F en feats {F}')
            print(f'Tipo de F en feats.py {type(F)}')
            return F

        if self.args['feature_type'] == 'embVoicing':
            compare = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                      feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                      sampling_rate=self.resampling_rate)

            row = df.loc[df['root'] == audio_id]

            roots_list = ast.literal_eval(row['audio_roots'].values[0])
            # print(f'Longitud: {len(transcriptions_list)}')

            for i, audio_root in enumerate(roots_list):
                try:
                    audio_root = os.path.join(dir_bas, audio_root)
                    s, fs = self._read_audio(audio_root)
                    s = s[None, :]
                    voicing = compare.process_signal(s, sr)

                    feature_subset = ['F0final_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
                                            'jitterDDP_sma', 'shimmerLocal_sma', 'logHNR_sma']

                    voicing = voicing[feature_subset].to_numpy()
                    voicing = np.nan_to_num(voicing)
                    voicing = torch.from_numpy(voicing).T

                    print(f'Características voicing shape: {voicing.shape}')

                    if self.args.get('extra_features', False):
                        deltas = torchaudio.functional.compute_deltas(voicing)
                        delta_deltas = torchaudio.functional.compute_deltas(deltas)
                        voicing = torch.cat((voicing, deltas, delta_deltas), dim=0)

                    voicing = voicing.squeeze(0)

                    voicing_mean = voicing.mean(dim=1)
                    voicing_std = voicing.std(dim=1)
                    voicing_iqr = iqr(voicing.numpy(), axis=1)

                    embVoicing = np.concatenate((voicing_mean.numpy(), voicing_std.numpy(), voicing_iqr), axis=0)
                    print(f'embVoicing shape: {embVoicing.shape}')

                    if F is None:
                        F = []
                    F.append(np.array(embVoicing))

                except Exception as e:
                    print("Error processing audio. Error: {e}")

            if len(F) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)
            print(f'F en feats {F.shape}')
            print(f'F en feats {F}')
            print(f'Tipo de F en feats.py {type(F)}')
            return F

        if self.args['feature_type'] == 'embEnergy':
            compare = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                      feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                      sampling_rate=self.resampling_rate)

            row = df.loc[df['root'] == audio_id]

            roots_list = ast.literal_eval(row['audio_roots'].values[0])
            # print(f'Longitud: {len(transcriptions_list)}')

            for i, audio_root in enumerate(roots_list):
                try:
                    audio_root = os.path.join(dir_bas, audio_root)
                    s, fs = self._read_audio(audio_root)
                    s = s[None, :]
                    energy = compare.process_signal(s, sr)

                    feature_subset = ['audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
                                            'pcm_RMSenergy_sma', 'pcm_zcr_sma']

                    energy = energy[feature_subset].to_numpy()
                    energy = np.nan_to_num(energy)
                    energy = torch.from_numpy(energy).T

                    print(f'Características energy shape: {energy.shape}')

                    if self.args.get('extra_features', False):
                        deltas = torchaudio.functional.compute_deltas(energy)
                        delta_deltas = torchaudio.functional.compute_deltas(deltas)
                        energy = torch.cat((energy, deltas, delta_deltas), dim=0)

                    energy = energy.squeeze(0)

                    energy_mean = energy.mean(dim=1)
                    energy_std = energy.std(dim=1)
                    energy_iqr = iqr(energy.numpy(), axis=1)

                    embEnergy = np.concatenate((energy_mean.numpy(), energy_std.numpy(), energy_iqr), axis=0)
                    print(f'embEnergy shape: {embEnergy.shape}')

                    if F is None:
                        F = []
                    F.append(np.array(embEnergy))

                except Exception as e:
                    print("Error processing audio. Error: {e}")

            if len(F) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)
            print(f'F en feats {F.shape}')
            print(f'F en feats {F}')
            print(f'Tipo de F en feats.py {type(F)}')
            return F

        if self.args['feature_type'] == 'bert_MFCC':

            nfft = int(float(self.args.get('window_size', 0) * 1e-3 * self.resampling_rate))
            hop_length = int(float(self.args.get('hop_length', 0)) * 1e-3 * self.resampling_rate)
            mfcc = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate,
                                              n_mfcc=int(self.args['n_mfcc']),
                                              melkwargs={'n_fft': nfft,
                                                         'n_mels': int(self.args['n_mels']),
                                                         'f_max': int(self.args['f_max']),
                                                         'hop_length': hop_length})

            row = df.loc[df['root'] == audio_id]

            transcriptions_list = ast.literal_eval(row['transcriptions'].values[0])
            roots_list = ast.literal_eval(row['audio_roots'].values[0])

            for i, transcription in enumerate(transcriptions_list):
                try:
                    audio_root = roots_list[i]

                    # BERT embedding
                    inputs = tokenizer_bert(transcription, return_tensors='pt', truncation=True, padding=True)

                    # Pass through the BERT model to get the embeddings
                    with torch.no_grad():
                        outputs = model_bert(**inputs)

                    cls_embedding = outputs.last_hidden_state[:, 0, :]

                    if F is None:
                        F = []

                    F.append(np.array(cls_embedding.squeeze(0)))

                    # MFCC embedding
                    audio_root = os.path.join(dir_bas, audio_root)
                    s, sr = torchaudio.load(audio_root)

                    mfccs = mfcc(s)
                    if self.args.get('extra_features', False):
                        deltas = torchaudio.functional.compute_deltas(mfccs)
                        delta_deltas = torchaudio.functional.compute_deltas(deltas)
                        mfccs = torch.cat((mfccs, deltas, delta_deltas), dim=1)

                    mfccs = mfccs.squeeze(0)

                    mfcc_mean = mfccs.mean(dim=1)
                    mfcc_std = mfccs.std(dim=1)
                    mfcc_iqr = iqr(mfccs.numpy(), axis=1)

                    embMFCC = np.concatenate((mfcc_mean.numpy(), mfcc_std.numpy(), mfcc_iqr), axis=0)

                    if F2 is None:
                        F2 = []
                    F2.append(np.array(embMFCC))

                except Exception as e:
                    print("Error processing transcription. Error: {e}")

            if len(F) == 0 or len(F2) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)
            F2 = np.stack(F2)

            # concatenar horizontalmente los embeddings
            F = np.concatenate((F, F2), axis=1)
            return F

        if self.args['feature_type'] == 'MFCC_spectral':
            nfft = int(float(self.args.get('window_size', 0) * 1e-3 * self.resampling_rate))
            hop_length = int(float(self.args.get('hop_length', 0)) * 1e-3 * self.resampling_rate)

            mfcc = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate,
                                              n_mfcc=int(self.args['n_mfcc']),
                                              melkwargs={'n_fft': nfft,
                                                         'n_mels': int(self.args['n_mels']),
                                                         'f_max': int(self.args['f_max']),
                                                         'hop_length': hop_length})

            compare = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                      feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                      sampling_rate=self.resampling_rate)

            row = df.loc[df['root'] == audio_id]

            roots_list = ast.literal_eval(row['audio_roots'].values[0])
            # print(f'Longitud: {len(transcriptions_list)}')

            for i, audio_root in enumerate(roots_list):
                try:
                    audio_root = os.path.join(dir_bas, audio_root)
                    s, sr = torchaudio.load(audio_root)

                    # MFCC
                    mfccs = mfcc(s)
                    if self.args.get('extra_features', False):
                        deltas = torchaudio.functional.compute_deltas(mfccs)
                        delta_deltas = torchaudio.functional.compute_deltas(deltas)
                        mfccs = torch.cat((mfccs, deltas, delta_deltas), dim=1)

                    mfccs = mfccs.squeeze(0)

                    mfcc_mean = mfccs.mean(dim=1)
                    mfcc_std = mfccs.std(dim=1)
                    mfcc_iqr = iqr(mfccs.numpy(), axis=1)

                    embMFCC = np.concatenate((mfcc_mean.numpy(), mfcc_std.numpy(), mfcc_iqr), axis=0)

                    print(f'embMFCC shape: {embMFCC.shape}')
                    print(f'embMFCC type: {type(embMFCC)}')

                    if F is None:
                        F = []
                    F.append(np.array(embMFCC))
                    # print(f'Tipo de F dentro del for {type(F)}')

                    # BASIC SPECTRAL
                    basic_spectral = compare.process_signal(s, sr)

                    # Definir el subconjunto de características
                    feature_subset = ['pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                                                'pcm_fftMag_spectralRollOff25.0_sma',
                                                'pcm_fftMag_spectralRollOff50.0_sma',
                                                'pcm_fftMag_spectralRollOff75.0_sma',
                                                'pcm_fftMag_spectralRollOff90.0_sma',
                                                'pcm_fftMag_spectralFlux_sma',
                                                'pcm_fftMag_spectralCentroid_sma',
                                                'pcm_fftMag_spectralEntropy_sma',
                                                'pcm_fftMag_spectralVariance_sma',
                                                'pcm_fftMag_spectralSkewness_sma',
                                                'pcm_fftMag_spectralKurtosis_sma',
                                                'pcm_fftMag_spectralSlope_sma',
                                                'pcm_fftMag_psySharpness_sma',
                                                'pcm_fftMag_spectralHarmonicity_sma']

                    # Extraer el subconjunto de características y convertir a NumPy
                    basic_spectral = basic_spectral[feature_subset].to_numpy()
                    basic_spectral = np.nan_to_num(basic_spectral)
                    basic_spectral = torch.from_numpy(basic_spectral).T

                    # Mostrar la forma de las características
                    print(f'Características basic_spectral shape: {basic_spectral.shape}')

                    if self.args.get('extra_features', False):
                        deltas = torchaudio.functional.compute_deltas(basic_spectral)
                        delta_deltas = torchaudio.functional.compute_deltas(deltas)
                        basic_spectral = torch.cat((basic_spectral, deltas, delta_deltas), dim=0)

                    basic_spectral = basic_spectral.squeeze(0)

                    basic_spectral_mean = basic_spectral.mean(dim=1)
                    basic_spectral_std = basic_spectral.std(dim=1)
                    basic_spectral_iqr = iqr(basic_spectral.numpy(), axis=1)

                    embBasicSpectral = np.concatenate((basic_spectral_mean.numpy(), basic_spectral_std.numpy(), basic_spectral_iqr), axis=0)

                    if F2 is None:
                        F2 = []
                    F2.append(np.array(embBasicSpectral))

                except Exception as e:
                    print("Error processing audio. Error: {e}")

            if len(F) == 0 or len(F2) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)
            F2 = np.stack(F2)

            # concatenar horizontalmente los embeddings
            F = np.concatenate((F, F2), axis=1)
            return F

        if self.args['feature_type'] == 'bert_wav2vec':
            row = df.loc[df['root'] == audio_id]

            transcriptions_list = ast.literal_eval(row['transcriptions'].values[0])
            roots_list = ast.literal_eval(row['audio_roots'].values[0])

            for i, transcription in enumerate(transcriptions_list):
                try:
                    audio_root = roots_list[i]

                    # BERT embedding
                    inputs = tokenizer_bert(transcription, return_tensors='pt', truncation=True, padding=True)

                    # Pass through the BERT model to get the embeddings
                    with torch.no_grad():
                        outputs = model_bert(**inputs)

                    cls_embedding = outputs.last_hidden_state[:, 0, :]

                    if F is None:
                        F = []

                    F.append(np.array(cls_embedding.squeeze(0)))

                    # wav2vec embedding
                    audio_root = os.path.join(dir_bas, audio_root)
                    waveform, sr = torchaudio.load(audio_root)
                    if sr != 16000:
                        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

                    inputs = tokenizer_w2v(waveform.squeeze().numpy(), return_tensors="pt", padding="longest")
                    with torch.no_grad():
                        outputs = model_w2v(**inputs)

                    w2v = outputs.last_hidden_state
                    w2v = w2v.squeeze()

                    w2v_mean = w2v.mean(dim=0)
                    w2v_std = w2v.std(dim=0)
                    w2v_iqr = iqr(w2v.numpy(), axis=0)

                    w2v = np.concatenate((w2v_mean.numpy(), w2v_std.numpy(), w2v_iqr), axis=0)

                    if F2 is None:
                        F2 = []
                    F2.append(np.array(w2v))

                except Exception as e:
                    print("Error processing transcription. Error: {e}")

            if len(F) == 0 or len(F2) == 0:
                raise ValueError("No valid embeddings extracted.")

            F = np.stack(F)
            F2 = np.stack(F2)

            # concatenar horizontalmente los embeddings
            F = np.concatenate((F, F2), axis=1)
            return F

        if self.args['feature_type'] == 'MelSpec':
            F = self.feature_transform(s)

        if self.args['feature_type'] == 'logMelSpec':
            F = self.feature_transform(s)
            F = torchaudio.functional.amplitude_to_DB(F, multiplier=10, amin=1e-10, db_multiplier=0)

        if self.args['feature_type'] == 'MFCC':
            F = self.feature_transform(s)

        if 'ComParE_2016' in self.args['feature_type']:
            #
            s = s[None, :]
            F = self.feature_transform.process_signal(s, sr)

            # feature subsets
            feature_subset = {}
            if self.args['feature_type'] == 'ComParE_2016_voicing':
                feature_subset['subset'] = ['F0final_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
                                            'jitterDDP_sma', 'shimmerLocal_sma', 'logHNR_sma']

            if self.args['feature_type'] == 'ComParE_2016_energy':
                feature_subset['subset'] = ['audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
                                            'pcm_RMSenergy_sma', 'pcm_zcr_sma']

            if self.args['feature_type'] == 'ComParE_2016_spectral':
                feature_subset['subset'] = ['audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 'audSpec_Rfilt_sma[2]',
                                            'audSpec_Rfilt_sma[3]', 'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]',
                                            'audSpec_Rfilt_sma[6]', 'audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]',
                                            'audSpec_Rfilt_sma[9]', 'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]',
                                            'audSpec_Rfilt_sma[12]', 'audSpec_Rfilt_sma[13]', 'audSpec_Rfilt_sma[14]',
                                            'audSpec_Rfilt_sma[15]', 'audSpec_Rfilt_sma[16]', 'audSpec_Rfilt_sma[17]',
                                            'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]',
                                            'audSpec_Rfilt_sma[21]', 'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]',
                                            'audSpec_Rfilt_sma[24]', 'audSpec_Rfilt_sma[25]',
                                            'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                                            'pcm_fftMag_spectralRollOff25.0_sma', 'pcm_fftMag_spectralRollOff50.0_sma',
                                            'pcm_fftMag_spectralRollOff75.0_sma', 'pcm_fftMag_spectralRollOff90.0_sma',
                                            'pcm_fftMag_spectralFlux_sma',
                                            'pcm_fftMag_spectralCentroid_sma',
                                            'pcm_fftMag_spectralEntropy_sma',
                                            'pcm_fftMag_spectralVariance_sma',
                                            'pcm_fftMag_spectralSkewness_sma',
                                            'pcm_fftMag_spectralKurtosis_sma',
                                            'pcm_fftMag_spectralSlope_sma',
                                            'pcm_fftMag_psySharpness_sma',
                                            'pcm_fftMag_spectralHarmonicity_sma',
                                            'mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]', 'mfcc_sma[5]',
                                            'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]', 'mfcc_sma[9]', 'mfcc_sma[10]',
                                            'mfcc_sma[11]', 'mfcc_sma[12]', 'mfcc_sma[13]', 'mfcc_sma[14]']

            if self.args['feature_type'] == 'ComParE_2016_mfcc':
                feature_subset['subset'] = ['mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]', 'mfcc_sma[5]',
                                            'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]',
                                            'mfcc_sma[9]', 'mfcc_sma[10]', 'mfcc_sma[11]', 'mfcc_sma[12]',
                                            'mfcc_sma[13]', 'mfcc_sma[14]']

            if self.args['feature_type'] == 'ComParE_2016_rasta':
                feature_subset['subset'] = ['audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 'audSpec_Rfilt_sma[2]',
                                            'audSpec_Rfilt_sma[3]',
                                            'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]', 'audSpec_Rfilt_sma[6]',
                                            'audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]', 'audSpec_Rfilt_sma[9]',
                                            'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]', 'audSpec_Rfilt_sma[12]',
                                            'audSpec_Rfilt_sma[13]',
                                            'audSpec_Rfilt_sma[14]', 'audSpec_Rfilt_sma[15]', 'audSpec_Rfilt_sma[16]',
                                            'audSpec_Rfilt_sma[17]',
                                            'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]',
                                            'audSpec_Rfilt_sma[21]',
                                            'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]', 'audSpec_Rfilt_sma[24]',
                                            'audSpec_Rfilt_sma[25]']

            if self.args['feature_type'] == 'ComParE_2016_basic_spectral':
                feature_subset['subset'] = ['pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                                            'pcm_fftMag_spectralRollOff25.0_sma',
                                            'pcm_fftMag_spectralRollOff50.0_sma',
                                            'pcm_fftMag_spectralRollOff75.0_sma',
                                            'pcm_fftMag_spectralRollOff90.0_sma',
                                            'pcm_fftMag_spectralFlux_sma',
                                            'pcm_fftMag_spectralCentroid_sma',
                                            'pcm_fftMag_spectralEntropy_sma',
                                            'pcm_fftMag_spectralVariance_sma',
                                            'pcm_fftMag_spectralSkewness_sma',
                                            'pcm_fftMag_spectralKurtosis_sma',
                                            'pcm_fftMag_spectralSlope_sma',
                                            'pcm_fftMag_psySharpness_sma',
                                            'pcm_fftMag_spectralHarmonicity_sma']

            if self.args['feature_type'] == 'ComParE_2016_llds':
                feature_subset['subset'] = list(F.columns)

            F = F[feature_subset['subset']].to_numpy()
            F = np.nan_to_num(F)
            F = torch.from_numpy(F).T

        if 'Spafe_' in self.args['feature_type']:
            # Spafe feature selected
            nfft = int(float(self.args.get('window_size', 0) * 1e-3 * self.resampling_rate))

            if self.args['feature_type'] in ['Spafe_mfcc', 'Spafe_imfcc', 'Spafe_gfcc', 'Spafe_lfcc', 'Spafe_msrcc',
                                             'Spafe_ngcc', 'Spafe_psrcc']:
                F = self.feature_transform(s, sr,
                                           num_ceps=int(self.args.get('n_mfcc')),
                                           low_freq=int(self.args.get('f_min')),
                                           high_freq=int(sr // 2),
                                           nfilts=int(self.args.get('n_mels')),
                                           nfft=nfft,
                                           use_energy=self.args.get('use_energy') == 'True')
            elif self.args['feature_type'] in ['Spafe_pncc']:
                F = self.feature_transform(s, sr, num_ceps=int(self.args.get('n_mfcc')),
                                           low_freq=int(self.args.get('f_min')),
                                           high_freq=int(sr // 2),
                                           nfilts=int(self.args.get('n_mels')),
                                           nfft=nfft)

            elif self.args['feature_type'] in ['Spafe_cqcc']:
                F = self.feature_transform(s, sr,
                                           num_ceps=int(self.args.get('n_mfcc')),
                                           low_freq=int(self.args.get('f_min')),
                                           high_freq=int(sr // 2),
                                           nfft=nfft)
                F = torch.from_numpy(F).T
            elif self.args['feature_type'] in ['Spafe_lpc', 'Spafe_lpcc', ]:
                F = self.feature_transform(s, sr, order=int(self.args.get('plp_order')))
                if isinstance(F, tuple):
                    F = F[0]

            elif self.args['feature_type'] in ['Spafe_plp', 'Spafe_rplp']:
                F = self.feature_transform(s, sr,
                                           order=int(self.args.get('plp_order')),
                                           conversion_approach=self.args.get('conversion_approach'),
                                           low_freq=int(self.args.get('f_min')),
                                           high_freq=int(sr // 2),
                                           normalize=self.args.get('normalize'),
                                           nfilts=int(self.args.get('n_mels')),
                                           nfft=nfft)
            F = np.nan_to_num(F)
            F = torch.from_numpy(F).T

        if self.args.get('compute_deltas', False):
            FD = torchaudio.functional.compute_deltas(F)
            F = torch.cat((F, FD), dim=0)

            if self.args.get('compute_deltas_deltas', False):
                FDD = torchaudio.functional.compute_deltas(FD)
                F = torch.cat((F, FDD), dim=0)

        if self.args.get('apply_mean_norm', False):
            F = F - torch.mean(F, dim=0)

        if self.args.get('apply_var_norm', False):
            F = F / torch.std(F, dim=0)

        # own feature selection
        if self.args.get('extra_features', False) and 'ComParE_2016' not in self.args['feature_type']:
            s = s[None, :]
            # Config OpenSMILE
            feature_subset = {'subset': [
                # Voicing
                'F0final_sma', 'voicingFinalUnclipped_sma',
                'jitterLocal_sma', 'jitterDDP_sma',
                'shimmerLocal_sma',
                'logHNR_sma',
                # Energy
                'audspec_lengthL1norm_sma',
                'audspecRasta_lengthL1norm_sma',
                'pcm_RMSenergy_sma',
                'pcm_zcr_sma',
                # Spectral
                'pcm_fftMag_fband250-650_sma',
                'pcm_fftMag_fband1000-4000_sma',
                'pcm_fftMag_spectralRollOff25.0_sma',
                'pcm_fftMag_spectralRollOff50.0_sma',
                'pcm_fftMag_spectralRollOff75.0_sma',
                'pcm_fftMag_spectralRollOff90.0_sma',
                'pcm_fftMag_spectralFlux_sma',
                'pcm_fftMag_spectralCentroid_sma',
                'pcm_fftMag_spectralEntropy_sma',
                'pcm_fftMag_spectralVariance_sma',
                'pcm_fftMag_spectralSkewness_sma',
                'pcm_fftMag_spectralKurtosis_sma',
                'pcm_fftMag_spectralSlope_sma',
                'pcm_fftMag_psySharpness_sma',
                'pcm_fftMag_spectralHarmonicity_sma'
            ]}
            extra_transform = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                              feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                              sampling_rate=self.resampling_rate)
            # Extract features
            F_extra = extra_transform.process_signal(s, sr)
            F_extra = F_extra[feature_subset['subset']].to_numpy()
            F_extra = np.nan_to_num(F_extra)
            F_extra = torch.from_numpy(F_extra).T
            # Concatenate the features
            common_shape = min(F.shape[1], F_extra.shape[1])
            F = torch.cat((F[:, :common_shape], F_extra[:, :common_shape]), dim=0)

        return F.T

    def extract(self, filepath, audio_id):
        """
        Extracts the features from the audio file
        :param filepath: path to the audio file
        :return: features
        """
        if not isinstance(filepath, str):
            return np.NAN
        else:
            self.audio_path = filepath
            s, fs = self._read_audio(filepath)
            return self._do_feature_extraction(s, fs, audio_id)
