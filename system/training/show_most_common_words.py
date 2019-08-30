import torch
import numpy as np
import pandas as pd
import os
from fire import Fire
from run import parse_poolingfunction
from tabulate import tabulate
from scipy.signal import find_peaks
import kaldi_io
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_transcripts(root: str, speakers: list, filterlen: int):
    transcripts = {}
    for speaker in speakers:
        # PRocess transcript first to get start_end
        transcript_file = glob.glob(
            os.path.join(root, str(speaker)) + '*TRANSCRIPT.csv')[0]
        transcript_df = pd.read_csv(transcript_file, sep='\t')
        transcript_df.dropna(inplace=True)
        transcript_df.value = transcript_df.value.str.strip()
        # Subset for
        transcript_df = transcript_df[
            transcript_df.value.str.split().apply(len) > filterlen]
        transcript_df = transcript_df[
            transcript_df.speaker ==
            'Participant'].reset_index().loc[:, 'value'].to_dict()
        transcripts[speaker] = transcript_df
    return transcripts


def forward_model(model, x, pooling):
    model = model.eval()
    model.to(device)
    if 'LSTMAttn' in model.__class__.__name__:
        x, _ = model.lstm.net(x)
        x, attention_weights = model.attn(x)
        return model.lstm.outputlayer(x), attention_weights
    elif 'GRUAttn' in model.__class__.__name__:
        x, _ = model.net(x)
        x, attention_weights = model.attn(x)
        return model.outputlayer(x), attention_weights
    else:
        outputs = model(x)
        return pooling(outputs, 1), outputs[:, :, 0]


def show_most_common(experiment_path: str,
                     max_heigth: float = 0.8,
                     filterlen: int = 0,
                     shift:int = 0):

    config = torch.load(glob.glob("{}/run_config*".format(experiment_path))[0],
                        map_location=lambda storage, loc: storage)
    model = torch.load(glob.glob("{}/run_model*".format(experiment_path))[0],
                       map_location=lambda storage, loc: storage)
    scaler = torch.load(glob.glob("{}/run_scaler*".format(experiment_path))[0],
                        map_location=lambda storage, loc: storage)
    data = config['devfeatures']
    dev_labels = config['devlabels']
    dev_labels_binary = pd.read_csv(dev_labels)
    poolingfunction = parse_poolingfunction(config['poolingfunction'])
    TRANSCRIPT_ROOT = '../data_preprocess/labels_processed/'
    transcripts = read_transcripts(TRANSCRIPT_ROOT,
                                   dev_labels_binary['Participant_ID'].values,
                                   filterlen)
    #
    all_words = []
    with torch.no_grad():
        for k, v in kaldi_io.read_mat_ark(data):
            v = scaler.transform(v)
            cur_transcripts = transcripts[int(k)]
            v = np.roll(v, shift, axis=0) # Shifting by n sentences
            v = torch.from_numpy(v).unsqueeze(0).float().to(device)
            output, weights = forward_model(model, v, poolingfunction)
            output, weights = output.squeeze().cpu(), weights.squeeze().cpu(
            ).numpy()
            model_thinks_depression = float(torch.sigmoid(output[1]).numpy())
            peaks = find_peaks(weights, height=np.max(weights) * max_heigth)[0]
            assert len(cur_transcripts) == len(
                weights), "Trans: {} Weight: {}".format(
                    len(cur_transcripts), len(weights))
            for peak in peaks:
                all_words.append({
                    'sent':
                    cur_transcripts[peak],
                    'label':
                    bool(dev_labels_binary[dev_labels_binary['Participant_ID']
                                           == int(k)].PHQ8_Binary.values[0]),
                    'predprob':
                    model_thinks_depression,
                    'pred':
                    model_thinks_depression > 0.5
                })

    df = pd.DataFrame(all_words)
    # aggregate = df.groupby('sent', as_index=False).agg(
    # ['count', 'mean'])
    # print(aggregate.head())
    # print(aggregate.pooledprob.head())
    # print(aggregate.columns)
    max_counts = df.sent.value_counts().to_frame().head(10)
    print(tabulate(max_counts))


if __name__ == "__main__":
    Fire(show_most_common)
