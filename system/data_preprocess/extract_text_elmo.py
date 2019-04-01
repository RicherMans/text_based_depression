import argparse
import numpy as np
import os
import pandas as pd
from glob import glob
import kaldi_io
from tqdm import tqdm
from allennlp.commands.elmo import ElmoEmbedder


parser = argparse.ArgumentParser()
parser.add_argument(
    'transcriptfile', default='labels_processed/all_transcripts.csv', type=str, nargs="?", help="All transcriptions (cleaned)")
parser.add_argument(
    '--subsetfile', default='labels/train_split_Depression_AVEC2017.csv', type=str)
parser.add_argument('--transcriptdir', type=str, default='labels_processed')
parser.add_argument('-ip', type=str, default=None)
parser.add_argument('-o', '--output', type=str,
                    default='train_elmo.ark', help='feature output')
parser.add_argument('-w', type=int, default=4, help="Worker count")
parser.add_argument('--filterlen', default=0, type=int)
parser.add_argument('--filterby', type=str, default='Participant')
args = parser.parse_args()

elmo = ElmoEmbedder()

# Extracting features for the Participant IDs
subset_df = pd.read_csv(args.subsetfile)
speakers = subset_df['Participant_ID'].values

with open(args.output, 'wb') as fd:
    for speaker in tqdm(speakers):
        # PRocess transcript first to get start_end
        transcript_file = glob(os.path.join(
            args.transcriptdir, str(speaker)) + '*TRANSCRIPT.csv')[0]
        transcript_df = pd.read_csv(transcript_file, sep='\t')
        transcript_df.value = transcript_df.value.str.strip()
        transcript_df.dropna(inplace=True)
        transcript_df = transcript_df[transcript_df.value.str.split().apply(
            len) > args.filterlen]
        # Filter for participant only
        if args.filterby:
            transcript_df = transcript_df[transcript_df.speaker ==
                                          args.filterby]
        features = elmo.embed_sentence(
            transcript_df.value.values.tolist()).mean(0)  # Mean over the 3 elmo layers

        kaldi_io.write_mat(fd, features, key=str(speaker))
