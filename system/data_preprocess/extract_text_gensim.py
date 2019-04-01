from gensim.models import KeyedVectors
import argparse
import numpy as np
import os
import pandas as pd
from glob import glob
import kaldi_io
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize


parser = argparse.ArgumentParser()
parser.add_argument(
    'transcriptfile', default='labels_processed/all_transcripts.csv', type=str, nargs="?", help="All transcriptions (cleaned)")
parser.add_argument('--transcriptdir', type=str, default='labels_processed')
parser.add_argument('-w', type=int, default=4, help="Worker count")
parser.add_argument('--filterlen', default=0, type=int)
parser.add_argument('--filterby', type=str, default='Participant')
parser.add_argument('--from_pretrained', default=None, type=str)
args = parser.parse_args()

if not args.from_pretrained:
    # Pretraining of Doc2Vec model
    df = pd.read_csv(args.transcriptfile, sep='\t')
    # Filter length ( remove uh yeah um .... )
    df = df[df.value.str.split().apply(len) > args.filterlen]
    all_text = df.value.values

    documents = [TaggedDocument(words=word_tokenize(pargraph.lower()), tags=[
                                i]) for i, pargraph in enumerate(all_text)]
    model = Doc2Vec(documents, vector_size=100, window=3,
                    min_count=3, workers=args.w, epochs=50, sample=1e-4, seed=1)

else:
    model = KeyedVectors.load_word2vec_format(
        args.from_pretrained, binary=True)

pretrained_flag = "_pretrained" if args.from_pretrained else ""
subsetfiles = ['labels/train_split_Depression_AVEC2017.csv',
               'labels/dev_split_Depression_AVEC2017.csv']
output_files = ['features/text/train_text{}_100dim_filter{}.ark'.format(pretrained_flag, args.filterlen),
                'features/text/dev_text{}_100dim_filter{}.ark'.format(pretrained_flag, args.filterlen)]
for subsetfile, outputfile in zip(subsetfiles, output_files):
    # Extracting features for the Participant IDs
    subset_df = pd.read_csv(subsetfile)
    speakers = subset_df['Participant_ID'].values

    with open(outputfile, 'wb') as fd:
        for speaker in tqdm(speakers):
            # PRocess transcript first to get start_end
            transcript_file = glob(os.path.join(
                args.transcriptdir, str(speaker)) + '*TRANSCRIPT.csv')[0]
            transcript_df = pd.read_csv(transcript_file, sep='\t')
            # Filter for participant only
            if args.filterby:
                transcript_df = transcript_df[transcript_df.speaker ==
                                              args.filterby]
            transcript_df = transcript_df[transcript_df.value.str.split().apply(
                len) > args.filterlen]
            if args.from_pretrained:
                features = []
                for paragraph in transcript_df.value.values:
                    for word in paragraph.split(' '):
                        if word in model:
                            vec = model[word.lower()] 
                        else:
                            vec = np.zeros(300)
                        features.append(vec)
                features = np.array(features)
            else:
                features = np.array([model.infer_vector(
                    paragraph.lower()) for paragraph in transcript_df.value.values])

            kaldi_io.write_mat(fd, features, key=str(speaker))
