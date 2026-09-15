# Using Google's USE to extract sentence embeddings

from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import csv
from sklearn.metrics.pairwise import cosine_similarity

# Hardcoded parameters
PARENTDIR = "/Users/jadynpark/Library/CloudStorage/Box-Box/Projects/CoupleSync/data"
CONVODIR = "4_audio/4_transcript_clean"
RECALLDIR = "5_recall/2_clean"
COUPLEID = range(1001, 1052)
CONDITION = "conflict" # "conflict", "originstory"

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

print ("module %s loaded" % module_url)

def embed(input):
  return model(input)

# Load recall text
recallDf = pd.read_csv(os.path.join(PARENTDIR, RECALLDIR, "recall_compiled.csv"))

# Compute for each couple and subject
similarityDf = []
for couple in COUPLEID:

    thisCoupleId = str(couple)
    member1 = thisCoupleId+'1'
    member2 = thisCoupleId+'2'

    print(f"Processing couple {thisCoupleId}")

    # == Conversation transcripts == 
    # Load conversation
    convoDir = os.path.join(PARENTDIR, CONVODIR, thisCoupleId, f"{thisCoupleId}_{CONDITION}_clean.csv")
    if not os.path.exists(convoDir):
        pass
    else:
        transcript = pd.read_csv(convoDir)
        transcriptText = transcript["text"]

        # Create an embedding for each sentence
        convoEmbeddings = []
        for i in range(0, len(transcriptText)):
            text = transcriptText.loc[i]
            embedding = embed([text])
            convoEmbeddings.append(embedding)
        
        convoEmbeddings = np.vstack(convoEmbeddings)

        print(f"Created conversation embeddings. Dim: {convoEmbeddings.shape}")

        # Average across sentences
        convoEmbeddingsMean = np.mean(convoEmbeddings,0)

        # == Recall text data ==
        # Extract recall transcript
        m1Recall = recallDf.loc[recallDf['id'] == int(member1)]
        m2Recall = recallDf.loc[recallDf['id'] == int(member2)]

        if CONDITION == "conflict":
            rowName = "conflict_memory"
        else:
            rowName = "originstory_memory"

        m1RecallText = m1Recall[rowName].iloc[0] if not m1Recall.empty else None
        m2RecallText = m2Recall[rowName].iloc[0] if not m2Recall.empty else None

        # Split the text into sentences to avoid going over token limit
        if pd.isna(m1RecallText):
            m1Sentences = "NaN"
            m2Sentences = m2RecallText.split('.')
        elif pd.isna(m2RecallText):
            m1Sentences = m1RecallText.split('.')
            m2Sentences = "NaN"
        else:
            m1Sentences = m1RecallText.split('.')
            m2Sentences = m2RecallText.split('.')

        # Create an embedding for each sentence
        m1Embeddings = []
        for i in range(0, (len(m1Sentences))):
            m1Text = m1Sentences[i]
            m1Embedding = embed([m1Text])
            m1Embeddings.append(m1Embedding)
        m1Embeddings = np.vstack(m1Embeddings)

        print(f"Created {member1} embeddings. Dim: {m1Embeddings.shape}")

        # Average across sentences
        m1EmbeddingsMean = np.mean(m1Embeddings,0)

        m2Embeddings = []
        for i in range(0, (len(m2Sentences))):
            m2Text = m2Sentences[i]
            m2Embedding = embed([m2Text])
            m2Embeddings.append(m2Embedding)
        m2Embeddings = np.vstack(m2Embeddings)

        print(f"Created {member2} embeddings. Dim: {m2Embeddings.shape}")

        # Average across sentences
        m2EmbeddingsMean = np.mean(m2Embeddings,0)

        # Calculate conversation-recall similarity
        convoM1Sim = cosine_similarity([convoEmbeddingsMean], [m1EmbeddingsMean])
        convoM2Sim = cosine_similarity([convoEmbeddingsMean], [m2EmbeddingsMean])

        print(f"Convo-M1 similarity: {round(convoM1Sim[0][0], 2)}; Convo-M2 similarity: {round(convoM2Sim[0][0],2)}")

        # Calculate m1recall-m2recall similarity
        recallSim = cosine_similarity([m1EmbeddingsMean], [m2EmbeddingsMean])

        print(f"Recall-recall similarity: {round(recallSim[0][0],2)}")

        similarityDf.append({
            "couple_id": thisCoupleId,
            "id": member1,
            "convo_similarity": float(convoM1Sim[0][0]),
            "recall_similarity": float(recallSim[0][0])
        })

        similarityDf.append({
            "couple_id": thisCoupleId,
            "id": member2,
            "convo_similarity": float(convoM2Sim[0][0]),
            "recall_similarity": float(recallSim[0][0])
        })

# Write results to CSV
SAVEPATH = os.path.join(PARENTDIR, "5_recall")
os.makedirs(SAVEPATH, exist_ok=True)

output_file = os.path.join(SAVEPATH, "3_similarity", f"use_similarity_{CONDITION}_clean.csv")

# Dynamically rename columns before saving
if CONDITION == "conflict":
    convo_col = "convo_similarity_conf"
    recall_col = "recall_similarity_conf"
else:
    convo_col = "convo_similarity_os"
    recall_col = "recall_similarity_os"

# Create DataFrame
df = pd.DataFrame(similarityDf)

# Rename columns
df = df.rename(columns={
    "convo_similarity": convo_col,
    "recall_similarity": recall_col
})

# Save to CSV
df.to_csv(output_file, index=False)
print(f"Saved similarity results to {output_file}")




