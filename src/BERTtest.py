import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import ast  # Para convertir el texto de la lista a una lista en Python

# Step 1: Load the CSV file
csv_file = "C:/Users/alvar/PycharmProjects/DementiaBank/ADReSS_db.csv"

# Read the CSV file
df = pd.read_csv(csv_file)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Function to generate BERT embeddings from text
def get_bert_embedding(text):
    # Tokenize the text (convert to input format for BERT)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Pass through the BERT model to get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # The last hidden state contains the embeddings for each token in the input
    # To get the embedding of the [CLS] token, which is at index 0
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding


# Step 2: Process the first 20 rows and get BERT embeddings for each fragment
df_subset = df.head(20)

for index, row in df_subset.iterrows():
    # Convert the string representation of the list to an actual list
    transcriptions_list = ast.literal_eval(row['transcriptions'])

    # Process each fragment in the list
    for i, transcription in enumerate(transcriptions_list):
        try:
            # Get BERT embedding for the transcription fragment
            embedding = get_bert_embedding(transcription)

            # Print the shape of the embedding
            print(f"Audio {index + 1}, Fragment {i + 1}: Shape of the embedding: ", embedding.shape)

        except Exception as e:
            print(f"Audio {index + 1}, Fragment {i + 1}: Error processing transcription. Error: {e}")
