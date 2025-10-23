import torch
import numpy as np
from torch.utils.data import Dataset


class EmbeddingPairsDatasetEnhanced(Dataset):
    def __init__(self, df, query_embed_column, chunk_embed_column,
                 label_column, param_df, label_encoder, use_chunk=True):
        self.query_embeddings = np.stack(df[query_embed_column].values)
        self.use_chunk = use_chunk

        if use_chunk and chunk_embed_column in df.columns:
            self.chunk_embeddings = np.stack(df[chunk_embed_column].values)
        else:
            self.chunk_embeddings = None
            self.use_chunk = False

        self.labels = df[label_column].values
        self.param_df = param_df.loc[df.index].reset_index(drop=True)
        self.label_encoder = label_encoder
        self.encoded_labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.query_embeddings)

    def __getitem__(self, idx):
        """
        返回对比学习样本对

        Returns:
            如果 use_chunk=True:
                (query1, chunk1, query2, chunk2, label, class_label1, class_label2, params1, params2)
            如果 use_chunk=False:
                (query1, query2, label, class_label1, class_label2, params1, params2)
        """
        if np.random.random() > 0.5:  # Positive pair (同一类 reranker)
            label = 0
            positive_indices = np.where(self.encoded_labels == self.encoded_labels[idx])[0]
            positive_indices = positive_indices[positive_indices != idx]

            if len(positive_indices) == 0:
                positive_idx = idx
            else:
                positive_idx = np.random.choice(positive_indices)

            pair_idx = positive_idx
        else:  # Negative pair (不同类 reranker)
            label = 1
            negative_indices = np.where(self.encoded_labels != self.encoded_labels[idx])[0]

            if len(negative_indices) == 0:
                negative_idx = idx
            else:
                negative_idx = np.random.choice(negative_indices)

            pair_idx = negative_idx

        if self.use_chunk:
            return (
                self.query_embeddings[idx],
                self.chunk_embeddings[idx],
                self.query_embeddings[pair_idx],
                self.chunk_embeddings[pair_idx],
                label,
                self.encoded_labels[idx],
                self.encoded_labels[pair_idx],
                self.param_df.iloc[idx].values,
                self.param_df.iloc[pair_idx].values
            )
        else:
            return (
                self.query_embeddings[idx],
                self.query_embeddings[pair_idx],
                label,
                self.encoded_labels[idx],
                self.encoded_labels[pair_idx],
                self.param_df.iloc[idx].values,
                self.param_df.iloc[pair_idx].values
            )



class EmbeddingPairsDataset(Dataset):
    def __init__(self, df, embed_column, label_column, param_df, label_encoder):
        self.embeddings = np.stack(df[embed_column].values)
        self.labels = df[label_column].values
        self.param_df = param_df
        self.label_encoder = label_encoder
        self.param_df = self.param_df.loc[df.index].reset_index(drop=True)
        self.encoded_labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if np.random.random() > 0.5:
            label = 0
            positive_indices = np.where(self.encoded_labels == self.encoded_labels[idx])[0]
            positive_indices = positive_indices[positive_indices != idx]
            if len(positive_indices) == 0:
                positive_idx = idx
            else:
                positive_idx = np.random.choice(positive_indices)
            return (self.embeddings[idx], self.embeddings[positive_idx], label,
                    self.encoded_labels[idx], self.encoded_labels[positive_idx],
                    self.param_df.iloc[idx].values, self.param_df.iloc[positive_idx].values)
        else:
            label = 1
            negative_indices = np.where(self.encoded_labels != self.encoded_labels[idx])[0]
            if len(negative_indices) == 0:
                negative_idx = idx
            else:
                negative_idx = np.random.choice(negative_indices)
            return (self.embeddings[idx], self.embeddings[negative_idx], label,
                    self.encoded_labels[idx], self.encoded_labels[negative_idx],
                    self.param_df.iloc[idx].values, self.param_df.iloc[negative_idx].values)
