import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from FlagEmbedding import BGEM3FlagModel
import joblib
from agents.network_enhanced import (
    SiameseNetwork, SiameseNetworkEnhanced,
    ContrastiveAndRegressionLoss, ContrastiveAndRegressionLossEnhanced
)
from agents.data_enhanced import EmbeddingPairsDatasetEnhanced


class SiameseNetworkTrainer:
    def __init__(self, df=None, embed_column=None, label_column=None, param_df=None, embed_dim=1024, margin=1.0,
                 lr=0.001, batch_size=32, num_epochs=60, device='cuda', embedding_model_name='BAAI/bge-m3',
                 use_fp16=False, chunk_embed_column=None, use_chunk_fusion=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_chunk_fusion = use_chunk_fusion
        print(f'Using device: {self.device}')
        print(f'Use chunk fusion: {use_chunk_fusion}')

        self.num_classes = 6
        self.model = SiameseNetworkEnhanced(embed_dim=embed_dim, num_classes=self.num_classes,
                                    use_chunk_fusion=use_chunk_fusion).to(self.device)
        self.criterion = ContrastiveAndRegressionLossEnhanced(margin=margin)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        if df is not None and embed_column is not None and label_column is not None and param_df is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(df[label_column])
            self.dataset = EmbeddingPairsDatasetEnhanced(
                df=df,
                query_embed_column=embed_column,
                chunk_embed_column=chunk_embed_column,
                label_column=label_column,
                param_df=param_df,
                label_encoder=self.label_encoder,
                use_chunk=use_chunk_fusion
            )
            train_size = int(0.7 * len(self.dataset))
            val_size = int(0.15 * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset,
                                                                                   [train_size, val_size, test_size])

            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

            joblib.dump(self.label_encoder, 'label_encoder.pkl')
        else:
            self.label_encoder = joblib.load('label_encoder.pkl')

        self.jinav2_class_index = self.label_encoder.transform(['jina-reranker-v2-base-multilingual'])[0]

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = BGEM3FlagModel(embedding_model_name, use_fp16=use_fp16)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for i, data in enumerate(self.train_loader):
                try:
                    if self.use_chunk_fusion:
                        # query1, chunk1, query2, chunk2, label, class_label1, class_label2, params1, params2
                        query1, chunk1, query2, chunk2, label, class_label1, class_label2, param_label1, param_label2 = data
                        query1 = query1.float().to(self.device)
                        chunk1 = chunk1.float().to(self.device)
                        query2 = query2.float().to(self.device)
                        chunk2 = chunk2.float().to(self.device)
                        label = label.float().to(self.device)
                        class_label1 = class_label1.long().to(self.device)
                        class_label2 = class_label2.long().to(self.device)
                        param_label1 = torch.tensor(param_label1).float().to(self.device)
                        param_label2 = torch.tensor(param_label2).float().to(self.device)

                        output1, param_output1 = self.model(query1, chunk1)
                        output2, param_output2 = self.model(query2, chunk2)
                    else:
                        # 原版: query1, query2, label, class_label1, class_label2, params1, params2
                        anchor, other, label, class_label1, class_label2, param_label1, param_label2 = data
                        anchor, other, label = anchor.float().to(self.device), other.float().to(
                            self.device), label.float().to(self.device)
                        class_label1, class_label2 = class_label1.long().to(self.device), class_label2.long().to(
                            self.device)
                        param_label1, param_label2 = torch.tensor(param_label1).float().to(self.device), torch.tensor(
                            param_label2).float().to(self.device)

                        output1, param_output1 = self.model(anchor)
                        output2, param_output2 = self.model(other)

                    class_output1, class_output2 = output1, output2
                    loss = self.criterion(output1, output2, label, class_output1, class_output2, class_label1,
                                          class_label2, param_output1, param_output2, param_label1, param_label2)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    continue

            avg_train_loss = running_loss / len(self.train_loader)
            avg_val_loss = self.validate()
            print(
                f'Epoch [{epoch + 1}/{self.num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        torch.save(self.model.state_dict(), 'siamese_model.pth')
        print("Model saved to siamese_model.pth")

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                try:
                    if self.use_chunk_fusion:
                        query1, chunk1, query2, chunk2, label, class_label1, class_label2, param_label1, param_label2 = data
                        query1 = query1.float().to(self.device)
                        chunk1 = chunk1.float().to(self.device)
                        query2 = query2.float().to(self.device)
                        chunk2 = chunk2.float().to(self.device)
                        label = label.float().to(self.device)
                        class_label1 = class_label1.long().to(self.device)
                        class_label2 = class_label2.long().to(self.device)
                        param_label1 = torch.tensor(param_label1).float().to(self.device)
                        param_label2 = torch.tensor(param_label2).float().to(self.device)

                        output1, param_output1 = self.model(query1, chunk1)
                        output2, param_output2 = self.model(query2, chunk2)
                    else:
                        anchor, other, label, class_label1, class_label2, param_label1, param_label2 = data
                        anchor, other, label = anchor.float().to(self.device), other.float().to(
                            self.device), label.float().to(self.device)
                        class_label1, class_label2 = class_label1.long().to(self.device), class_label2.long().to(
                            self.device)
                        param_label1, param_label2 = torch.tensor(param_label1).float().to(self.device), torch.tensor(
                            param_label2).float().to(self.device)

                        output1, param_output1 = self.model(anchor)
                        output2, param_output2 = self.model(other)

                    loss = self.criterion(output1, output2, label, output1, output2, class_label1, class_label2,
                                          param_output1, param_output2, param_label1, param_label2)
                    val_loss += loss.item()
                except Exception as e:
                    print(f"Error in validation batch {i}: {e}")
                    continue
        return val_loss / len(self.val_loader)

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                anchor, other, label, class_label1, class_label2, param_label1, param_label2 = data
                anchor, other, label = anchor.to(self.device).float(), other.to(self.device).float(), label.to(
                    self.device).float()

                output1, param_output1 = self.model(anchor)
                output2, param_output2 = self.model(other)

                euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                predictions = (euclidean_distance < 1.0).float()
                correct += (predictions == label).sum().item()
                total += label.size(0)

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy

    def generate_embedding(self, text):
        embedding = self.embedding_model.encode(text, batch_size=12, max_length=8192)['dense_vecs']
        return embedding.tolist()

    def load_model(self, model_path, use_chunk_fusion=None):
        checkpoint = torch.load(model_path, map_location=self.device)

        if use_chunk_fusion is None:
            if 'fc1.weight' in checkpoint:
                fc1_shape = checkpoint['fc1.weight'].shape
                use_chunk_fusion = (fc1_shape[1] == 2048)
                is_legacy_model = True
                print(f"Auto-detected legacy model: {'Fusion' if use_chunk_fusion else 'Query-only'} (fc1 input: {fc1_shape[1]})")
            elif 'query_encoder.0.weight' in checkpoint:
                if 'chunk_encoder.0.weight' in checkpoint:
                    use_chunk_fusion = True
                else:
                    use_chunk_fusion = False
                is_legacy_model = False
                print(f"Auto-detected enhanced model: {'Fusion' if use_chunk_fusion else 'Query-only'}")
            else:
                raise ValueError("Unknown model format")
        else:
            is_legacy_model = 'fc1.weight' in checkpoint

        self.use_chunk_fusion = use_chunk_fusion

        if is_legacy_model:
            self.model = SiameseNetwork(embed_dim=1024, num_classes=self.num_classes).to(self.device)
        else:
            self.model = SiameseNetworkEnhanced(embed_dim=1024, num_classes=self.num_classes,
                                                use_chunk_fusion=use_chunk_fusion).to(self.device)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, query, chunk_text=None):
        query_embedding = self.generate_embedding(query)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).to(self.device)
        query_tensor = query_tensor.unsqueeze(0)

        if self.use_chunk_fusion:
            if chunk_text is not None:
                chunk_embedding = self.generate_embedding(chunk_text)
            else:
                chunk_embedding = [0.0] * 1024
            chunk_tensor = torch.tensor(chunk_embedding, dtype=torch.float32).to(self.device)
            chunk_tensor = chunk_tensor.unsqueeze(0)
        else:
            chunk_tensor = None

        with torch.no_grad():
            if self.use_chunk_fusion:
                class_output, param_output = self.model(query_tensor, chunk_tensor)
            else:
                class_output, param_output = self.model(query_tensor)

            _, predicted = torch.max(class_output, dim=1)

        predicted_label = self.label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
        predicted_params = param_output.cpu().numpy()[0]
        return predicted_label, predicted_params
