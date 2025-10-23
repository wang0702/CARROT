#!/usr/bin/env python3
"""
Build FAISS index for WikiPassageQA using BGE-M3 dense embeddings.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import faiss
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from FlagEmbedding import BGEM3FlagModel


class FAISSIndexBuilder:
    def __init__(
        self,
        model_name: str = 'BAAI/bge-m3',
        chunk_size: int = 256,
        batch_size: int = 32,
        use_fp16: bool = True
    ):
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        print(f"Loading BGE-M3 model: {model_name}")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

        # Get embedding dimension from model
        test_output = self.model.encode(
            ["test"],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        self.embed_dim = test_output['dense_vecs'].shape[-1]
        print(f"Dense embedding dimension: {self.embed_dim}")

    def chunk_text(self, text: str, doc_id: str, passage_id: str) -> List[Dict]:
        chunks = []
        words = text.split()
        overlap = self.chunk_size // 4

        if len(words) <= self.chunk_size:
            chunks.append({
                'text': text,
                'doc_id': doc_id,
                'passage_id': passage_id,
                'start_word': 0,
                'end_word': len(words)
            })
        else:
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_text = ' '.join(words[start:end])
                chunks.append({
                    'text': chunk_text,
                    'doc_id': doc_id,
                    'passage_id': passage_id,
                    'start_word': start,
                    'end_word': end
                })
                if end >= len(words):
                    break
                start += max(1, self.chunk_size - overlap)

        return chunks

    def encode_chunks_dense(self, chunks: List[str]) -> np.ndarray:
        all_embeddings = []

        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Encoding chunks"):
            batch = chunks[i:i + self.batch_size]

            outputs = self.model.encode(
                batch,
                batch_size=self.batch_size,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )

            all_embeddings.append(outputs['dense_vecs'])

        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings).astype('float32')
        else:
            all_embeddings = np.zeros((0, self.embed_dim), dtype='float32')

        return all_embeddings

    def build_index(
        self,
        passages_file: Path,
        output_dir: Path,
        max_chunks: int = None
    ):
        print(f"Loading passages from {passages_file}")
        with open(passages_file, 'r', encoding='utf-8') as f:
            passages_dict = json.load(f)

        print("Preparing text chunks...")
        all_chunks = []
        chunk_metadata = []
        chunk_id = 0

        for doc_id, passages in tqdm(passages_dict.items(), desc="Chunking documents"):
            for passage_id, text in passages.items():
                if not text or not text.strip():
                    continue

                chunks = self.chunk_text(text, doc_id, passage_id)
                for chunk_data in chunks:
                    all_chunks.append(chunk_data['text'])
                    chunk_metadata.append({
                        'chunk_id': chunk_id,
                        'doc_id': chunk_data['doc_id'],
                        'passage_id': chunk_data['passage_id'],
                        'text': chunk_data['text'],
                        'start_word': chunk_data['start_word'],
                        'end_word': chunk_data['end_word']
                    })
                    chunk_id += 1

                    if max_chunks and chunk_id >= max_chunks:
                        break
            if max_chunks and chunk_id >= max_chunks:
                break

        print(f"Total chunks prepared: {len(all_chunks)}")

        print("Encoding chunks with BGE-M3 dense embeddings...")
        embeddings = self.encode_chunks_dense(all_chunks)

        print(f"Embeddings shape: {embeddings.shape}")

        faiss.normalize_L2(embeddings)

        print("Building FAISS index...")
        index = faiss.IndexFlatIP(self.embed_dim)
        index.add(embeddings)

        output_dir.mkdir(parents=True, exist_ok=True)

        index_path = output_dir / 'faiss_index.bin'
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to {index_path}")

        metadata_path = output_dir / 'chunk_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunk_metadata, f)
        print(f"Saved chunk metadata to {metadata_path}")

        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'model': 'BAAI/bge-m3',
                'embedding_type': 'dense',
                'chunk_size': self.chunk_size,
                'num_chunks': len(all_chunks),
                'num_documents': len(passages_dict),
                'embed_dim': self.embed_dim,
                'index_type': 'IndexFlatIP (cosine similarity)'
            }, f, indent=2)
        print(f"Saved configuration to {config_path}")

        return index, chunk_metadata


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index for WikiPassageQA')
    parser.add_argument('--chunk-size', type=int, default=256,
                        help='Chunk size in words (default: 256)')
    parser.add_argument('--max-chunks', type=int, default=None,
                        help='Maximum chunks to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for encoding (default: 32)')
    args = parser.parse_args()

    passages_file = PROJECT_ROOT / 'data' / 'wikipassage' / 'document_passages.json'
    output_dir = PROJECT_ROOT / 'data' / 'wikipassage' / 'faiss_index'

    if not passages_file.exists():
        print(f"Error: Passages file not found at {passages_file}")
        print("Please ensure document_passages.json exists in data/wikipassage/")
        return 1

    print("=" * 80)
    print("Building FAISS Index for WikiPassageQA (BGE-M3 Dense Embeddings)")
    print("=" * 80)
    print(f"Chunk size: {args.chunk_size}")
    print(f"Batch size: {args.batch_size}")
    if args.max_chunks:
        print(f"Max chunks: {args.max_chunks} (testing mode)")
    print("=" * 80)

    builder = FAISSIndexBuilder(
        model_name='BAAI/bge-m3',
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        use_fp16=True
    )

    try:
        index, metadata = builder.build_index(
            passages_file=passages_file,
            output_dir=output_dir,
            max_chunks=args.max_chunks
        )

        print("\n" + "=" * 80)
        print("FAISS Index Build Complete!")
        print("=" * 80)
        print(f"Total chunks indexed: {len(metadata)}")
        print(f"Index size: {index.ntotal}")
        print(f"Index location: {output_dir}")
        print("\nFiles created:")
        print("  - faiss_index.bin (main FAISS index)")
        print("  - chunk_metadata.pkl (chunk information)")
        print("  - config.json (index configuration)")
        print("\nThis index uses BGE-M3 dense embeddings for efficient retrieval.")

        return 0

    except Exception as e:
        print(f"Error building index: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
