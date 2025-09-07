from typing import Dict

import random

import torch

from data.base import BaseDataset


class SCIFIDataset(BaseDataset):

    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]

        prompt = row["query"]
        equiv_prompt = random.choice(row["query_rephrased"])
        unrel_prompt = row["query_unrelated"]
        alt = row["label"]
        ans = row["label_unrelated"]
        embeddings = row["document_embeddings"]

        return {
            "edit_tuples": self.tok_tuples(prompt, alt),
            "equiv_tuples": self.tok_tuples(equiv_prompt, alt),
            "unrel_tuples": self.tok_tuples(unrel_prompt, ans),
            "embeddings": torch.tensor(embeddings)
        }
    
    def tok_tuples(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, torch.LongTensor]:
        
        tok_tuples = self.tok(
            prompt,
            max_length = 512,
            return_tensors = "pt",
            truncation = True
        )
        tok_tuples["labels"] = torch.tensor(answer)

        return tok_tuples


