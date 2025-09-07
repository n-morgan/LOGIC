from omegaconf import DictConfig

import torch
import torch.nn as nn

import transformers
from transformers import AutoModel

from util import get_module
import logging

class AutoModelForFEVER(nn.Module):
    
    def __init__(self, name_or_path: str):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(name_or_path)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)



    def forward(self, **kwargs):

        # Fix input tensors to add batch dimension if needed
        for key in ["input_ids", "attention_mask"]:
            if key in kwargs and kwargs[key].dim() == 1:
                kwargs[key] = kwargs[key].unsqueeze(0)
        
        hidden_states = self.backbone(**{
            k: v for k, v in kwargs.items() if k not in ["labels", "embeddings"]
        })["last_hidden_state"][:, 0]

        embeddings = kwargs["embeddings"]
        logits = hidden_states @ embeddings.T  # Note transpose

        return {"logits": logits}


class AutoModelForSCIFI(nn.Module):
    
    def __init__(self, name_or_path: str):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(name_or_path)
        


    def forward(self, **kwargs):
        #print (type(kwargs))
        #for k,v in kwargs.items():
            #print(k, v, v.size())
        
        hidden_states = self.backbone(**{
            k: v for k, v in kwargs.items() if k not in ["labels", "embeddings"]
        })["last_hidden_state"][:, 0]

        hidden_states_unsqueezed =  hidden_states.unsqueeze(1)
        embeddings_tensor = kwargs["embeddings"] 
        classifier = embeddings_tensor.transpose(1,2) # extract embeddings from classifier
        print(hidden_states_unsqueezed.size(), classifier.size())
        logits = torch.bmm(hidden_states_unsqueezed, classifier)     # finish making layer


        print('***********************LINE BREAK ***************************')
        print(logits, logits.shape)


        return {"logits": logits}



def make_model(config: DictConfig):
    if config.class_name == "AutoModelForFEVER":
        model = AutoModelForFEVER(config.weight_path)


    if config.class_name == "AutoModelForSCIFI":

        model = AutoModelForSCIFI(config.weight_path)
    else: # hitting this else statement
        model_class = getattr(transformers, config.class_name)
        model = model_class.from_pretrained(config.name_or_path)

    if config.half:
        model.bfloat16()

    for param in model.parameters():
        param.requires_grad = False
        
    for module_name in config.edit_modules:
        module = get_module(model, module_name)
        module.weight.requires_grad = True
        
    return model
