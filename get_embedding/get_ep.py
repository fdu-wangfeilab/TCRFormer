from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np
import re
from tqdm import tqdm

def get_ep_emb(eps_data,pretrain_path='/mnt/sdb/tyh/prot_t5_xl_half_uniref50-enc/',device='cuda:1'):
    eps=list(set(eps_data))
    device = torch.device(device)

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(pretrain_path, do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained(pretrain_path).to(device)
    
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in eps]
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    
    eps=np.array(eps)
    eps_data=np.array(eps_data)
    
    save=[]
    for i in tqdm(eps_data,desc='loading ep embedding'):
        arg=np.argwhere(eps==i)[0][0]
        save.append(embedding_repr.last_hidden_state[arg].detach().cpu().numpy())
    save=np.array(save)
    
    
    return save
    