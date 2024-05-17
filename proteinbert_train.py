#!/usr/bin/env python
import abc
import pickle
import math

import wandb

import torch

from tqdm import tqdm
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, SGD
from collections import namedtuple

import proteinbert_gen.constants as consts
import proteinbert_gen.mask_diffusion as mask_diffusion

from proteinbert_gen.debugging import print2
from proteinbert_gen.proteinbert import ProteinBERT, load_pretrained_weights
from proteinbert_gen.word_freq import create_word_freq_tensor
from proteinbert_gen.tokenizer import ProteinTokenizer
from proteinbert_gen.dataset import SwissProtDataset, calculate_splits

Hyperparameters = namedtuple(
    "Hyperparameters",
    [
        "batch_size",
        "epochs",
        "num_steps",
        "word_freq_lambda",
        "device",
        "hybrid_lambda",
        "lr",
        "logging_steps",
        "eval_step_size",
        "clip_grad",
        "clip_grad_val",
        "warmup_scheduler",
        "optimizer_cls",
        "warmup_steps",
        "data_csvfile",
        "word_freq_dict_pkl",
        "num_blocks",
        "d_local",
        "d_global"
    ]
)

args = Hyperparameters(
    batch_size=64,
    epochs=100,
    num_steps=4096,
    word_freq_lambda=0.3,
    device="cuda",
    hybrid_lambda=1e-3,
    lr=5e-4,
    logging_steps=25,
    eval_step_size=4,
    clip_grad_val=10,
    clip_grad=False,
    warmup_scheduler=True,
    optimizer_cls=AdamW,
    warmup_steps=50000,
    num_blocks=12,
    d_local=256,
    d_global=1024,
    data_csvfile="../data/uniprot_sprot_1m_1024.csv",
    word_freq_dict_pkl="../data/uniprot_sprot_1m_1024_word_freq_dict.pkl"
)

run = wandb.init(
    project="proteinbert_gen",
    config={k:str(v) for k, v in args._asdict().items()},
    # mode="disabled"
)

sprot_all = SwissProtDataset(args.data_csvfile)
sprot_train, sprot_val, sprot_test = torch.utils.data.random_split(
        sprot_all,
        calculate_splits(0.7, 0.2, 0.1, len(sprot_all)),
        generator=torch.Generator().manual_seed(42)
        )

class SampleClassBase(abc.ABC):
    def sample(self, logits, x_0):
        raise NotImplementedError

    def post_process_sample_in_prediction(self, sample, x_0):
        return sample


class Categorical(SampleClassBase):
    def sample(self, logits, x_0):
        return torch.distributions.categorical.Categorical(logits=logits).sample()


def word_freq_preprocess_fn(wf):
    wf = wf + 1
    wf = wf.log()
    wf = wf / wf.max()

    # range: 0 - 1
    return wf


def process_fn_in_collate(wf):
    return wf - wf.mean()


tokenizer = ProteinTokenizer()
wf_tensor = create_word_freq_tensor(args.word_freq_dict_pkl, tokenizer.ALL_TOKENS)
# wf_tensor[tokenizer.mask_token_id] = 0
wf_tensor[tokenizer.pad_token_id] = 0
wf_tensor = word_freq_preprocess_fn(wf_tensor)
print(wf_tensor)


def collate(batch_input, *, tokenizer, word_freq: torch.Tensor):
    input_ids = []
    attention_mask = []
    word_freq_logits = []
    
    for item in batch_input:
        seq = item["seq"]
        ids = torch.tensor(tokenizer.tokenize(seq))
        mask = torch.ones_like(ids)
        logits = process_fn_in_collate(
            word_freq.gather(0, ids)
        )
        
        input_ids.append(ids)
        attention_mask.append(mask)
        word_freq_logits.append(logits)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    word_freq_logits = pad_sequence(word_freq_logits, batch_first=True)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_freq_logits": word_freq_logits
    }

collate_fn = partial(collate, tokenizer=tokenizer, word_freq=wf_tensor)

train_loader = torch.utils.data.DataLoader(
    sprot_train,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

sample_batch = next(iter(train_loader))
print(sample_batch)
print(sample_batch["input_ids"].size())


def denoise(targets, timestep, attention_mask, *, model):
    ret = model(targets)
    # ret = model(targets, attention_mask=attention_mask)
    # print("denoise output:", ret.shape)
    return ret


with open("../weights/epoch_92400_sample_23500000.pkl", "rb") as f:
    _, pretrained_model_weights, _ = pickle.load(f)

model = ProteinBERT(
    tokenizer.vocab_size,
    consts.GO_ANN_SIZE,
    d_local=256,
    d_global=1024,
    num_blocks=12
)
print(model)

# trainable_params = load_pretrained_weights(model, pretrained_model_weights)
trainable_params = list(model.parameters())
model = model.to(args.device)
denoise_fn = partial(denoise, model=model)


# OPTIMIZER

optimizer = args.optimizer_cls(trainable_params, lr=args.lr)
if args.warmup_scheduler:
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda n: n / args.warmup_steps + 1e-3 if n < args.warmup_steps else math.sqrt(args.warmup_steps) / math.sqrt(n)
    )


# DIFFUSION

sample_cls = Categorical()

diffusion_schedule = mask_diffusion.create_discrete_diffusion_schedule(num_steps=args.num_steps)
diffusion_instance = mask_diffusion.MaskDiffusion(
    dim=tokenizer.vocab_size,
    schedule=diffusion_schedule,
    tokenizer=tokenizer,
    sample_cls=sample_cls,
    word_freq_lambda=args.word_freq_lambda,
    device=args.device
)


train_loss = 0.
has_nan_log = 0
nan_count = 0

# torch.autograd.set_detect_anomaly(True)

for epoch in range(args.epochs):
    for i, batch in enumerate(tqdm(train_loader)):
        run.log({"epoch": epoch, "minibatch": i}, commit=False)
        
        optimizer.zero_grad()
        diffusion_t = diffusion_instance.sample_t()
        # print(diffusion_t)

        metrics = mask_diffusion.compute_kl_reverse_process(
            batch["input_ids"].to(args.device),
            diffusion_t,
            denoise_fn=denoise_fn,
            diffusion=diffusion_instance,
            target_mask=batch["attention_mask"].to(args.device),
            hybrid_lambda=args.hybrid_lambda,
            predict_x0=True, # False,
            word_freq_logits=batch["word_freq_logits"].to(args.device),
            device=args.device
        )

        # print(metrics)

        loss = metrics["loss"] / args.batch_size / batch["input_ids"].size(1)

        if loss.isnan():
            nan_count += 1
            if i % args.logging_steps == args.logging_steps - 1:
                run.log({"nan_count": nan_count})
            continue
            
        train_loss += loss.item()
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_value_(trainable_params, args.clip_grad_val)
        
        has_nan = 0
        for param in trainable_params:
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)
                    has_nan = 1

        has_nan_log += has_nan
        
        optimizer.step()
        if args.warmup_scheduler:
            warmup_scheduler.step()

        if i % args.logging_steps == args.logging_steps - 1:
            run.log(metrics, commit=False)
            if args.warmup_scheduler:
                run.log({"last_lr": warmup_scheduler.get_last_lr()[0]}, commit=False)
            run.log({"nan_count": nan_count, "nan -> zero": has_nan_log})
            has_nan_log = 0

    # generate some proteins
    print(f"Post epoch {epoch}")
    
    generated_final_states = []
    for length in (200, 500, 800):
        generated = mask_diffusion.discrete_diffusion_predict_fn((4, length), denoise_fn, diffusion_instance, topp=1.0)
        generated_final_states.extend(generated["final_state"].tolist())
    
    generated_table = wandb.Table(columns=["gen_id", "seq"])
    for j, genseq in enumerate(generated_final_states):
        genprot = tokenizer.untokenize(genseq)
        generated_table.add_data(j, genprot)
        print(genprot)
    run.log({"generated_proteins": generated_table})

    torch.save(model.state_dict(), f"../checkpoints/{run.name}-postepoch-{epoch}.pt")
    torch.save(optimizer.state_dict(), f"../checkpoints/{run.name}-postepoch-{epoch}-optimizer.pt")

