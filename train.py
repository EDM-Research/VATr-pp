import argparse
import random
import math
import time
import os

import numpy as np
import torch
import wandb

from data.dataset import TextDataset, CollectionTextDataset
from models.model import VATr
from util.misc import EpochLossTracker, add_vatr_args, LinearScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action='store_true')
    parser = add_vatr_args(parser)

    args = parser.parse_args()

    rSeed(args.seed)
    dataset = CollectionTextDataset(
        args.dataset, 'files', TextDataset, file_suffix=args.file_suffix, num_examples=args.num_examples,
        collator_resolution=args.resolution, min_virtual_size=339, validation=False, debug=False, height=args.img_height
    )
    datasetval = CollectionTextDataset(
        args.dataset, 'files', TextDataset, file_suffix=args.file_suffix, num_examples=args.num_examples,
        collator_resolution=args.resolution, min_virtual_size=161, validation=True, height=args.img_height
    )

    args.num_writers = dataset.num_writers

    if args.dataset == 'IAM' or args.dataset == 'CVL':
        args.alphabet = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
    else:
        args.alphabet = ''.join(sorted(set(dataset.alphabet + datasetval.alphabet)))
        args.special_alphabet = ''.join(c for c in args.special_alphabet if c not in dataset.alphabet)

    args.exp_name = f"{args.dataset}-{args.num_writers}-{args.num_examples}-LR{args.g_lr}-bs{args.batch_size}-{args.tag}"

    config = {k: v for k, v in args.__dict__.items() if isinstance(v, (bool, int, str, float))}
    args.wandb = args.wandb and (not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'Tesla K80')
    wandb_id = wandb.util.generate_id()

    MODEL_PATH = os.path.join(args.save_model_path, args.exp_name)
    os.makedirs(MODEL_PATH, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        collate_fn=dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        datasetval,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        collate_fn=datasetval.collate_fn)

    model = VATr(args)
    start_epoch = 0

    del config['alphabet']
    del config['special_alphabet']

    wandb_params = {
        'project': 'VATr',
        'config': config,
        'name': args.exp_name,
        'id': wandb_id
    }

    checkpoint_path = os.path.join(MODEL_PATH, 'model.pth')

    loss_tracker = EpochLossTracker()

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        wandb_params['id'] = checkpoint['wandb_id']
        wandb_params['resume'] = True
        print(checkpoint_path + ' : Model loaded Successfully')
    elif args.resume:
        raise FileNotFoundError(f'No model found at {checkpoint_path}')
    else:
        if args.feat_model_path is not None and args.feat_model_path.lower() != 'none':
            print('Loading...', args.feat_model_path)
            assert os.path.exists(args.feat_model_path)
            checkpoint = torch.load(args.feat_model_path, map_location=args.device)
            checkpoint['model']['conv1.weight'] = checkpoint['model']['conv1.weight'].mean(1).unsqueeze(1)
            del checkpoint['model']['fc.weight']
            del checkpoint['model']['fc.bias']
            miss, unexp = model.netG.Feat_Encoder.load_state_dict(checkpoint['model'], strict=False)
            if not os.path.isdir(MODEL_PATH):
                os.mkdir(MODEL_PATH)
        else:
            print(f'WARNING: No resume of Resnet-18, starting from scratch')

    if args.wandb:
        wandb.init(**wandb_params)
        wandb.watch(model)

    print(f"Starting training")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        log_time = time.time()
        loss_tracker.reset()
        model.d_acc.update(0.0)
        if args.text_augment_strength > 0:
            model.set_text_aug_strength(args.text_augment_strength)

        for i, data in enumerate(train_loader):
            model.update_parameters(epoch)
            model._set_input(data)

            model.optimize_G_only()
            model.optimize_G_step()

            model.optimize_D_OCR()
            model.optimize_D_OCR_step()

            model.optimize_G_WL()
            model.optimize_G_step()

            model.optimize_D_WL()
            model.optimize_D_WL_step()

            if time.time() - log_time > 10:
                print(
                    f'Epoch {epoch} {i / len(train_loader) * 100:.02f}% running, current time: {time.time() - start_time:.2f} s')
                log_time = time.time()

            batch_losses = model.get_current_losses()
            batch_losses['d_acc'] = model.d_acc.avg
            loss_tracker.add_batch(batch_losses)

        end_time = time.time()
        data_val = next(iter(val_loader))
        losses = loss_tracker.get_epoch_loss()
        page = model._generate_page(model.sdata, model.input['swids'])
        page_val = model._generate_page(data_val['simg'].to(args.device), data_val['swids'])

        d_train, d_val, d_fake = model.compute_d_stats(train_loader, val_loader)

        if args.wandb:
            wandb.log({
                'loss-G': losses['G'],
                'loss-D': losses['D'],
                'loss-Dfake': losses['Dfake'],
                'loss-Dreal': losses['Dreal'],
                'loss-OCR_fake': losses['OCR_fake'],
                'loss-OCR_real': losses['OCR_real'],
                'loss-w_fake': losses['w_fake'],
                'loss-w_real': losses['w_real'],
                'd_acc': losses['d_acc'],
                'd-rv': (d_train - d_val) / (d_train - d_fake),
                'd-fake': d_fake,
                'd-real': d_train,
                'd-val': d_val,
                'l_cycle': losses['cycle'],
                'epoch': epoch,
                'timeperepoch': end_time - start_time,
                'result': [wandb.Image(page, caption="page"), wandb.Image(page_val, caption="page_val")],
                'd-crop-size': model.netD.augmenter.get_current_width() if model.netD.crop else 0
            })

        print({'EPOCH': epoch, 'TIME': end_time - start_time, 'LOSSES': losses})
        print(f"Text sample: {model.get_text_sample(10)}")

        checkpoint = {
            'model': model.state_dict(),
            'wandb_id': wandb_id,
            'epoch': epoch
        }
        if epoch % args.save_model == 0:
            torch.save(checkpoint, os.path.join(MODEL_PATH, 'model.pth'))

        if epoch % args.save_model_history == 0:
            torch.save(checkpoint, os.path.join(MODEL_PATH, f'{epoch:04d}_model.pth'))


def rSeed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)


if __name__ == "__main__":
    print("Training Model")
    main()
    wandb.finish()
