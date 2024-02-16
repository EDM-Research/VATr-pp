import torch.utils.data
from torch.nn import CTCLoss
from torch.nn.utils import clip_grad_norm_
import sys
import torchvision.models as models

from analysis.inception import InceptionV3
from models.transformer import *
from util.augmentations import OCRAugment
from util.misc import SmoothedValue
from util.text import get_generator, AugmentedGenerator
from .BigGAN_networks import *
from .OCR_network import *
from models.blocks import Conv2dBlock, ResBlocks
from util.util import loss_hinge_dis, loss_hinge_gen, make_one_hot

import models.config as config
from .positional_encodings import PositionalEncoding1D
from models.unifont_module import UnifontModule
from PIL import Image


def get_rgb(x):
    R = 255 - int(int(x > 0.5) * 255 * (x - 0.5) / 0.5)
    G = 0
    B = 255 + int(int(x < 0.5) * 255 * (x - 0.5) / 0.5)
    return R, G, B


def get_page_from_words(word_lists, MAX_IMG_WIDTH=800):
    line_all = []
    line_t = []

    width_t = 0

    for i in word_lists:

        width_t = width_t + i.shape[1] + 16

        if width_t > MAX_IMG_WIDTH:
            line_all.append(np.concatenate(line_t, 1))

            line_t = []

            width_t = i.shape[1] + 16

        line_t.append(i)
        line_t.append(np.ones((i.shape[0], 16)))

    if len(line_all) == 0:
        line_all.append(np.concatenate(line_t, 1))

    max_lin_widths = MAX_IMG_WIDTH  # max([i.shape[1] for i in line_all])
    gap_h = np.ones([16, max_lin_widths])

    page_ = []

    for l in line_all:
        pad_ = np.ones([l.shape[0], max_lin_widths - l.shape[1]])

        page_.append(np.concatenate([l, pad_], 1))
        page_.append(gap_h)

    page = np.concatenate(page_, 0)

    return page * 255


class FCNDecoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(FCNDecoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        y = self.model(x)

        return y


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        INP_CHANNEL = 1

        encoder_layer = TransformerEncoderLayer(config.tn_hidden_dim, config.tn_nheads,
                                                config.tn_dim_feedforward,
                                                config.tn_dropout, "relu", True)
        encoder_norm = nn.LayerNorm(config.tn_hidden_dim) if True else None
        self.encoder = TransformerEncoder(encoder_layer, config.tn_enc_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(config.tn_hidden_dim, config.tn_nheads,
                                                config.tn_dim_feedforward,
                                                config.tn_dropout, "relu", True)
        decoder_norm = nn.LayerNorm(config.tn_hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, config.tn_dec_layers, decoder_norm,
                                          return_intermediate=True)

        self.Feat_Encoder = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.Feat_Encoder.conv1 = nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.Feat_Encoder.fc = nn.Identity()
        self.Feat_Encoder.avgpool = nn.Identity()

        # self.query_embed = nn.Embedding(self.args.vocab_size, self.args.tn_hidden_dim)
        self.query_embed = UnifontModule(
            config.tn_dim_feedforward,
            self.args.alphabet + self.args.special_alphabet,
            input_type=self.args.query_input,
            device=self.args.device
        )

        self.pos_encoder = PositionalEncoding1D(config.tn_hidden_dim)

        self.linear_q = nn.Linear(config.tn_dim_feedforward, config.tn_dim_feedforward * 8)

        self.DEC = FCNDecoder(res_norm='in', dim=config.tn_hidden_dim)

        self.noise = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.0]))

    def evaluate(self, style_images, queries):
        style = self.compute_style(style_images)

        results = []

        for i in range(queries.shape[1]):
            query = queries[:, i, :]
            h = self.generate(style, query)

            results.append(h.detach())

        return results

    def compute_style(self, style_images):
        B, N, R, C = style_images.shape
        FEAT_ST = self.Feat_Encoder(style_images.view(B * N, 1, R, C))
        FEAT_ST = FEAT_ST.view(B, 512, 1, -1)
        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2, 0, 1)
        memory = self.encoder(FEAT_ST_ENC)
        return memory

    def generate(self, style_vector, query):
        query_embed = self.query_embed(query).permute(1, 0, 2)

        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, style_vector, query_pos=query_embed)

        h = hs.transpose(1, 2)[-1]

        if self.args.add_noise:
            h = h + self.noise.sample(h.size()).squeeze(-1).to(self.args.device)

        h = self.linear_q(h)
        h = h.contiguous()

        h = h.view(h.size(0), h.shape[1] * 2, 4, -1)
        h = h.permute(0, 3, 2, 1)

        h = self.DEC(h)

        return h

    def forward(self, style_images, query):
        enc_attn_weights, dec_attn_weights = [], []

        self.hooks = [

            self.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            self.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        style = self.compute_style(style_images)

        h = self.generate(style, query)

        self.dec_attn_weights = dec_attn_weights[-1].detach()
        self.enc_attn_weights = enc_attn_weights[-1].detach()

        for hook in self.hooks:
            hook.remove()

        return h, style


class VATr(nn.Module):

    def __init__(self, args):
        super(VATr, self).__init__()
        self.args = args
        self.args.vocab_size = len(args.alphabet)

        self.epsilon = 1e-7
        self.netG = Generator(self.args).to(self.args.device)
        self.netD = Discriminator(
            resolution=self.args.resolution, crop_size=args.d_crop_size,
        ).to(self.args.device)

        self.netW = WDiscriminator(resolution=self.args.resolution, n_classes=self.args.vocab_size, output_dim=self.args.num_writers)
        self.netW = self.netW.to(self.args.device)
        self.netconverter = strLabelConverter(self.args.alphabet + self.args.special_alphabet)

        self.netOCR = CRNN(self.args).to(self.args.device)

        self.ocr_augmenter = OCRAugment(prob=0.5, no=3)
        self.OCR_criterion = CTCLoss(zero_infinity=True, reduction='none')

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx]).to(self.args.device)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=self.args.g_lr, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizer_OCR = torch.optim.Adam(self.netOCR.parameters(),
                                              lr=self.args.ocr_lr, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=self.args.d_lr, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizer_wl = torch.optim.Adam(self.netW.parameters(),
                                             lr=self.args.w_lr, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizers = [self.optimizer_G, self.optimizer_OCR, self.optimizer_D, self.optimizer_wl]

        self.optimizer_G.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.loss_G = 0
        self.loss_D = 0
        self.loss_Dfake = 0
        self.loss_Dreal = 0
        self.loss_OCR_fake = 0
        self.loss_OCR_real = 0
        self.loss_w_fake = 0
        self.loss_w_real = 0
        self.Lcycle = 0
        self.d_acc = SmoothedValue()

        self.word_generator = get_generator(args)

        self.epoch = 0

        with open('mytext.txt', 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.text = self.text.replace('\n', ' ')
            self.text = self.text.replace('\n', ' ')
            self.text = ''.join(c for c in self.text if c in (self.args.alphabet + self.args.special_alphabet))  # just to avoid problems with the font dataset
            self.text = [word.encode() for word in self.text.split()]  # [:args.num_examples]

        self.eval_text_encode, self.eval_len_text, self.eval_encode_pos = self.netconverter.encode(self.text)
        self.eval_text_encode = self.eval_text_encode.to(self.args.device).repeat(self.args.batch_size, 1, 1)

        self.rv_sample_size = 64 * 4
        self.last_fakes = []

    def update_last_fakes(self, fakes):
        for fake in fakes:
            self.last_fakes.append(fake)
        self.last_fakes = self.last_fakes[-self.rv_sample_size:]

    def update_acc(self, pred_real, pred_fake):
        correct = (pred_real >= 0.5).float().sum() + (pred_fake < 0.5).float().sum()
        self.d_acc.update(correct / (len(pred_real) + len(pred_fake)))

    def set_text_aug_strength(self, strength):
        if not isinstance(self.word_generator, AugmentedGenerator):
            print("WARNING: Text generator is not augmented, strength cannot be set")
        else:
            self.word_generator.set_strength(strength)

    def get_text_aug_strength(self):
        if isinstance(self.word_generator, AugmentedGenerator):
            return self.word_generator.strength
        else:
            return 0.0

    def update_parameters(self, epoch: int):
        self.epoch = epoch
        self.netD.update_parameters(epoch)
        self.netW.update_parameters(epoch)

    def get_text_sample(self, size: int) -> list:
        return [self.word_generator.generate() for _ in range(size)]

    def _generate_fakes(self, ST, eval_text_encode=None, eval_len_text=None):
        if eval_text_encode == None:
            eval_text_encode = self.eval_text_encode
        if eval_len_text == None:
            eval_len_text = self.eval_len_text

        self.fakes = self.netG.evaluate(ST, eval_text_encode)

        np_fakes = []
        for batch_idx in range(self.fakes[0].shape[0]):
            for idx, fake in enumerate(self.fakes):
                fake = fake[batch_idx, 0, :, :eval_len_text[idx] * self.args.resolution]
                fake = (fake + 1) / 2
                np_fakes.append(fake.cpu().numpy())
        return np_fakes

    def _generate_page(self, ST, SLEN, eval_text_encode=None, eval_len_text=None, eval_encode_pos=None, lwidth=260, rwidth=980):
        # ST -> Style?

        if eval_text_encode == None:
            eval_text_encode = self.eval_text_encode
        if eval_len_text == None:
            eval_len_text = self.eval_len_text
        if eval_encode_pos is None:
            eval_encode_pos = self.eval_encode_pos

        text_encode, text_len, _ = self.netconverter.encode(self.args.special_alphabet)
        symbols = self.netG.query_embed.symbols[text_encode].reshape(-1, 16, 16).cpu().numpy()
        imgs = [Image.fromarray(s).resize((32, 32), resample=0) for s in symbols]
        special_examples = 1 - np.concatenate([np.array(i) for i in imgs], axis=-1)

        self.fakes = self.netG.evaluate(ST, eval_text_encode)

        page1s = []
        page2s = []

        for batch_idx in range(ST.shape[0]):

            word_t = []
            word_l = []

            gap = np.ones([self.args.img_height, 16])

            line_wids = []

            for idx, fake_ in enumerate(self.fakes):

                word_t.append((fake_[batch_idx, 0, :, :eval_len_text[idx] * self.args.resolution].cpu().numpy() + 1) / 2)

                word_t.append(gap)

                if sum(t.shape[-1] for t in word_t) >= rwidth or idx == len(self.fakes) - 1 or (len(self.fakes) - len(self.args.special_alphabet) - 1) == idx:
                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            # add the examples from the UnifontModules
            word_l.append(special_examples)
            line_wids.append(special_examples.shape[1])

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:
                pad_ = np.ones([self.args.img_height, max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page1 = np.concatenate(page_, 0)

            word_t = []
            word_l = []


            line_wids = []

            sdata_ = [i.unsqueeze(1) for i in torch.unbind(ST, 1)]
            gap = np.ones([sdata_[0].shape[-2], 16])

            for idx, st in enumerate((sdata_)):

                word_t.append((st[batch_idx, 0, :, :int(SLEN.cpu().numpy()[batch_idx][idx])].cpu().numpy() + 1) / 2)
                # word_t.append((st[batch_idx, 0, :, :].cpu().numpy() + 1) / 2)

                word_t.append(gap)

                if sum(t.shape[-1] for t in word_t) >= lwidth or idx == len(sdata_) - 1:
                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:
                pad_ = np.ones([sdata_[0].shape[-2], max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page2 = np.concatenate(page_, 0)

            merge_w_size = max(page1.shape[0], page2.shape[0])

            if page1.shape[0] != merge_w_size:
                page1 = np.concatenate([page1, np.ones([merge_w_size - page1.shape[0], page1.shape[1]])], 0)

            if page2.shape[0] != merge_w_size:
                page2 = np.concatenate([page2, np.ones([merge_w_size - page2.shape[0], page2.shape[1]])], 0)

            page1s.append(page1)
            page2s.append(page2)

            # page = np.concatenate([page2, page1], 1)

        page1s_ = np.concatenate(page1s, 0)
        max_wid = max([i.shape[1] for i in page2s])
        padded_page2s = []

        for para in page2s:
            padded_page2s.append(np.concatenate([para, np.ones([para.shape[0], max_wid - para.shape[1]])], 1))

        padded_page2s_ = np.concatenate(padded_page2s, 0)

        return np.concatenate([padded_page2s_, page1s_], 1)

    def get_current_losses(self):

        losses = {}

        losses['G'] = self.loss_G
        losses['D'] = self.loss_D
        losses['Dfake'] = self.loss_Dfake
        losses['Dreal'] = self.loss_Dreal
        losses['OCR_fake'] = self.loss_OCR_fake
        losses['OCR_real'] = self.loss_OCR_real
        losses['w_fake'] = self.loss_w_fake
        losses['w_real'] = self.loss_w_real
        losses['cycle'] = self.Lcycle

        return losses

    def _set_input(self, input):
        self.input = input

        self.real = self.input['img'].to(self.args.device)
        self.label = self.input['label']

        self.set_ocr_data(self.input['img'], self.input['label'])

        self.sdata = self.input['simg'].to(self.args.device)
        self.slabels = self.input['slabels']

        self.ST_LEN = self.input['swids']

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.text_encode, self.len_text, self.encode_pos = self.netconverter.encode(self.label)
        self.text_encode = self.text_encode.to(self.args.device).detach()
        self.len_text = self.len_text.detach()

        self.words = [self.word_generator.generate().encode('utf-8') for _ in range(self.args.batch_size)]
        self.text_encode_fake, self.len_text_fake, self.encode_pos_fake = self.netconverter.encode(self.words)
        self.text_encode_fake = self.text_encode_fake.to(self.args.device)
        self.one_hot_fake = make_one_hot(self.text_encode_fake, self.len_text_fake, self.args.vocab_size).to(
            self.args.device)

        self.fake, self.style = self.netG(self.sdata, self.text_encode_fake)

        self.update_last_fakes(self.fake)

    def pad_width(self, t, new_width):
        result = torch.ones((t.size(0), t.size(1), t.size(2), new_width), device=t.device)
        result[:,:,:,:t.size(-1)] = t

        return result

    def compute_real_ocr_loss(self, ocr_network = None):
        network = ocr_network if ocr_network is not None else self.netOCR
        real_input = self.ocr_images
        input_images = real_input
        input_labels = self.ocr_labels

        input_images = input_images.detach()

        if self.ocr_augmenter is not None:
            input_images = self.ocr_augmenter(input_images)

        pred_real = network(input_images)
        preds_size = torch.IntTensor([pred_real.size(0)] * len(input_labels)).detach()
        text_encode, len_text, _ = self.netconverter.encode(input_labels)

        loss = self.OCR_criterion(pred_real, text_encode.detach(), preds_size, len_text.detach())

        return torch.mean(loss[~torch.isnan(loss)])

    def compute_fake_ocr_loss(self, ocr_network = None):
        network = ocr_network if ocr_network is not None else self.netOCR

        pred_fake_OCR = network(self.fake)
        preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * self.args.batch_size).detach()
        loss_OCR_fake = self.OCR_criterion(pred_fake_OCR, self.text_encode_fake.detach(), preds_size,
                                           self.len_text_fake.detach())
        return torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])

    def set_ocr_data(self, images, labels):
        self.ocr_images = images.to(self.args.device)
        self.ocr_labels = labels

    def backward_D_OCR(self):
        self.real.__repr__()
        self.fake.__repr__()
        pred_real = self.netD(self.real.detach())
        pred_fake = self.netD(**{'x': self.fake.detach()})

        self.update_acc(pred_real, pred_fake)

        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(),
                                                          self.len_text.detach(), True)

        self.loss_D = self.loss_Dreal + self.loss_Dfake

        if not self.args.no_ocr_loss:
            self.loss_OCR_real = self.compute_real_ocr_loss()
            loss_total = self.loss_D + self.loss_OCR_real
        else:
            loss_total = self.loss_D

        # backward
        loss_total.backward()
        if not self.args.no_ocr_loss:
            self.clean_grad(self.netOCR.parameters())

        return loss_total

    def clean_grad(self, params):
        for param in params:
            param.grad[param.grad != param.grad] = 0
            param.grad[torch.isnan(param.grad)] = 0
            param.grad[torch.isinf(param.grad)] = 0

    def backward_D_WL(self):
        # Real
        pred_real = self.netD(self.real.detach())

        pred_fake = self.netD(**{'x': self.fake.detach()})

        self.update_acc(pred_real, pred_fake)

        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(),
                                                          self.len_text.detach(), True)

        self.loss_D = self.loss_Dreal + self.loss_Dfake

        if not self.args.no_writer_loss:
            self.loss_w_real = self.netW(self.real.detach(), self.input['wcl'].to(self.args.device)).mean()
            # total loss
            loss_total = self.loss_D + self.loss_w_real * self.args.writer_loss_weight
        else:
            loss_total = self.loss_D

        # backward
        loss_total.backward()

        return loss_total

    def optimize_D_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], True)
        self.set_requires_grad([self.netW], True)

        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.backward_D_WL()

    def optimize_D_WL_step(self):
        self.optimizer_D.step()
        self.optimizer_wl.step()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

    def compute_cycle_loss(self):
        fake_input = torch.ones_like(self.sdata)
        width = min(self.sdata.size(-1), self.fake.size(-1))
        fake_input[:, :, :, :width] = self.fake.repeat(1, 15, 1, 1)[:, :, :, :width]
        with torch.no_grad():
            fake_style = self.netG.compute_style(fake_input)

        return torch.sum(torch.abs(self.style.detach() - fake_style), dim=1).mean()

    def backward_G_only(self):
        self.gb_alpha = 0.7
        if self.args.is_cycle:
            self.Lcycle = self.compute_cycle_loss()

        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake}), self.len_text_fake.detach(), True).mean()

        compute_ocr = not self.args.no_ocr_loss

        if compute_ocr:
            self.loss_OCR_fake = self.compute_fake_ocr_loss()

        self.loss_G = self.loss_G + self.Lcycle

        if compute_ocr:
            self.loss_T = self.loss_G + self.loss_OCR_fake
        else:
            self.loss_T = self.loss_G

        if compute_ocr:
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]
            self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)

        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        self.loss_T.backward(retain_graph=True)

        if compute_ocr:
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
            a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR))
            self.loss_OCR_fake = a.detach() * self.loss_OCR_fake
            self.loss_T = self.loss_G + self.loss_OCR_fake
        else:
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
            a = 1
            self.loss_T = self.loss_G

        if a is None:
            print(self.loss_OCR_fake, self.loss_G, torch.std(grad_fake_adv))
        if a > 1000 or a < 0.0001:
            print(f'WARNING: alpha > 1000 or alpha < 0.0001 - alpha={a.item()}')

        self.loss_T.backward(retain_graph=True)
        if compute_ocr:
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        with torch.no_grad():
            self.loss_T.backward()
        if compute_ocr:
            if any(torch.isnan(torch.unsqueeze(self.loss_OCR_fake, dim=0))) or torch.isnan(self.loss_G):
                print('loss OCR fake: ', self.loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
                sys.exit()

    def backward_G_WL(self):
        self.gb_alpha = 0.7
        if self.args.is_cycle:
            self.Lcycle = self.compute_cycle_loss()

        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake}), self.len_text_fake.detach(), True).mean()

        if not self.args.no_writer_loss:
            self.loss_w_fake = self.netW(self.fake, self.input['wcl'].to(self.args.device)).mean()

        self.loss_G = self.loss_G + self.Lcycle

        if not self.args.no_writer_loss:
            self.loss_T = self.loss_G + self.loss_w_fake * self.args.writer_loss_weight
        else:
            self.loss_T = self.loss_G

        self.loss_T.backward(retain_graph=True)

        if not self.args.no_writer_loss:
            grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
            a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_WL))
            self.loss_w_fake = a.detach() * self.loss_w_fake
            self.loss_T = self.loss_G + self.loss_w_fake
        else:
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
            a = 1
            self.loss_T = self.loss_G

        if a is None:
            print(self.loss_w_fake, self.loss_G, torch.std(grad_fake_adv))
        if a > 1000 or a < 0.0001:
            print(f'WARNING: alpha > 1000 or alpha < 0.0001 - alpha={a.item()}')

        self.loss_T.backward(retain_graph=True)

        if not self.args.no_writer_loss:
            grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_WL = 10 ** 6 * torch.mean(grad_fake_WL ** 2)
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        with torch.no_grad():
            self.loss_T.backward()

    def backward_G(self):
        self.opt.gb_alpha = 0.7
        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake, 'z': self.z}), self.len_text_fake.detach(),
                                     self.opt.mask_loss)
        # OCR loss on real data
        compute_ocr = not self.args.no_ocr_loss

        if compute_ocr:
            self.loss_OCR_fake = self.compute_fake_ocr_loss()
        else:
            self.loss_OCR_fake = 0.0

        self.loss_w_fake = self.netW(self.fake, self.wcl)
        # self.loss_OCR_fake = self.loss_OCR_fake + self.loss_w_fake
        # total loss

        # l1 = self.params[0]*self.loss_G
        # l2 = self.params[0]*self.loss_OCR_fake
        # l3 = self.params[0]*self.loss_w_fake
        self.loss_G_ = 10 * self.loss_G + self.loss_w_fake
        self.loss_T = self.loss_G_ + self.loss_OCR_fake

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]

        self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
        grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        if not False:

            self.loss_T.backward(retain_graph=True)

            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=True, retain_graph=True)[0]
            # grad_fake_wl = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=True, retain_graph=True)[0]

            a = self.opt.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR))

            # a0 = self.opt.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_wl))

            if a is None:
                print(self.loss_OCR_fake, self.loss_G_, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
            if a > 1000 or a < 0.0001:
                print(f'WARNING: alpha > 1000 or alpha < 0.0001 - alpha={a.item()}')
            b = self.opt.gb_alpha * (torch.mean(grad_fake_adv) -
                                     torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR)) *
                                     torch.mean(grad_fake_OCR))
            # self.loss_OCR_fake = a.detach() * self.loss_OCR_fake + b.detach() * torch.sum(self.fake)
            self.loss_OCR_fake = a.detach() * self.loss_OCR_fake
            # self.loss_w_fake = a0.detach() * self.loss_w_fake

            self.loss_T = (1 - 1 * self.opt.onlyOCR) * self.loss_G_ + self.loss_OCR_fake  # + self.loss_w_fake
            self.loss_T.backward(retain_graph=True)
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
            self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
            with torch.no_grad():
                self.loss_T.backward()
        else:
            self.loss_T.backward()

        if self.opt.clip_grad > 0:
            clip_grad_norm_(self.netG.parameters(), self.opt.clip_grad)
        if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G_):
            print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
            sys.exit()

    def optimize_D_OCR(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.optimizer_D.zero_grad()
        # if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
        self.optimizer_OCR.zero_grad()
        self.backward_D_OCR()

    def optimize_D_OCR_step(self):
        self.optimizer_D.step()

        self.optimizer_OCR.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()

    def optimize_G_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_WL()

    def optimize_G_only(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_only()

    def optimize_G_step(self):
        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

    def save_networks(self, epoch, save_dir):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    if len(self.gpu_ids) > 1:
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def compute_d_scores(self, data_loader: torch.utils.data.DataLoader, amount: int = None):
        scores = []
        words = []
        amount = len(data_loader) if amount is None else amount // data_loader.batch_size

        with torch.no_grad():
            for i in range(amount):
                data = next(iter(data_loader))
                words.extend([d.decode() for d in data['label']])
                scores.extend(list(self.netD(data['img'].to(self.args.device)).squeeze().detach().cpu().numpy()))

        return scores, words

    def compute_d_scores_fake(self, data_loader: torch.utils.data.DataLoader, amount: int = None):
        scores = []
        words = []
        amount = len(data_loader) if amount is None else amount // data_loader.batch_size

        with torch.no_grad():
            for i in range(amount):
                data = next(iter(data_loader))
                to_generate = [self.word_generator.generate().encode('utf-8') for _ in range(data_loader.batch_size)]
                text_encode_fake, len_text_fake, encode_pos_fake = self.netconverter.encode(to_generate)
                fake, _ = self.netG(data['simg'].to(self.args.device), text_encode_fake.to(self.args.device))

                words.extend([d.decode() for d in to_generate])
                scores.extend(list(self.netD(fake).squeeze().detach().cpu().numpy()))

        return scores, words

    def compute_d_stats(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        train_values = []
        val_values = []
        fake_values = []
        with torch.no_grad():
            for i in range(self.rv_sample_size // train_loader.batch_size):
                data = next(iter(train_loader))
                train_values.append(self.netD(data['img'].to(self.args.device)).squeeze().detach().cpu().numpy())

            for i in range(self.rv_sample_size // val_loader.batch_size):
                data = next(iter(val_loader))
                val_values.append(self.netD(data['img'].to(self.args.device)).squeeze().detach().cpu().numpy())

            for i in range(self.rv_sample_size):
                data = self.last_fakes[i]
                fake_values.append(self.netD(data.unsqueeze(0)).squeeze().detach().cpu().numpy())

        return np.mean(train_values), np.mean(val_values), np.mean(fake_values)
