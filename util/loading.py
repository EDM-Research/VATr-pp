import collections

OLD_KEYS = ['netG._logvarD.bias', 'netG._logvarD.weight', 'netG._logvarE.bias', 'netG._logvarE.weight', 'netG._muD.bias', 'netG._muD.weight', 'netG._muE.bias', 'netG._muE.weight', 'netD.embed.weight', 'netD.embed.u0', 'netD.embed.sv0', 'netD.embed.bias']


def load_generator(model, checkpoint):
    if not isinstance(checkpoint, collections.OrderedDict):
        checkpoint = checkpoint['model']

    checkpoint = {k.replace("netG.",""): v for k, v in checkpoint.items() if k.startswith("netG") and k not in OLD_KEYS}
    model.netG.load_state_dict(checkpoint)

    return model


def load_checkpoint(model, checkpoint):
    if not isinstance(checkpoint, collections.OrderedDict):
        checkpoint = checkpoint['model']
    old_model = model.state_dict()
    if len(checkpoint.keys()) == 241:  # default
        counter = 0
        for k, v in checkpoint.items():
            if k in old_model:
                old_model[k] = v
                counter += 1
            elif 'netG.' + k in old_model:
                old_model['netG.' + k] = v
                counter += 1

        ckeys = [k for k in checkpoint.keys() if 'Feat_Encoder' in k]
        okeys = [k for k in old_model.keys() if 'Feat_Encoder' in k]
        for ck, ok in zip(ckeys, okeys):
            old_model[ok] = checkpoint[ck]
            counter += 1
        assert counter == 241
        checkpoint_dict = old_model
    else:
        checkpoint = {k: v for k, v in checkpoint.items() if k not in OLD_KEYS}
        assert len(old_model) == len(checkpoint)
        checkpoint_dict = {k2: v1 for (k1, v1), (k2, v2) in zip(checkpoint.items(), old_model.items()) if
                           v1.shape == v2.shape}
    assert len(old_model) == len(checkpoint_dict)
    model.load_state_dict(checkpoint_dict, strict=False)
    return model