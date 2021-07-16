import torch


def _get_torch_version():
    version = torch.__version__.split('.')
    return float(version[0]) + float(version[1]) / 10


def get_params(layer, as_list=True):
    params = layer.parameters()

    if as_list:
        return list(params)

    return params


def get_grad_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def count_grad_params(layer):
    params = get_params(layer, as_list=False)
    return sum(
        p.data.nelement()
        for p in params
        if p.requires_grad
    )


def freeze_layer(layer, verbose=False):
    if verbose:
        print('freezing {}'.format(layer._get_name()))

    for p in layer.parameters():
        p.requires_grad = False


def reset_layer(layer):
    try:
        layer.reset_parameters()
    except:
        for layer in layer.children():
            reset_layer(layer)


def show_params(model, verbose=False):
    if verbose:
        print(model)

    tot = count_grad_params(model)

    for name, m in model.named_children():
        n_params = count_grad_params(m)
        as_perc = n_params / tot * 100.0
        print('{:>30} has {:10.0f} params (~ {:4.1f}) %'.format(
            name, n_params, as_perc
        ))

    print('total params: {:20.0f}'.format(
        tot
    ))


def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)

    for key in list(state_dict.keys()):
        new_key = key.replace("module.", "")
        state_dict[new_key] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)

    print('successfully loaded pretrained weights from {}'.format(checkpoint_path))


def freeze_backbone(model):
    """ BB has already been pre-trained on the COCO dataset and finetuned jointly on MPII and Human3.6M for 10 epochs using the Adam optimizer with 10−4 learning rate => freeze most layers and optimize just the last ones """

    freeze_layer(model.backbone.conv1)
    freeze_layer(model.backbone.bn1)
    freeze_layer(model.backbone.relu)
    freeze_layer(model.backbone.maxpool)
    freeze_layer(model.backbone.layer1)
    freeze_layer(model.backbone.layer2)
    freeze_layer(model.backbone.layer3)
    freeze_layer(model.backbone.layer4)

    # reset_layer(model.backbone.alg_confidences)
    # reset_layer(model.backbone.deconv_layers)
    # reset_layer(model.backbone.final_layer)

    # debug only show_params(model.backbone)
