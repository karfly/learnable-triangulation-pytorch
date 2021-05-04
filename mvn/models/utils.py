import torch.optim as optim


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


def freeze_layer(layer):
    print('freezing {}'.format(layer._get_name()))

    for p in layer.parameters():
        p.requires_grad = False


def reset_layer(layer):
    try:
        layer.reset_parameters()
    except:
        for layer in layer.children():
            reset_layer(layer)


def show_params(model):
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


def freeze_backbone(model):
    """ BB has already been pre-trained on the COCO dataset and finetuned jointly on MPII and Human3.6M for 10 epochs using the Adam optimizer with 10−4 learning rate => freeze most layers and optimize just the last ones """

    show_params(model.backbone)

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

    show_params(model.backbone)


def build_opt(model, cam2cam_model, config, base_optim=optim.Adam):
    freeze_backbone(model)

    if config.model.cam2cam_estimation:
        print('cam2cam model:')
        show_params(cam2cam_model)

        return base_optim(
            [
                {
                    'params': get_grad_params(model.backbone),
                    'lr': 1e-6  # BB already optimized
                },
                {
                    'params': get_grad_params(cam2cam_model),
                    'lr': 1e-5  # try me: 1e-4 seems too much larger, NaN when triangulating
                }
            ]
        )
    elif config.model.name == "vol":
        return base_optim(
            [
                {
                    'params': get_grad_params(model.backbone),
                    'lr': 1e-6  # BB already optimized
                },
                {
                    'params': model.process_features.parameters(),
                    'lr': config.opt.process_features_lr if hasattr(config.opt, 'process_features_lr') else config.opt.lr
                },
                {
                    'params': model.volume_net.parameters(),
                    'lr': config.opt.volume_net_lr if hasattr(config.opt, 'volume_net_lr') else config.opt.lr
                }
            ],
            lr=config.opt.lr
        )
    else:
        return base_optim(
            get_grad_params(model.backbone),
            lr=1e-6  # BB already optimized
        )
