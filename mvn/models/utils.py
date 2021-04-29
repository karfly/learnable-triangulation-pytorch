import torch.optim as optim


def get_params(layer, as_list=True):
    params = layer.parameters()

    if as_list:
        return list(params)

    return params


def count_grad_params(layer):
    params = get_params(layer, as_list=False)
    return sum(
        p.data.nelement()
        for p in params
        if p.requires_grad
    )


def build_opt(model, config, base_optim=optim.Adam):
    bb_params = list(model.backbone.parameters())

    for name, m in model.backbone.named_children():
        print(name, count_grad_params(m))

    print(model)

    1/0

    if config.model.name == "vol":
        return base_optim(
            [
                {
                    'params': bb_params
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

    return base_optim(
        filter(lambda p: p.requires_grad, bb_params),
        lr=config.opt.lr
    )
