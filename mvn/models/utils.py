import torch.optim as optim


def build_opt(model, config, base_optim=optim.Adam):
    bb_params = model.backbone.parameters()

    if config.model.name == "vol":
        return base_optim(
            [
                {
                    'params': bb_params
                },
                {
                    'params': model.process_features.parameters(),
                    'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr
                },
                {
                    'params': model.volume_net.parameters(),
                    'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr
                }
            ],
            lr=config.opt.lr
        )

    return base_optim(
        filter(lambda p: p.requires_grad, bb_params),
        lr=config.opt.lr
    )
