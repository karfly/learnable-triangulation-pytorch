import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


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


def build_opt(model, cam2cam_model, config, base_optim=optim.Adam):  # if _get_torch_version() >= 1.8 else optim.AdamW):
    freeze_backbone(model)

    if config.model.cam2cam_estimation:
        print('cam2cam estimation => adding {:.0f} params to grad ...'.format(
            count_grad_params(cam2cam_model)
        ))

        params = [
            {
                'params': get_grad_params(cam2cam_model),
                'lr': config.cam2cam.opt.lr  # try me: 1e-4 seems too much larger, NaN when triangulating
            }
        ]

        if not config.cam2cam.using_gt:  # predicting KP and HM -> need to opt
            print('using predicted KPs => adding model.backbone to grad ...')
            params.append(
                {
                    'params': get_grad_params(model.backbone),
                    'lr': 1e-6  # BB already optimized
                }
            )

        opt = base_optim(params, weight_decay=1e1)
    elif config.model.name == "vol":
        print('volumetric method => adding model.{{ {}, {}, {} }} params to grad ...'.format(
            'backbone',
            'process_features',
            'volume_net'
        ))

        opt = base_optim(
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
        print('standard method => adding model.backbone {:.0f} params to grad ...'.format(
            count_grad_params(model.backbone)
        ))

        opt = base_optim(
            get_grad_params(model.backbone),
            lr=1e-6  # BB already optimized
        )

    scheduler = ReduceLROnPlateau(
        opt,
        factor=5e-1,  # new lr = x * lr
        patience=75,  # n max iterations since optimum
        # threshold=42,  # no matter what, do lr decay
        min_lr=1e-6,
        verbose=True
    )  # https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/weight-loss-plateau/art-20044615

    return opt, scheduler
