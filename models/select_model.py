from .src.mantranet.mantranet import get_mantranet


def select_model(model_name, pretrained=True):
    if model_name == 'mantranet':
        return get_mantranet(pretrained=pretrained)

    else:
        print(f'[Error] Model {model_name} is not supported!')
        exit()