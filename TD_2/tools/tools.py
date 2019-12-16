import torch
import torchvision
from PIL import Image
from IPython.display import display
import numpy as np


def load_image_in_vgg_format(infilename):

    # Tout d'abord l'image est chargée depuis le fichier d'entrée
    # Notez que l'image obtenue n'est pas un tensor torch mais un objet Image
    # dans un autre format (PIL en l'occurence)
    img = Image.open(infilename)
    img.load()

    # Le réseau vgg ne prend en entrée que des images couleurs avec L = H = 244
    # Il faut donc redimensioner correctement l'image.
    # Cette transformation prend en entrée une image au format PIL et renvoie
    # la version redimensionnée de l'image précédente, toujours au format PIL
    size_vgg_16 = (224, 224)
    resize = torchvision.transforms.Resize(size_vgg_16, interpolation=2)

    # Le format PIL c'est bien mais nous travaillons avec des tensors.
    # Il faut donc une transformation qui convertit une image PIL en un
    # torch.tensor
    # Notes:
    # 1) cette fonction normalize aussi l'image: elle divise la valeur de
    # chaque pixel par 255 pour n'avoir que des valeurs comprises entre 0 et 1
    # 2) de le tensor de sortie est au format 3 x H x L (et non plus LxHx3)
    to_tensor = torchvision.transforms.ToTensor()

    # Le réseau VGG requiert que l'image soit normalisée, en d'autres termes
    # il faut que pour tout pixel X (x_r, x_v, x_b) de l'image R V B on ait
    #
    #  x_i = (x_i - mean_i) / (std_i) ( i dans {r, v, b})
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])


    # La transformation finale est la composition des deux transformations
    # précédente et de le transformation torchvision.transforms.ToTensor()
    # qui convertit une image au format PIL en un tensor
    vgg_transform = torchvision.transforms.Compose([resize,
                                                    to_tensor,
                                                    normalize])

    # La transformation est appliquée à l'image
    img = vgg_transform(img)
    C, L, H = img.size()

    # vgg veut des images au format batch N x C x H x L. Or quand nous n'avons
    # qu'une seule image N=1
    return img.view(1, C, L, H)


def show_image_from_path(infilename):
    pil_im = Image.open(infilename, 'r')
    display(pil_im)


def show_images_from_path_list(file_names):
    data = []
    w, h = 0, 0
    for name in file_names:
        pil_im = Image.open(name, 'r')
        loc_arr = np.asarray(pil_im)
        h = max(h, loc_arr.shape[0])
        w = w + loc_arr.shape[1]
        data.append(loc_arr)
    out_array = np.zeros((h, w, 3), dtype='uint8')
    shift = 0
    for image in data:
        h, w, c = image.shape
        out_array[:h, shift: shift+w] = image
        shift += w
    display(Image.fromarray(out_array))


def show_batch(in_batch):
    N, H, L = in_batch.size()
    s = 224
    out = np.zeros((s, s*N), dtype='uint8')
    for n in range(N):
        x = torch.clamp(in_batch[n], 0, 255)
        x = x.detach().numpy().astype(np.uint8)
        im = Image.fromarray(x)
        im = im.resize((s, s), resample=0)
        out[:, s*n:(n+1)*s] = np.asarray(im)
    display(Image.fromarray(out))

def load_image_net_labels(path_data):
    with open(path_data, 'r') as file:
        data = file.readlines()

    output = [ None  for x in range(len(data))]

    for line in data:
        index = line.split()[0]
        if index[-1] == ':':
            index = index[:-1]
        index = int(index)
        start_label = line.find("'")
        label = line[start_label + 1:]
        end_label = label.find("'")
        label = label[:end_label]
        output[index] = label
    return output


class ImageNetLabelMatch():
    def __init__(self, path_data):
        self.labels = load_image_net_labels(path_data)

    def __call__(self, vgg_output_tensor):

        N, L = vgg_output_tensor.size()
        assert(len(self.labels) == L)

        proba_tensor = torch.nn.functional.softmax(vgg_output_tensor, dim=1)
        prediction_values, prediction_index = proba_tensor.max(dim=1)
        output = []
        for n in range(N):
            output.append({"nom": self.labels[prediction_index[n]], "proba": prediction_values[n].item()})
        return output
