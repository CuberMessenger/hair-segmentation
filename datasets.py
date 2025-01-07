import os
import cv2
import torch
import torchvision.transforms.v2 as transforms

from torch.utils.data import Dataset


def pad_to_square(image):
    """
    image: torch.Tensor, shape [C, H, W]
    """

    _, h, w = image.shape

    if h == w:
        return image

    if h > w:
        pad = (h - w) // 2
        pad_left = pad
        pad_right = h - w - pad
        pad_top = 0
        pad_bottom = 0
    else:
        pad = (w - h) // 2
        pad_top = pad
        pad_bottom = w - h - pad
        pad_left = 0
        pad_right = 0

    image = torch.nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom))

    return image


class FigaroDataset(Dataset):
    def __init__(self, dataset_folder, split):
        """
        dataset_folder: str, path to the dataset folder
                - folder
                    - GT
                        - Training
                            Frame00001-gt.pbm
                            ...
                        - Testing
                            ...
                    - Original
                        - Training
                            Frame00001-org.jpg
                            ...
                        - Testing
                            ...

        split: str, "train" or "test"
        """
        super().__init__()

        if split == "train":
            split = "Training"
        elif split == "test":
            split = "Testing"
        else:
            raise ValueError(f"Invalid split: {split}")

        self.original_folder = os.path.join(dataset_folder, "Original", split)
        self.groundtruth_folder = os.path.join(dataset_folder, "GT", split)

        self.names = [name.split("-")[0] for name in os.listdir(self.original_folder)]

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomAffine(
                        degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.transform = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        image = cv2.imread(
            os.path.join(self.original_folder, name + "-org.jpg"), cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.Tensor(image).permute(2, 0, 1).float() / 255.0

        mask = cv2.imread(
            os.path.join(self.groundtruth_folder, name + "-gt.pbm"),
            cv2.IMREAD_GRAYSCALE,
        )
        mask = torch.Tensor(mask)[None, ...].float() / 255.0

        pair = torch.cat([image, mask], dim=0)
        pair = pad_to_square(pair)
        pair = self.transform(pair)

        image = pair[:3, ...]
        mask = pair[3, ...][None, ...].clamp(0, 1)

        ###
        # image_to_write = (
        #     cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR) * 255
        # )
        # cv2.imwrite(os.path.join("outputs", f"image_{index}.jpg"), image_to_write)
        # cv2.imwrite(os.path.join("outputs", f"mask_{index}.jpg"), mask[0].numpy() * 255)
        ###
        return image, mask


if __name__ == "__main__":
    # path = r"datasets\Figaro-1k\GT\Testing\Frame00010-gt.pbm"
    # image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # print(image.shape)
    # print(np.min(image), np.max(image))
    # print(np.unique(image))

    d = FigaroDataset(r"datasets\Figaro-1k", "test")
    for i in range(10):
        print(d[i][0].shape, d[i][1].shape)
