from torchvision import transforms


class CoreTranform:
    @staticmethod
    def get_transorms() -> dict[str, transforms.Compose]:
        """
        Returns a dictionary of data augmentation transforms for
        training and validation datasets.

        - "train" transform is designed for data augmentation during
        training, including random resizing, horizontal flipping, and
        normalization.

        - "val" transform is used for the validation dataset, including
        resizing, center cropping, and normalization.
        """
        return {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(260),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }
