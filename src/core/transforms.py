from torchvision import transforms

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(260),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
