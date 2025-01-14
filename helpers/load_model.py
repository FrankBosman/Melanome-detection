# Load the models
def load_model(model_name, models, link_cuda=True, directory="models"):
    # Initiate pretrained model
    if "resnet18" in model_name:
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif "resnet50" in model_name:
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif "alexnet" in model_name:
        model = models.alexnet(weights='DEFAULT')
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif "efficientnet_b0" in model_name:
        model = models.efficientnet_b0(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif "efficientnet_b1" in model_name:
        model = models.efficientnet_b1(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif "efficientnet_b2" in model_name:
        model = models.efficientnet_b2(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError("Invalid model name")
   
    # Load the trained model weights
    model.load_state_dict(torch.load(os.path.join(directory, f"{model_name}_model.pt"), weights_only=True))

    # link the model to the CPU or CUDA
    if link_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    return model, device