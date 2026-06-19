import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []

    for x, y in loader:
        x = x.to(device)
        z = model.encode(x).detach().cpu()
        features.append(z)
        labels.append(y.cpu())

    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def linear_probe_eval(
    model,
    train_loader,
    test_loader,
    device,
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
):
    x_train, y_train = extract_features(model, train_loader, device)
    x_test, y_test = extract_features(model, test_loader, device)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver=solver,
        ),
    )

    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    return accuracy_score(y_test, preds)