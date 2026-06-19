import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    xs, ys = [], []

    for x, y in loader:
        x = x.to(device)
        z = model.encode(x).detach().cpu()
        xs.append(z)
        ys.append(y.cpu())

    return torch.cat(xs).numpy(), torch.cat(ys).numpy()


def knn_eval(model, train_loader, test_loader, device, k=5):
    x_train, y_train = extract_embeddings(model, train_loader, device)
    x_test, y_test = extract_embeddings(model, test_loader, device)

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    return accuracy_score(y_test, preds)