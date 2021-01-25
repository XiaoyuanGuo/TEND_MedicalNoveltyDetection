import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set2(label[i]/ 2.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def get_tsne(embs, targets, dataset_name, stage=1):
    
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features = embs
    targets = np.array(targets)
    if len(targets.shape) != 1:
        targets = np.squeeze(targets, axis=1)
        
    result = tsne.fit_transform(features)
    fig = plot_embedding(result, targets, '%s_Stage_%d' % (dataset_name, stage))
    fig.savefig("./logs/"+dataset_name+"/stage"+str(stage)+".png", bbox_inches = 'tight', pad_inches = 0)
