import matplotlib.pyplot as plt


def plot_roc(fprs, tprs, dataset_name):
    # plot the roc curve for the model
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    for i in range(0, len(fprs)):
        plt.plot(fprs[i], tprs[i], label='stage'+str(i))        
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.savefig("./logs/"+dataset_name+"/roc.png", bbox_inches = 'tight', pad_inches = 0)
