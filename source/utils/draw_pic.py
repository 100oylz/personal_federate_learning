import matplotlib.pyplot as plt
import pandas as pd


def draw_line_chart(x,y,title,save_path,is_show=False):
    plt.plot(x,y)
    plt.title(title)
    plt.savefig(save_path)
    if(is_show):
        plt.show()

def draw_line_chart_from_csv(filepath,num_clients,title,save_path,is_show=False):
    df=pd.read_csv(filepath)
    x_list=[]
    y_list=[]
    num_epochs=df['epoch'].max()+1
    for i in range(num_clients):
        x=[]
        y=[]
        for j in range(num_epochs):
            if(df[f'client_{i}_loss'][j]!=float('nan')):
                x.append(j)
                y.append(df[f'client_{i}_loss'][j])
        x_list.append(x)
        y_list.append(y)
    fig=plt.figure(dpi=300)
    ax=fig.add_subplot(111)
    for i in range(num_clients):
        ax.plot(x_list[i],y_list[i],label=f'client_{i}')

    ax.set_title(title[0])
    ax.legend()
    fig.savefig(save_path[0])
    if(is_show):
        ax.show()
    plt.close(fig)
    x_list=[]
    y_list=[]
    for i in range(num_clients):
        x = []
        y = []
        for j in range(num_epochs):
            if (df[f'client_{i}_acc'][j] != float('nan')):
                x.append(j)
                y.append(df[f'client_{i}_acc'][j])
        x_list.append(x)
        y_list.append(y)
    fig1 = plt.figure(dpi=300)
    ax = fig1.add_subplot(111)
    for i in range(num_clients):
        ax.plot(x_list[i], y_list[i], label=f'client_{i}')

    ax.set_title(title[1])
    ax.legend()
    fig1.savefig(save_path[1])
    if (is_show):
        ax.show()
    plt.close(fig1)



