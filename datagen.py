import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

def gaussian(mean1=[-2,-2],cov1=[[1,0],[0,1]],mean2=[2,2],cov2=[[1,0],[0,1]],spc=500):
    gauss_1 = np.random.multivariate_normal(mean=mean1,cov=cov1,size=spc)
    gauss_2 = np.random.multivariate_normal(mean=mean2,cov=cov2,size=spc)
    gauss = np.concatenate([gauss_1,gauss_2])
    label = np.concatenate([np.zeros(spc),np.ones(spc)]).astype(int)
    return gauss,label

def xor(spc=500,xor_shift=3):
    n_xor = int(spc/2)
    xor_1_l = np.random.uniform(-2.5,2.5,(n_xor,2))+np.array([-xor_shift,xor_shift])
    xor_1_r = np.random.uniform(-2.5,2.5,(n_xor,2))+np.array([xor_shift,-xor_shift])

    xor_2_l = np.random.uniform(-2.5,2.5,(n_xor,2))+np.array([xor_shift,xor_shift])
    xor_2_r = np.random.uniform(-2.5,2.5,(n_xor,2))+np.array([-xor_shift,-xor_shift])

    xor = np.concatenate([xor_1_l,xor_1_r,xor_2_l,xor_2_r])
    label = np.concatenate([np.zeros(spc),np.ones(spc)]).astype(int)
    return xor,label

def circular(spc=500):
    sample_per_class = spc
    cir_1_angle = np.random.uniform(0,2*np.pi,[sample_per_class,1])
    cir_1_r = np.random.uniform(0,2,[sample_per_class,2])
    cir_1 = cir_1_r*np.hstack((np.sin(cir_1_angle),np.cos(cir_1_angle)))

    cir2_angle = np.random.uniform(0,2*np.pi,[sample_per_class,1])
    cir_2 = 4*np.hstack((np.sin(cir2_angle),np.cos(cir2_angle))) + np.random.uniform(-1/2,1/2,[500,2])

    circular = np.concatenate([cir_1,cir_2])
    label = np.concatenate([np.zeros(spc),np.ones(spc)]).astype(int)
    return circular,label

def spiral(spc=500):
    sample_per_class = spc
    spical_r = np.linspace(start=0,stop=5,num = sample_per_class).reshape(-1,1)
    spiral_angle = np.linspace(0,3*np.pi,sample_per_class).reshape(-1,1)

    spiral_1 = spical_r*np.hstack((np.sin(spiral_angle),np.cos(spiral_angle)))
    spiral_2 = -1*spiral_1

    spiral = np.concatenate([spiral_1,spiral_2])
    label = np.concatenate([np.zeros(spc),np.ones(spc)]).astype(int)
    return spiral,label


def plot_decisionBoundary(model,data,label,plot_min=-6,plot_max=6):
    xx,yy = np.meshgrid(np.arange(plot_min,plot_max,0.05),np.arange(plot_min,plot_max,0.05))
    pred_db = model.predict(np.c_[xx.ravel(),yy.ravel()])
    colors = ListedColormap(['red','blue'])
    colors_db = ['pink','cyan']
    plt.figure(figsize=(7,7))
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.contourf(xx,yy,pred_db.reshape(xx.shape),cmap=matplotlib.colors.ListedColormap(colors_db))
    scatterplot = plt.scatter(data[:,0],data[:,1],c=label,cmap=colors)
    plt.legend(handles=scatterplot.legend_elements()[0],labels=['class_0','class_1'])

def myplot(data,label,title=None):
    colors = ListedColormap(['red','blue'])
    plt.figure(figsize=(6,6))
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    scatterplot = plt.scatter(data[:,0],data[:,1],c=label,cmap=colors)
    plt.legend(handles=scatterplot.legend_elements()[0],labels=['class_0','class_1'])
    plt.title(title)
    plt.show()


def pred_plot(x,y,pred):
    class_plot = y + ~(pred==y)*2 
    plt.figure(figsize=(6,6))
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    colors_dict = {0:'red',1:'blue',2:'green',3:'orange'}
    colors = ListedColormap(['red','blue','green','orange'])
    scatter = plt.scatter(x[:,0],x[:,1],c=[colors_dict[i] for i in class_plot],cmap=colors,label=class_plot)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='class_0',markerfacecolor='red', markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='class_1',markerfacecolor='blue', markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='class_0 miss',markerfacecolor='green', markersize=7),
                       Line2D([0], [0], marker='o', color='w', label='class_1 miss',markerfacecolor='orange', markersize=7)
                         ]

    plt.legend(handles=legend_elements)
    plt.show()
