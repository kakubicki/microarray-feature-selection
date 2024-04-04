import pickle
import glob
import os
from pandas import DataFrame as df
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np
import re

basepath = os.path.abspath(os.path.dirname(__file__))

frames = []
files = glob.glob(os.path.join(basepath,'outputs/wrapper_df/*.pkl'))
for f in files:
    with open(f,'rb') as file:
        frames.append(pickle.load(file))

frame = df.from_records(frames)

print(frame)

grouped = frame.groupby(['Dataset','Feature Selection method'])
particles = sorted([100,   50,  200,   10,  500,   20, 1000, 2000])
iterations = sorted([2000,   50, 1000,  500,  100,  200])
req = len(particles)*len(iterations)*4

fig, ax = plt.subplots(6,2,figsize=(10,30))
ax = ax.ravel()
sum = 0
for i,group in enumerate(grouped.groups):
    g = grouped.get_group(group)
    ax[i].plot(g['No. particles'],g['No. iterations'],'g.')
    perc = len(g.index)/req*100
    ax[i].set_title(f"{group[0]}\n{group[1]} [{perc:.1f}% ready]")
    ax[i].set_ylabel('iterations')
    ax[i].set_xlabel('particles')
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    n = set(zip(g['No. particles'],g['No. iterations']))
    n1 = set(itertools.product(particles,iterations))
    n1 = n1 - n
    ax[i].plot(*zip(*list(n1)),'rx')
    print(f"{group[0]} - {group[1]} : {perc:.1f}% ready, {len(g.index)//4} of {req//4}")
    sum = sum + len(g.index)

ll = len(grouped.groups)
print(f"Total: {100*sum/(ll*req):.1f}%, {sum//4} of {(ll*req)//4}")
fig.tight_layout()
fig.savefig(os.path.join(basepath,'res.png'),dpi=300)

# results
for method in frame['Classification method'].unique():    
    #field = 'Accuracy'
    #data = g['Accuracy']
    #field = 'Percentage of genes selected'
    #data = g['No. selected genes']/g['No. genes']*100.0
    filtered = frame[frame['Classification method'] == method]
    grouped = filtered.groupby(['Dataset','Feature Selection method'])
    fig = plt.figure(figsize=(10,30))
    for i,group in enumerate(grouped.groups):
        g = grouped.get_group(group)
        perc = len(g.index)*4/req*100
        mm = (g['No. genes']/g['No. selected genes']*g['Accuracy']).max()
        colors = plt.cm.jet(g['No. genes']/g['No. selected genes']*g['Accuracy']/mm)        
        ax = fig.add_subplot(6, 2, i+1, projection='3d')
        a = list(zip(np.log10(g['No. particles']),np.log10(g['No. iterations']), iter(int,1)))
        x,y,z = zip(*a)
        ax.bar3d(x,y,z, .1,.1,g['No. genes']/g['No. selected genes']*g['Accuracy']*(100.0/mm), color=colors)
        ax.set_title(f"{group[0]}\n{group[1]} [{perc:.1f}% ready]")
        ax.set_ylabel('log10(iterations)')
        ax.set_xlabel('log10(particles)')
    fig.suptitle(f"Classification method: {method}\n1/Percentage of genes selected * Accuracy [%]")
    #fig.tight_layout()
    fn = ("".join([c for c in method if re.match(r'\w', c)])).lower()
    fig.savefig(os.path.join(basepath,f'res_percacc_{fn}.png'),dpi=300)


# results
for method in frame['Classification method'].unique():
    filtered = frame[frame['Classification method'] == method]
    grouped = filtered.groupby(['Dataset','Feature Selection method'])
    fig = plt.figure(figsize=(10,30))
    for i,group in enumerate(grouped.groups):
        g = grouped.get_group(group)
        perc = len(g.index)*4/req*100
        colors = plt.cm.jet(g['No. selected genes']/g['No. genes'])        
        ax = fig.add_subplot(6, 2, i+1, projection='3d')
        a = list(zip(np.log10(g['No. particles']),np.log10(g['No. iterations']), iter(int,1)))
        x,y,z = zip(*a)
        ax.bar3d(x,y,z, .1,.1,g['No. selected genes']/g['No. genes']*100.0, color=colors)
        ax.set_title(f"{group[0]}\n{group[1]} [{perc:.1f}% ready]")
        ax.set_ylabel('log10(iterations)')
        ax.set_xlabel('log10(particles)')
    fig.suptitle(f"Classification method: {method}\nPercentage of genes selected")
    #fig.tight_layout()
    fn = ("".join([c for c in method if re.match(r'\w', c)])).lower()
    fig.savefig(os.path.join(basepath,f'res_perc_{fn}.png'),dpi=300)

for method in frame['Classification method'].unique():
    filtered = frame[frame['Classification method'] == method]
    grouped = filtered.groupby(['Dataset','Feature Selection method'])
    fig = plt.figure(figsize=(10,30))
    for i,group in enumerate(grouped.groups):
        g = grouped.get_group(group)
        perc = len(g.index)*4/req*100
        colors = plt.cm.jet(g['Accuracy'])        
        ax = fig.add_subplot(6, 2, i+1, projection='3d')
        a = list(zip(np.log10(g['No. particles']),np.log10(g['No. iterations']), iter(int,1)))
        x,y,z = zip(*a)
        ax.bar3d(x,y,z, .1,.1,g['Accuracy']*100.0, color=colors)
        ax.set_title(f"{group[0]}\n{group[1]} [{perc:.1f}% ready]")
        ax.set_ylabel('log10(iterations)')
        ax.set_xlabel('log10(particles)')
    fig.suptitle(f"Classification method: {method}\nAccuracy")
    #fig.tight_layout()
    fn = ("".join([c for c in method if re.match(r'\w', c)])).lower()
    fig.savefig(os.path.join(basepath,f'res_acc_{fn}.png'),dpi=300)