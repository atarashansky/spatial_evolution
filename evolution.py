from __future__ import print_function
from scipy import ndimage
from functools import reduce
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
import scipy.sparse as sp
from PIL import Image
import pickle
import time;
import matplotlib.animation as manimation
from matplotlib.ticker import FormatStrFormatter
import errno

def find_boundary(im):
    return np.argmax(np.sum(np.abs(np.diff(im,axis=1)),axis=0))
    
def data_visualization(tree_info,all_nodes,colorsx,sbins,threshold=0.2,height=400,sample=250,deltat=1200,do_plot=True,GG=1e5,WIDTH=800):
    print('Execution started 1');
    
    
    
    
    width=int(tree_info.shape[2])   
    if(WIDTH < tree_info.shape[2]):
        width=WIDTH
    color_map = np.zeros([height,width])-100;
    
    all_nodes=np.array([])
    all_parent_nodes=np.array([],dtype=object)
    
    
    for i in range(tree_info.shape[2]):
        ti=tree_info[:,:,i]        
        nodes=ti[:,0].astype(np.int64)
        ind=np.where(nodes!=-1)[0]
        nodes = nodes[ind]
        sh = ti[:,2][ind]
        all_nodes = np.append(all_nodes,nodes)
        
        all_parent_nodes=np.append(all_parent_nodes,sh)
        [all_nodes,uniq_ind]=np.unique(all_nodes,return_index=True);
        all_parent_nodes = all_parent_nodes[uniq_ind]
    
    abundances=np.zeros([all_nodes.size,width]) 
    
    print('Execution started 2');

    for i in range(width):
        ti=tree_info[:,:,i]    
        nodes=ti[:,0].astype(np.int64)
        ind=np.where(nodes!=-1)[0]
        nodes = nodes[ind]
        tnc = ti[:,1].astype(np.int64)[ind]
        tn = ti[:,3].astype(np.int64)[ind]
        tn = tn/tnc[0]    
        tnc=tnc/tnc[0]
        sh = ti[:,2][ind]
        
        ind_node=np.in1d(all_nodes,nodes)
        abundances[ind_node,i]=tnc        


    ninclude=np.zeros(all_nodes.size).astype('bool')
    th=np.max(abundances,axis=1)>=threshold
    ninclude[th]=True;
    
    colorsx=colorsx[ninclude,:]
    sbins=sbins[np.append(ninclude,True)]
    all_nodes = all_nodes[ninclude]  
    all_parent_nodes=all_parent_nodes[ninclude]    

    abundances= abundances[ninclude,:]
    clone_heights = np.zeros(all_nodes.size)-1;
    clone_heights[0]=height;
    mmax=[]
    print('Execution started 3');  
    for i in range(width):
        sh=tree_info[:,2,i]
        nodes=tree_info[:,0,i].astype(np.int64)
        ind=np.where(nodes!=-1)[0]
        nodes=nodes[ind]
        sh=sh[ind]
        maxsize=[]
        
        inc=np.in1d(nodes,all_nodes)
        nodes=nodes[inc]  
        sh = sh[inc]
        
        for j in range(nodes.size):
            maxsize.append(sh[j].size)
        
        mmax.append(max(maxsize))
            
        
    print('Printing...');
    
    for i in range(width):
        if(i%10==0):
            print(i)
        ti=tree_info[:,:,i]    
        nodes=ti[:,0].astype(np.int64)
        ind=np.where(nodes!=-1)[0]
        nodes=nodes[ind]
        tnc = ti[:,1].astype(np.int64)[ind]
        tnc=np.round(tnc/tnc[0]*height)
        sh = ti[:,2][ind]
        
        #denoising
        inc=np.in1d(nodes,all_nodes)
        nodes=nodes[inc]  
        sh = sh[inc]
        tnc=tnc[inc]
        
    
        m=mmax[i]
    
    
        shm=np.zeros([sh.size,m])-1;
        for j in range(nodes.size):
            shm[j,0:sh[j].size]=sh[j]   
            
        node_gen=[];    
        tnc_gen=[]
        for j in range(0,m):
            temp=np.unique(shm[:,j])
            temp=temp[temp!=-1]
            node_gen.append(temp)
            tnc_gen.append(tnc[np.in1d(nodes,temp)])
            
        parent_gen=[];    
        for j in range(0,m):
            pg=np.array([])
            for k in range(node_gen[j].size):
                if(node_gen[j][k]==0):
                    pg = np.append(pg,-1)
                else:
                    [rows,cols]=np.where(shm==node_gen[j][k])
                    pg = np.append(pg,shm[rows[0],cols[0]-1])
                    
            parent_gen.append(pg)
                
        starting_coords=[np.array([0])]
        ending_coords=[np.array([height])]
                       
        if(m>GG):
            m=GG;
        for j in range(1,m):
            sc=np.zeros(node_gen[j].size)-10
            ec=np.zeros(node_gen[j].size)-10
            
            x=np.unique(parent_gen[j])
            
            for l in x:
                ind=np.where(parent_gen[j]==l)[0]
                ind2=np.where(node_gen[j-1]==l)[0]
                sil_width = np.sum(tnc_gen[j][ind]) 
                starting_position = starting_coords[-1][ind2]
                ending_position = ending_coords[-1][ind2]
                
                bp_sil=int((starting_position+ending_position)/2)
                sc_sil = bp_sil-int(sil_width/2)
                ec_sil = bp_sil+int(sil_width/2)
                if(sc_sil<0):
                    sc_sil=0
                if(ec_sil>height):
                    ec_sil=height
                for k in range(ind.size):
                    if(k == 0):
                        #print('dummy')
                        sc[ind[k]] = sc_sil
                        ec[ind[k]] = sc_sil+tnc_gen[j][ind[k]]
                    else:
                        sc[ind[k]]=ec[ind[k-1]]
                        ec[ind[k]]=sc[ind[k]]+tnc_gen[j][ind[k]]
            starting_coords.append(sc)
            ending_coords.append(ec)
        for j in range(0,m):
            for k in range(node_gen[j].size):
                color_map[int(starting_coords[j][k]):int(ending_coords[j][k]),i]=node_gen[j][k]
        
    if(do_plot==True):
        plt.subplots(figsize=(16,16*height/width))
        cmap = mpl.colors.ListedColormap(colorsx)
        norm = mpl.colors.BoundaryNorm(sbins, cmap.N)
        ax=plt.subplot(111)
        plt.imshow(color_map,cmap=cmap,norm=norm)
        plt.ylabel('Abundance')
        plt.xlabel('Time (yrs)')
        plt.yticks([])
        #ax.set_xticks(nums)
        #ax.set_xticklabels(np.floor(nums*sample*deltat/3600/24/365.25).astype('int64'))

    
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
            
def evolve(input_image,dirname,name,dt=300*8,numgen=200000,make_movies=True,sb=0.1,sd=0.01,mud=0.1,mub=1e-5,sampling=250,avg=1.16e-5):
    make_sure_path_exists(dirname)
    
    NR=8 #Number of rows
    
    avgD = avg*dt #average death rate per frame of simulation
    
    SHbank=np.zeros(int(1e6),dtype=object) #bank of mutation ancestries
    SHIbank=np.zeros(int(1e6),dtype=object) #bank of repeated copies of most recent mutation IDs in each ancestry, used for constructing lineage tree
    
    
    cells_init=np.array(Image.open(input_image).convert("L")).astype('int64')/255-1 #input geometry
    
    cells_init[cells_init>-1]=0                
    cells_init[cells_init==0]=1;
    
    cells_init[0,:]=-1
    cells_init[:,0]=-1
    
    cells_init[-1,:]=-1
    cells_init[:,-1]=-1
    
    
    cells=np.zeros([cells_init.shape[0],cells_init.shape[1],NR])-1;
    cells[:,:,0]=cells_init
    cells[cells_init==1,1::]=0
    cells[cells_init==1,1]=1
    cells[cells_init==1,2]=1
    cells[cells_init==1,5]=0
    cells[cells_init==1,6]=-5
    cells[cells_init==1,7]=0
    
    """ 
    THE DATA STRUCTURE (CELLS) IS AS FOLLOWS:
    dimensions 1 and 2: 2D spatial map of cell locations
    dimension 3:
    0, obsolete (no idea what I used this for to be honest)
    1, cell fitness
    2, ?!?! no idea what I used this for
    3, number deleterious mutations
    4, number beneficial mutations
    5, mutation ID
    6, death time
    7, obsolete -_-
    """
    acv = np.where(cells_init==1) #cell locations
    N = acv[0].size
    f=0; #simulation frame counter
    finished=False; #flag for whether sim is finished or not
    unique_identifier = 1 #the next mutation ID
    SHbank[0]=np.array([0])
    SHIbank[0]=np.array([0])
    uqis = 0
    
    ims3=[] #contains fitness snapshots of each cell for each frame
    ims4=[] #contains mutation ID snapshots for each cell in each frame
    
    
    cells[acv[0],acv[1],6] = np.floor(f+-1/avgD*np.log(1-np.random.rand(acv[0].size))).astype('int64') #sample the time to death for each cell from an exponential distribution, assuming a poissonian death process

    die_index=np.array([]) #location of dead cells
    
    saved_times=cells[acv[0],acv[1],6] #data structure used to speed up computation
    while(finished == False):       
        
        N = acv[0].size
        if(die_index.size>0):    #if cells died
            random_numbers=(f+np.floor(-1/avgD*np.log(1-np.random.rand(die_index.size)))).astype('int64') #generate new death times for the dead (soon-to-be replaced) cells
            cells[acv[0][die_index],acv[1][die_index],6] = random_numbers #store in data structure
            saved_times[die_index] = random_numbers #update the death times data structure
        
        
        died = np.where(saved_times==f)[0] #if death time = current frame of simulation, kill cells.
        die_index=died.copy() #no reason to copy this...   
        cells[acv[0][died],acv[1][died],1]=-1 # set fitness of dead cells to -1
    
        ncv = (acv[0][died],acv[1][died]) #spatial coordinates of dead cells
        ncv2 = np.vstack(ncv).T #stack 'em into an Nx2 matrix
    
        if(ncv2.size>0):
            x=np.hstack((ncv2-[1,1],ncv2-[1,0],ncv2-[1,-1],ncv2-[0,-1],ncv2-[-1,-1],ncv2-[-1,0],ncv2-[-1,1],ncv2-[0,1])) #get coordinates of 8 positions around each dead cell
    
            xx=np.reshape(cells[x[:,0::2].flatten(),x[:,1::2].flatten(),1],(ncv2.shape[0],8)) #index cell fitnesses of all surrounding positions and reshape into an N x 8 matrix
            xx[xx==-1]=0 #set fitnesses of surrounding position that = -1 (indicating it is a dead cell) to 0
            inc=np.where(np.invert(np.all(xx == 0, axis=1)))[0] #get rid of rows that only have dead cells in the surrounding neighborhood
            z=ncv2[inc] #target of division index
    
            x=x[inc,:]
            xx=xx[inc,:]
            idx_not=np.arange(z.shape[0])
            source=np.zeros(z.shape,dtype='int64')

            divided_neighbors=np.sum(np.tile(np.random.rand(idx_not.size),(xx.shape[1],1)).T>=np.cumsum(xx[idx_not,:]/np.sum(xx[idx_not,:],axis=1)[:,None],axis=1),axis=1) #oh god. this line chooses which of the 8 surrounding neighbors actually divides into the empty spot
            dn_index=np.vstack((divided_neighbors*2,divided_neighbors*2+1)).T #generates index to recover the spatial coordinates of divided neighbors
            stemp=x[np.repeat(idx_not,2),dn_index.flatten()]
            source[idx_not,:]=np.reshape(stemp,(idx_not.size,2)).astype('int64') #source of division index
        
            cells[z[:,0],z[:,1],0:NR-1]=cells[source[:,0],source[:,1],0:NR-1].copy()
    
            del_mutate_s = np.where(np.random.uniform(0,1,source.shape[0])<=0.5*mud)[0] #of the divided cells, sample which ones mutate
            del_mutate_d = np.where(np.random.uniform(0,1,z.shape[0])<=0.5*mud)[0]
            ben_mutate_s = np.where(np.random.uniform(0,1,source.shape[0])<=0.5*mub)[0]
            ben_mutate_d = np.where(np.random.uniform(0,1,z.shape[0])<=0.5*mub)[0]            
        
           
            cells[z[del_mutate_d,0],z[del_mutate_d,1],1]*=(1-sd)#np.random.exponential(sd,del_mutate_d.size))
            cells[z[del_mutate_d,0],z[del_mutate_d,1],3]+=1
        
            cells[source[del_mutate_s,0],source[del_mutate_s,1],1]*=(1-sd)#np.random.exponential(sd,del_mutate_s.size))
            cells[source[del_mutate_s,0],source[del_mutate_s,1],3]+=1
        
            cells[z[ben_mutate_d,0],z[ben_mutate_d,1],1]*=(1+sb)#np.random.exponential(sb,ben_mutate_d.size))
            cells[z[ben_mutate_d,0],z[ben_mutate_d,1],4]+=1
        
            cells[source[ben_mutate_s,0],source[ben_mutate_s,1],1]*=(1+sb)#np.random.exponential(sb,ben_mutate_s.size))
            cells[source[ben_mutate_s,0],source[ben_mutate_s,1],4]+=1
            
            for i in range(ben_mutate_d.size):
                SHbank[unique_identifier-uqis]=np.append(SHbank[int(cells[z[ben_mutate_d[i],0],z[ben_mutate_d[i],1],5]-uqis)],unique_identifier)
                SHIbank[unique_identifier-uqis]=np.ones(SHbank[unique_identifier-uqis].size)*unique_identifier
                cells[z[ben_mutate_d[i],0],z[ben_mutate_d[i],1],5]=unique_identifier
                unique_identifier+=1
        
        
            for i in range(ben_mutate_s.size):
                SHbank[unique_identifier-uqis]=np.append(SHbank[int(cells[source[ben_mutate_s[i],0],source[ben_mutate_s[i],1],5]-uqis)],unique_identifier)
                SHIbank[unique_identifier-uqis]=np.ones(SHbank[unique_identifier-uqis].size)*unique_identifier
                cells[source[ben_mutate_s[i],0],source[ben_mutate_s[i],1],5]=unique_identifier                                    
                unique_identifier+=1
                        
        f+=1
        
        if(f%sampling==0):
    
            ims3.append(cells[:,:,1].copy())
            ims4.append(cells[:,:,5].copy())
            print(f)
        
        if(f>numgen):#
            finished=True; 
    
    print(str(f*20*60/3600/24/30) + ' months')
    
    SHbank=SHbank[0:unique_identifier-uqis]
    SHIbank=SHIbank[0:unique_identifier-uqis]
    
    all_nodes=[]
    node_time=[]
    nodes=[]
    node_counts=[]
    
    for i in range(0,len(ims4)):
        
        if(i%10==0):
            print(i)
        
        a,c=np.unique(ims4[i].flatten(),return_counts=True)     
        nodes.append(a[a>=0])
        node_counts.append(c[a>=0])
        all_nodes.extend(a[a>=0])
    
    [all_nodes,idx]=np.unique(all_nodes,return_index=True)
    all_nodes=all_nodes[all_nodes>=0]
    #all_nodes=np.append(0,all_nodes)
    
    all_nodes=np.unique(np.concatenate(SHbank[all_nodes.astype('int64')-uqis]))                    
    
    
    species_history=SHbank[np.in1d(np.arange(uqis,unique_identifier),all_nodes)]
    species_history_identifier=SHIbank[np.in1d(np.arange(uqis,unique_identifier),all_nodes)]
    
    tree_info = np.zeros([int(all_nodes.size),5,len(ims4)],dtype=object)-1;
    for i in range(len(ims4)):  
        if(i%10==0):
            print(i)
         #for j in range(nodes[i].size):
        xx=np.concatenate(species_history[np.in1d(all_nodes,nodes[i])])
        tree_node=np.unique(xx)
        #tree_node=tree_node[tree_node>0]
        tree_number=np.zeros(tree_node.size)
        tree_number[np.in1d(tree_node,nodes[i])]=node_counts[i]
        xy=np.concatenate(species_history_identifier[np.in1d(all_nodes,nodes[i])])
        [_,invxy]=np.unique(xy,return_inverse=True)
        [_,invxx]=np.unique(xx,return_inverse=True)
        tree_number_children=np.array(sp.coo_matrix((tree_number[np.in1d(tree_node,nodes[i])][invxy],(invxy,invxx))).sum(axis=0)).flatten()   
        tree_info[0:tree_node.size,0,i]=tree_node;
        tree_info[0:tree_number.size,1,i]=tree_number_children;
        tree_info[0:tree_node.size,2,i]=species_history[np.in1d(all_nodes,tree_node)]
        tree_info[0:tree_node.size,3,i]=tree_number
    
    sbins=np.append(all_nodes-0.01,all_nodes[-1]+0.01)                            
    colorsx=np.random.rand(all_nodes.size,4)
    colorsx[0,:]=np.array([0.,0.,0.,1])
    colorsx[:,-1]=1.0
    #colorsx[1,:]=np.array([1,0,0,1.])
    #colorsx[2,:]=np.array([0,0,1,1.])
    colorsx[1,:]=np.array([70./255,70./255,70./255,1])
    cmap = mpl.colors.ListedColormap(colorsx)
    norm = mpl.colors.BoundaryNorm(sbins, cmap.N)
    
    data_visualization(tree_info,all_nodes,colorsx,sbins,threshold=0.0,deltat=dt,WIDTH=4000,height=2000//5)
    plt.savefig(dirname+"/"+name+"_muller.png",bbox_inches="tight")                        
    
    
    SPLbank=np.zeros(SHbank.size)                
    for i in range(len(SHbank)):
        if(SHbank[i].size==1):
            SPLbank[i] = SHbank[i][0]
        else:
            SPLbank[i] = SHbank[i][1]
    ims4pl = []
    for i in range(len(ims4)):
        if(i%10==0):
            print(i)
        cmp = ims4[i].copy()
        nodes=(np.unique(cmp)[1::]).astype('int64')
        for j in nodes:
            if(j>0):
                cmp[cmp==j]=SPLbank[j-uqis]
                
        cmp[cmp==-1]=0
        ims4pl.append(cmp)
    
    pickle.dump(ims4pl,open(dirname+"/"+name+"_ims_primarylineage.p","wb"))
    pickle.dump(ims4,open(dirname+"/"+name+"_ims_extant.p","wb"))
    pickle.dump(ims3,open(dirname+"/"+name+"_ims_fitness.p","wb"))
    
    colorsx=np.vstack((np.array([0,0,0,1]),colorsx))
    
    sbins=np.append(-1.01,sbins)
    
    cmap = mpl.colors.ListedColormap(colorsx)
    norm = mpl.colors.BoundaryNorm(sbins, cmap.N)
    
    if(make_movies==True):
        print('MAKING PRIMARY LINEAGES MOVIE...')
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='wowow', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=40, metadata=metadata)
        
        fig,ax = plt.subplots(figsize=(10,10*ims4[0].shape[0]/ims4[0].shape[1]))    
        #t=np.concatenate(ims)
        #cmin=np.min(t[t>0])
        #cmax=np.log(np.max(t))
        #im = plt.imshow(np.log(ims[0]+1),vmin=0,vmax=cmax)
        im = plt.imshow(ims4pl[0],cmap=cmap,norm=norm,  animated=True)
        plt.gca().set_axis_off()
        fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax.axis('tight')
        ax.axis('off')
        plt.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
        with writer.saving(fig, dirname+"/"+name+"_primary.mp4", 100):
            for i in range(len(ims4)):
                if(i%100==0):
                    print(i)
                #im.set_array(np.log(ims[i]+1))
                im.set_array(ims4pl[i])
        
                writer.grab_frame()
        
        print('MAKING EXTANT MUTATIONS MOVIE...')
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='wowow', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=40, metadata=metadata)
        
        fig,ax = plt.subplots(figsize=(10,10*ims4[0].shape[0]/ims4[0].shape[1]))    
    
    
        im = plt.imshow(ims4[0],cmap=cmap,norm=norm,  animated=True)
        plt.gca().set_axis_off()
        fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax.axis('tight')
        ax.axis('off')
        plt.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
        with writer.saving(fig, dirname+"/"+name+"_extant.mp4", 100):
            for i in range(len(ims4)):
                if(i%100==0):
                    print(i)
                #im.set_array(np.log(ims[i]+1))
                im.set_array(ims4[i])
        
                writer.grab_frame()    
                  
