#prettyPCA


"""
This functions plots more interesting plot of PCA reduced data in 2d.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def pretty_pca( skPCA, transformed_data, var_names = None, fraction_data = 1., scale = 3, scatter_color = None):
    """
    skPCA: a sklearn-fitted PCA instance.
    transformed_data: the pca-reduced data. 
    var_names: the variable names; defaults to numbers starting at 0.
    fraction_data: fraction of data points to plot.
    scale: how much to scale the lines by, default 3.
    
    """
    line_color = "k"
    transformed_data = transformed_data[::int(1.0/fraction_data),:2]
    components = skPCA.components_[:2,:].T
    n_features = components.shape[0]
    if var_names == None:
        var_names = [ "%d"%i for i in range(n_features) ]
    else:
        var_names = [ "%s, %d"%(name, i) for i,name in enumerate(var_names) ]

    
    fig = plt.figure(1,figsize=(8,5))
    gs = gridspec.GridSpec( 2, 1, height_ratios=[3,1] )
    
    ax = plt.subplot( gs[0] )
    if scatter_color is not None:
        ax.scatter( transformed_data[:,0], transformed_data[:,1], edgecolors='none', alpha = 0.6, c = scatter_color )
    else:
        ax.scatter( transformed_data[:,0], transformed_data[:,1], edgecolors='none', alpha = 0.5 )

    ax.scatter( [0], [0], s = 5, c = "k" )
    for i in range( n_features ):
        #ax.plot( *zip([0,0], scale*components[i,:]) , c = line_color, lw = 2, alpha = 0.8 )
        ax.annotate( "", scale*components[i, :], (0,0), arrowprops = dict( arrowstyle="->"))
        ax.annotate(var_names[i], xy=scale*components[i,:],  xycoords='data',
                #xytext=(-50, 30),
                textcoords='offset points',
                size = 12,
                )
    ax.set_title("2 Dimensional PCA data")
        
    ax = plt.subplot( gs[1] )
    
    ax.bar( range(skPCA.explained_variance_ratio_.shape[0]), skPCA.explained_variance_ratio_ )
    ax.bar( range(2), skPCA.explained_variance_ratio_[:2], color = "r" )
    ax.set_title( "Explained variance ratio" )
    plt.show()
    return
    