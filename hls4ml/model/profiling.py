import numpy as np
import pandas
import keras
import seaborn as sb
import matplotlib.pyplot as plt

def violinplot(data):
    vp = sb.violinplot(x='x', y='weight', hue='layer', data=data)
    vp.set_yticklabels(vp.get_yticklabels(), rotation=45, ha='right')
    vp.get_legend().remove()
    vp.set_xscale('log', basex=2)
    plt.title('Distribution of weights')
    plt.tight_layout()
    return vp

def boxplot(data):
    vp = sb.boxplot(x='x', y='weight', hue='layer', data=data, showfliers=False)
    vp.set_yticklabels(vp.get_yticklabels(), rotation=45, ha='right')
    vp.get_legend().remove()
    vp.set_xscale('log', basex=2)
    plt.title('Distribution of weights')
    plt.tight_layout()
    return vp

def FacetGrid(data):
    vp = sb.FacetGrid(data, row='weight', hue='layer')
    vp.map(sb.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    vp.map(sb.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
    vp.map(plt.axhline, y=0, lw=2, clip_on=False)
    vp.fig.subplots_adjust(hspace=-.25)
    return vp

def numerical(model, X=None, plot='boxplot'):
    data = {'x' : [], 'layer' : [], 'weight' : []}
    print("Profiling weights")
    for layer in model.layers:
        name = layer.name
        weights = layer.get_weights()
        for i, w in enumerate(weights):
            w = w.flatten()
            n = len(w)
            data['x'].extend(abs(w).tolist())
            data['layer'].extend([name for j in range(n)])
            #data['weight'].extend([i for j in range(len(w))])
            data['weight'].extend(['{}/{}'.format(name, i) for j in range(n)])

    data = pandas.DataFrame(data)

    plots = {'violinplot' : violinplot,
             'boxplot' : boxplot,
             'FacetGrid' : FacetGrid}

    wp = plots[plot](data) # weight plot

    ap = None
    if X is not None:
        print("Profiling activations")
        # test layer by layer on data
        data = {'x' : [], 'layer' : []}
        partial_model = keras.models.Sequential()
        for layer in model.layers:
            print("   {}".format(layer.name))
            partial_model.add(layer)
            partial_model.compile(optimizer='adam', loss='mse')
            if not isinstance(layer, keras.layers.InputLayer):
                y = partial_model.predict(X).flatten()
                data['x'].extend(abs(y).tolist())
                data['layer'].extend([layer.name for i in range(len(y))])

        plt.figure()
        data = pandas.DataFrame(data)
        ap = sb.violinplot(x='x', y='layer', data=data)
        ap.set_yticklabels(ap.get_yticklabels(), rotation=45, ha='right')
        ap.set_xscale('log', basex=2)
        plt.title('Distribution of activations')

    return wp, ap

