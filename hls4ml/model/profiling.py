import numpy as np
import pandas
import keras
import seaborn as sb
import matplotlib.pyplot as plt

def violinplot(data):
    vp = sb.violinplot(x='weight', y='x', hue='layer', data=data)
    vp.set_xticklabels(vp.get_xticklabels(), rotation=45)
    vp.get_legend().remove()
    vp.set_yscale('log', basey=2)
    return vp

def boxplot(data):
    vp = sb.boxplot(x='weight', y='x', hue='layer', data=data)
    vp.set_xticklabels(vp.get_xticklabels(), rotation=45)
    vp.get_legend().remove()
    vp.set_yscale('log', basey=2)
    return vp

def FacetGrid(data):
    vp = sb.FacetGrid(data, row='weight', hue='layer')
    vp.map(sb.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    vp.map(sb.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
    vp.map(plt.axhline, y=0, lw=2, clip_on=False)
    vp.fig.subplots_adjust(hspace=-.25)
    return vp

def numerical(model, X=None, plot='violinplot'):
    data = {'x' : [], 'layer' : [], 'weight' : []}
    print("Profiling weights")
    for layer in model.layers:
        name = layer.name
        weights = layer.get_weights()
        for i, w in enumerate(weights):
            w = w.flatten()
            n = len(w)
            data['x'].extend(w.tolist())
            data['layer'].extend([name for j in range(n)])
            #data['weight'].extend([i for j in range(len(w))])
            data['weight'].extend(['{}/{}'.format(name, i) for j in range(n)])

    data = pandas.DataFrame(data)

    plots = {'violinplot' : violinplot,
             'boxplot' : boxplot,
             'FacetGrid' : FacetGrid}

    vp = plots[plot](data)

    act_plot = None
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
                data['x'].extend(y.tolist())
                data['layer'].extend([layer.name for i in range(len(y))])

        plt.figure()
        data = pandas.DataFrame(data)
        act_plot = sb.violinplot(x='layer', y='x', data=data)
        act_plot.set_xticklabels(act_plot.get_xticklabels(), rotation=45)
        act_plot.set_yscale('log', basey=2)

    return vp, act_plot

