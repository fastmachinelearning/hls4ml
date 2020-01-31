import importlib
from hls4ml.model.hls_model import HLSModel

libs = [('numpy', 'np'), ('pandas', 'pandas'), ('tensorflow', 'tensorflow'),
        ('seaborn', 'sb'), ('matplotlib.pyplot', 'plt')]
for (name, short) in libs:
    try:
        lib = importlib.import_module(name)
    except ImportError as error:
        print(error)
        print('Install hls4ml[profiling] extra depencies.')
    except Exception as exception:
        print(exception)
    else:
        globals()[short] = lib
globals()['keras'] = tensorflow.keras

def violinplot(data):
    f = plt.figure()
    hue = 'layer' if 'layer' in data.keys() else None
    vp = sb.violinplot(x='x', y='weight', hue=hue, data=data[data['x'] > 0])
    vp.set_yticklabels(vp.get_yticklabels(), rotation=45, ha='right')
    if hue is not None:
        vp.get_legend().remove()
    vp.set_xscale('log', basex=2)
    return f

def boxplot(data):
    from matplotlib.ticker import MaxNLocator
    f = plt.figure()
    hue = 'layer' if 'layer' in data.keys() else None
    vp = sb.boxplot(x='x', y='weight', hue=hue, data=data[data['x'] > 0], showfliers=False)
    vp.set_yticklabels(vp.get_yticklabels(), rotation=45, ha='right')
    if hue is not None:
        vp.get_legend().remove()
    vp.set_xscale('log', basex=2)
    return f

def histogram(data):
    from matplotlib.ticker import MaxNLocator
    # Power of 2 bins covering data range
    high = np.ceil(np.log2(max(data['x']))) + 1
    low = np.floor(np.log2(min(data[data['x'] > 0]['x']))) - 1
    bits = np.arange(low, high, 1)
    bins = 2 ** bits
    f = plt.figure()
    colors = sb.color_palette("husl", len(data['weight'].unique()))
    for i, weight in enumerate(data['weight'].unique()):
        x = data[data['weight'] == weight]['x']
        h, b = np.histogram(x, bins=bins)
        h = h * 1. / float(sum(h)) # normalize
        plt.bar(bits[:-1], h, width=1, fill=False, label=weight, edgecolor=colors[i])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('log2(x)')
    plt.ylabel('frequency')
    plt.legend()
    return f

def FacetGrid(data):
    hue = 'layer' if 'layer' in data.keys() else None
    vp = sb.FacetGrid(data[data['x'] > 0], row='weight', hue=hue)
    vp.map(sb.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    vp.map(sb.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
    vp.map(plt.axhline, y=0, lw=2, clip_on=False)
    vp.fig.subplots_adjust(hspace=-.25)
    return vp.fig

plots = {'violinplot' : violinplot,
         'boxplot' : boxplot,
         'FacetGrid' : FacetGrid,
         'histogram' : histogram}

def weights_hlsmodel(model):
    data = {'x' : [], 'layer' : [], 'weight' : []}
    for layer in model.get_layers():
        for iw, weight in enumerate(layer.get_weights()):
            w = weight.data.flatten()
            data['x'].extend(abs(w).tolist())
            data['layer'].extend([layer.name for i in range(len(w))])
            data['weight'].extend(['{}/{}'.format(layer.name, iw) for i in range(len(w))])

    data = pandas.DataFrame(data)
    return data

def weights_keras(model):
    data = {'x' : [], 'layer' : [], 'weight' : []}
    for layer in model.layers:
        name = layer.name
        weights = layer.get_weights()
        for i, w in enumerate(weights):
            w = w.flatten()
            n = len(w)
            data['x'].extend(abs(w).tolist())
            data['layer'].extend([name for j in range(n)])
            data['weight'].extend(['{}/{}'.format(name, i) for j in range(n)])

    data = pandas.DataFrame(data)
    return data

def types_boxplot(data):
    from matplotlib.patches import PathPatch
    from matplotlib.patches import Rectangle
    ax = plt.gca()
    f = plt.gcf()
    # Scale the data
    data['low'] = 2.**data['low']
    data['high'] = 2.**data['high']

    # Plot the custom precisions
    ticks = np.array([tick.get_text() for tick in plt.yticks()[1]])
    # Get the coordinates of the boxes to place the markers
    boxes = [c.get_extents().inverse_transformed(ax.transData) for c in ax.get_children() if isinstance(c, PathPatch)]
    ys = [(box.y0 + box.y1) / 2 for box in boxes]
    ys = [(y, y) for y in ys]
    for irow, row in data[data['layer'] != 'model'].iterrows():
        if row['layer'] in ticks:
            iy = np.argwhere(ticks == row['layer'])[0][0] # Determine which layer in the plot
            rectangle = Rectangle((row['low'], ys[iy][0]-0.4), row['high']-row['low'], 0.8, fill=True, color='grey', alpha=0.2)
            ax.add_patch(rectangle)

def types_histogram(data):
    ax = plt.gca()
    layers = np.array(ax.get_legend_handles_labels()[1])
    colors = sb.color_palette("husl", len(layers))
    ylim = ax.get_ylim()
    for irow, row in data[data['layer'] != 'model'].iterrows():
        if row['layer'] in layers:
            col = colors[np.argwhere(layers == row['layer'])[0][0]]
            plt.plot((row['low'], row['low']), ylim, '--', color=col)
            plt.plot((row['high'], row['high']), ylim, '--', color=col)

types_plots = {'boxplot' : types_boxplot,
               'histogram' : types_histogram}

def ap_fixed_WIF(type_str):
    if 'ap_fixed' in type_str:
        W = int(type_str.split(',')[0].split('<')[1])
        I = int(type_str.split(',')[1].split('>')[0])
        F = W - I
    elif 'ap_int' in type_str:
        W = int(type_str.replace('ap_int<','').replace('>',''))
        I = W
        F = 0
    else:
        W, I, F = 0, 0, 0
    return W, I, F

def types_hlsmodel(model):
    data = {'layer' : [], 'low' : [], 'high' : []}
    # Plot the default precision
    default_precision = model.config.model_precision['default']
    # assumes ap_fixed
    W, I, F = ap_fixed_WIF(default_precision)
    data['layer'].append('model')
    data['low'].append(-F)
    data['high'].append(I-1)

    for layer in model.get_layers():
        for iw, weight in enumerate(layer.get_weights()):
            wname = '{}/{}'.format(layer.name, iw)
            T = weight.type
            if T.name != 'model':
                W, I, F = ap_fixed_WIF(T.precision)
                data['layer'].append(wname)
                data['low'].append(-F)
                data['high'].append(I-1)
    data = pandas.DataFrame(data)
    return data

def activation_types_hlsmodel(model):
    data = {'layer' : [], 'low' : [], 'high' : []}
    # Get the default precision
    default_precision = model.config.model_precision['default']
    W, I, F = ap_fixed_WIF(default_precision)
    data['layer'].append('model')
    data['low'].append(-F)
    data['high'].append(I-1)
    for layer in model.get_layers():
        T = layer.get_output_variable().type.precision
        W, I, F = ap_fixed_WIF(T)
        data['layer'].append(layer.name)
        data['low'].append(-F)
        data['high'].append(I-1)
    data = pandas.DataFrame(data)
    return data

def activations_keras(model, X):
    # test layer by layer on data
    data = {'x' : [], 'weight' : []}
    partial_model = keras.models.Sequential()
    for layer in model.layers:
        print("   {}".format(layer.name))
        partial_model.add(layer)
        partial_model.compile(optimizer='adam', loss='mse')
        if not isinstance(layer, keras.layers.InputLayer):
            y = partial_model.predict(X).flatten()
            data['x'].extend(abs(y).tolist())
            data['weight'].extend([layer.name for i in range(len(y))])

    data = pandas.DataFrame(data)
    return data

def numerical(keras_model=None, hlsmodel=None, X=None, plot='boxplot'):
    """
    Perform numerical profiling of a model

    Parameters
    ----------
    model : keras model
        The keras model to profile
    X : array-like, optional
        Test data on which to evaluate the model to profile activations
        Must be formatted suitably for the model.predict(X) method
    plot : str, optional
        The type of plot to produce.
        Options are: 'boxplot' (default), 'violinplot', 'histogram', 'FacetGrid'

    Returns
    -------
    tuple
        The pair of produced figures. First weights and biases, then activations
    """

    print("Profiling weights")
    if hlsmodel is not None and isinstance(hlsmodel, HLSModel):
        data = weights_hlsmodel(hlsmodel)
    elif keras_model is not None and isinstance(keras_model, keras.Model):
        data = weights_keras(keras_model)
    else:
        print("Only keras and HLSModel models can currently be profiled")
        return False, False

    wp = plots[plot](data) # weight plot
    if isinstance(hlsmodel, HLSModel) and plot in types_plots:
        t_data = types_hlsmodel(hlsmodel)
        types_plots[plot](t_data)

    plt.title("Distribution of (non-zero) weights")
    plt.tight_layout()

    ap = None
    if X is not None and isinstance(keras_model, keras.Model):
        print("Profiling activations")
        data = activations_keras(keras_model, X)
        ap = plots[plot](data) # activation plot
        plt.title("Distribution of (non-zero) activations")
        plt.tight_layout()

    if X is not None and isinstance(hlsmodel, HLSModel):
        t_data = activation_types_hlsmodel(hlsmodel)
        types_plots[plot](t_data)

    return wp, ap

