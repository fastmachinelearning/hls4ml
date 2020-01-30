import importlib
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
    f = plt.figure()
    hue = 'layer' if 'layer' in data.keys() else None
    vp = sb.boxplot(x='x', y='weight', hue=hue, data=data[data['x'] > 0], showfliers=False)
    vp.set_yticklabels(vp.get_yticklabels(), rotation=45, ha='right')
    if hue is not None:
        vp.get_legend().remove()
    vp.set_xscale('log', basex=2)
    return f

def histogram(data):
    from cycler import cycler
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

def numerical(model, X=None, plot='boxplot'):
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
    wp = plots[plot](data) # weight plot
    plt.title("Distribution of (non-zero) weights")
    plt.tight_layout()

    ap = None
    if X is not None:
        print("Profiling activations")
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
        ap = plots[plot](data) # activation plot
        plt.title("Distribution of (non-zero) activations")
        plt.tight_layout()
    return wp, ap

