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

def array_to_summary(x, fmt='boxplot'):
    if fmt == 'boxplot':
        y = {'med' : np.median(x),
             'q1' : np.percentile(x, 25),
             'q3' : np.percentile(x, 75),
             'whislo' : min(x),
             'whishi' : max(x)
        }
    elif fmt == 'histogram':
        # Power of 2 bins covering data range
        high = np.ceil(np.log2(max(x))) + 1
        low = np.floor(np.log2(min(x))) - 1
        bits = np.arange(low, high, 1)
        bins = 2 ** bits
        h, b = np.histogram(x, bins=bins)
        h = h * 1. / float(sum(h)) # normalize
        y = {'h' : h,
             'b' : np.log2(b)}
    return y

def boxplot(data, fmt='longform'):
    if fmt == 'longform':
        f = plt.figure() #figsize=(3, 3))
        hue = 'layer' if 'layer' in data.keys() else None
        vp = sb.boxplot(x='x', y='weight', hue=hue, data=data[data['x'] > 0], showfliers=False)
        vp.set_yticklabels(vp.get_yticklabels(), rotation=45, ha='right')
        if hue is not None:
            vp.get_legend().remove()
        vp.set_xscale('log', basex=2)
        return f
    elif fmt == 'summary':
        from matplotlib.patches import Rectangle
        medianprops = dict(linestyle='-', color='k')
        f, ax = plt.subplots(1, 1)
        data.reverse()
        colors = sb.color_palette("Blues", len(data))
        bp = ax.bxp(data, showfliers=False, vert=False, medianprops=medianprops)
        # add colored boxes
        for line, color in zip(bp['boxes'], colors):
            x = line.get_xdata()
            xl, xh = min(x), max(x)
            y = line.get_ydata()
            yl, yh = min(y), max(y)
            rect = Rectangle((xl, yl), (xh-xl), (yh-yl), fill=True, color=color)
            ax.add_patch(rect)
        ax.set_yticklabels([d['weight'] for d in data])
        ax.set_xscale('log', basex=2)
        plt.xlabel('x')
        return f
    else:
        return None

def histogram(data, fmt='longform'):
    f = plt.figure()
    from matplotlib.ticker import MaxNLocator
    n = len(data) if fmt == 'summary' else len(data['weight'].unique())
    colors = sb.color_palette("husl", n)
    if fmt == 'longform':
        for i, weight in enumerate(data['weight'].unique()):
            y = array_to_summary(data[data['weight'] == weight]['x'], fmt='histogram')
            plt.bar(y['b'][:-1], y['h'], width=1, fill=False, label=weight, edgecolor=colors[i])
    elif fmt == 'summary':
        for i, weight in enumerate(data):
            plt.bar(weight['b'][:-1], weight['h'], width=1, fill=False, label=weight['weight'], edgecolor=colors[i])

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('log2(x)')
    plt.ylabel('frequency')
    plt.legend()
    return f

plots = {'boxplot' : boxplot,
         'histogram' : histogram}

def types_boxplot(data, fmt='longform'):
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
    if fmt == 'longform':
        # seaborn adjusts the box positions slightly in groups
        boxes = [c.get_extents().inverse_transformed(ax.transData) for c in ax.get_children() if isinstance(c, PathPatch)]
        ys = [(box.y0 + box.y1) / 2 for box in boxes]
        ys = [(y, y) for y in ys]
    elif fmt == 'summary':
        ys = [(y, y) for y in plt.yticks()[0]]
    for irow, row in data[data['layer'] != 'model'].iterrows():
        if row['layer'] in ticks:
            iy = np.argwhere(ticks == row['layer'])[0][0] # Determine which layer in the plot
            rectangle = Rectangle((row['low'], ys[iy][0]-0.4), row['high']-row['low'], 0.8, fill=True, color='grey', alpha=0.2)
            ax.add_patch(rectangle)

def types_histogram(data, fmt='longform'):
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

def weights_hlsmodel(model, fmt='longform', plot='boxplot'):
    if fmt == 'longform':
        data = {'x' : [], 'layer' : [], 'weight' : []}
    elif fmt == 'summary':
        data = []
    for layer in model.get_layers():
        for iw, weight in enumerate(layer.get_weights()):
            w = weight.data.flatten()
            w = abs(w[w != 0])
            n = len(w)
            if n == 0:
                break
            if fmt == 'longform':
                data['x'].extend(w.tolist())
                data['layer'].extend([layer.name for i in range(len(w))])
                data['weight'].extend(['{}/{}'.format(layer.name, iw) for i in range(len(w))])
            elif fmt == 'summary':
                data.append(array_to_summary(w, fmt=plot))
                data[-1]['layer'] = layer.name
                data[-1]['weight'] = '{}/{}'.format(layer.name, iw)

    if fmt == 'longform':
        data = pandas.DataFrame(data)
    return data

def weights_keras(model, fmt='longform', plot='boxplot'):
    if fmt == 'longform':
        data = {'x' : [], 'layer' : [], 'weight' : []}
    elif fmt == 'summary':
        data = []
    for layer in model.layers:
        name = layer.name
        weights = layer.get_weights()
        for i, w in enumerate(weights):
            w = w.flatten()
            w = abs(w[w != 0])
            n = len(w)
            if n == 0:
                break
            if fmt == 'longform':
                data['x'].extend(w.tolist())
                data['layer'].extend([name for j in range(n)])
                data['weight'].extend(['{}/{}'.format(name, i) for j in range(n)])
            elif fmt == 'summary':
                data.append(array_to_summary(w, fmt=plot))
                data[-1]['layer'] = name
                data[-1]['weight'] = '{}/{}'.format(name, i)

    if fmt == 'longform':
        data = pandas.DataFrame(data)
    return data

def activations_keras(model, X, fmt='longform', plot='boxplot'):
    # test layer by layer on data
    if fmt == 'longform':
        # return long form pandas dataframe for
        # seaborn boxplot
        data = {'x' : [], 'weight' : []}
    elif fmt == 'summary':
        # return summary statistics for matplotlib.axes.Axes.bxp
        # or histogram bin edges and heights
        data = []

    partial_model = keras.models.Sequential()
    for layer in model.layers:
        print("   {}".format(layer.name))
        partial_model.add(layer)
        partial_model.compile(optimizer='adam', loss='mse')
        if not isinstance(layer, keras.layers.InputLayer):
            y = partial_model.predict(X).flatten()
            y = abs(y[y != 0])
            if fmt == 'longform':
                data['x'].extend(y.tolist())
                data['weight'].extend([layer.name for i in range(len(y))])
            elif fmt == 'summary':
                data.append(array_to_summary(y, fmt=plot))
                data[-1]['weight'] = layer.name

    if fmt == 'longform':
        data = pandas.DataFrame(data)
    return data

def numerical(keras_model=None, hls_model=None, X=None, plot='boxplot'):
    """
    Perform numerical profiling of a model

    Parameters
    ----------
    keras_model : keras model
        The keras model to profile
    hls_model : HLSModel
        The HLSModel to profile
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
    if hls_model is not None and isinstance(hls_model, HLSModel):
        data = weights_hlsmodel(hls_model, fmt='summary', plot=plot)
    elif keras_model is not None and isinstance(keras_model, keras.Model):
        data = weights_keras(keras_model, fmt='summary', plot=plot)
    else:
        print("Only keras and HLSModel models can currently be profiled")
        return False, False

    wp = plots[plot](data, fmt='summary') # weight plot
    if isinstance(hls_model, HLSModel) and plot in types_plots:
        t_data = types_hlsmodel(hls_model)
        types_plots[plot](t_data, fmt='summary')

    plt.title("Distribution of (non-zero) weights")
    plt.tight_layout()

    ap = None
    if X is not None and isinstance(keras_model, keras.Model):
        print("Profiling activations")
        data = activations_keras(keras_model, X, fmt='summary', plot=plot)
        ap = plots[plot](data, fmt='summary') # activation plot
        plt.title("Distribution of (non-zero) activations")
        plt.tight_layout()

    if X is not None and isinstance(hls_model, HLSModel):
        t_data = activation_types_hlsmodel(hls_model)
        types_plots[plot](t_data, fmt='summary')

    return wp, ap

