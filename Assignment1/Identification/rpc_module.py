import numpy as np
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module


# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between model image i, and query image j
#
# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image

def plot_rpc(D, plot_color):
    recall = []
    precision = []
    num_queries = D.shape[1]

    num_images = D.shape[0]
    assert (num_images == num_queries), 'Distance matrix should be a square matrix'

    labels = np.diag([1] * num_images)

    d = D.reshape(D.size)
    l = labels.reshape(labels.size)

    sortidx = d.argsort()
    d = d[sortidx]
    l = l[sortidx]

    # ... (your code here)
    tp = 0

    # we initialize fp and fn
    fp = 0
    fn = 0
    for idt in range(len(d)):
        # elements at index where l == 1 are considered as correct
        # As elements are ordered, each iteration acts like a treshold
        # doing iterative sum on l gives us true correct elements

        tp += l[idt]

        # Same reasoning with false positive
        # But his time we only count number of index with 0
        if l[idt] == 0:
            fp += 1

        # or we could simply substract tp from (currentIndex + 1)
        # fn = idt + 1 - tp

        # for False negative we consider indexes with 1 above current index
        # A simple way to achieve this is to substract
        # all positive indexes below current inex  from all correct indexes num

        fn = num_images - tp

        # Compute precision and recall values and append them to "recall" and "precision" vectors
        # ... (your code here)

        p = tp / (tp + fp)  # we compute precision
        r = tp / (tp + fn)  # we compute recall

        precision.append(p)
        recall.append(r)

    plt.plot([1 - precision[i] for i in range(len(precision))], recall, plot_color + '-')


def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):
    assert len(plot_colors) == len(dist_types), 'number of distance types should match the requested plot colors'

    for idx in range(len(dist_types)):
        [best_match, D] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)
        print(hist_type)
        plot_rpc(D, plot_colors[idx])

    plt.axis([0, 1, 0, 1])
    plt.xlabel('1 - precision')
    plt.ylabel('recall')

    # legend(dist_types, 'Location', 'Best')

    plt.legend(dist_types, loc='best')

    # could u see my commit?





