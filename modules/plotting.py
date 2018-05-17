import numpy as np
import matplotlib.pyplot as plt
from data_processing import predict_test_points
from scipy import stats

def get_pred_vs_real_scatterplot(model, training_data_dict, unit_dict, data_keys, predicted_feat, title=None):
    
    if not predicted_feat in training_data_dict['output_features']:
        print('That output feature is not available. Choose between\n%s' % 
              (', '.join(training_data_dict['output_features'])))
        return 
    
    predicted_points = predict_test_points(model, training_data_dict)

    x_test = training_data_dict['x_test']
    y_test = training_data_dict['y_test']
    
    feat_nr = training_data_dict['y_data_keys'][predicted_feat]

    fig = plt.figure(figsize=(8,8))

    plt.plot(y_test[:,feat_nr], y_test[:,feat_nr], 'k.')
    plt.plot(predicted_points[:,feat_nr], y_test[:,feat_nr], 'g.')
    plt.ylabel('True %s %s' % (predicted_feat, unit_dict[predicted_feat]), fontsize=15)
    plt.xlabel('Predicted %s %s' % (predicted_feat, unit_dict[predicted_feat]), fontsize=15)
    plt.legend(['Ideal result', 'predicted ' + predicted_feat], loc='upper center')
    if title is not None:
        plt.title(title, y=1.03, fontsize=20)
    
    return fig




def get_real_vs_pred_boxplot(model, training_data_dict, unit_dict, data_keys, predicted_feat, binning_feat, nBins=8, title=None):
    
    if not predicted_feat in training_data_dict['output_features']:
        print('That output feature is not available. Choose between\n%s' % 
              (', '.join(training_data_dict['output_features'])))
        return 
    if binning_feat in training_data_dict['output_features']:
        binning_feature_data = training_data_dict['y_test'][:,training_data_dict['y_data_keys'][binning_feat]]
        
    elif binning_feat in training_data_dict['input_features']:
        binning_feature_data = training_data_dict['x_test'][:,training_data_dict['x_data_keys'][binning_feat]]
        
    else:   
        print('That binning feature is not available. Choose between\n%s' % 
              (', '.join(training_data_dict['output_features']) + '\nand\n' + 
               ', '.join(training_data_dict['input_features'])))
        return 
    
    predicted_points = predict_test_points(model, training_data_dict)
    
    binned_feat_min_value = np.amin(binning_feature_data)
    binned_feat_max_value = np.amax(binning_feature_data)
    bin_edges = np.linspace(binned_feat_min_value, binned_feat_max_value, nBins+1)

    x_test = training_data_dict['x_test']
    y_test = training_data_dict['y_test']
    
    pred_feat_nr = training_data_dict['y_data_keys'][predicted_feat]

    # bin_means contain (0: mean of the binned values, 1: bin edges, 2: numbers pointing each example to a bin)
    bin_means_true = stats.binned_statistic(binning_feature_data, y_test[:,pred_feat_nr], bins=bin_edges)
    bin_means_pred = stats.binned_statistic(binning_feature_data, predicted_points[:,pred_feat_nr].flatten(), 
                                            bins=bin_edges)

    bin_centers = []
    for iBin in range(nBins):
        bin_center = (bin_means_true[1][iBin] + bin_means_true[1][iBin+1]) / 2
        bin_centers.append('%.2f' % (bin_center))
    sorted_true_y_data = []
    sorted_pred_y_data = []
    for iBin in range(1,nBins+1):
        sorted_true_y_data.append(y_test[bin_means_true[2] == iBin, pred_feat_nr])
        sorted_pred_y_data.append(predicted_points[bin_means_pred[2] == iBin, pred_feat_nr])

    # get standard deviations of the binned values
    stds_true = np.zeros((nBins))
    stds_pred = np.zeros((nBins))
    for iBin in range(nBins):
        stds_true[iBin] = np.std(sorted_true_y_data[iBin])
        stds_pred[iBin] = np.std(sorted_pred_y_data[iBin])

    fig = plt.figure(figsize=(16,8))
    ax = plt.subplot(111)

    bin_pos = np.array([-2,-1]) # (because this makes it work)
    x_label_centers = []
    for iBin in range(nBins):
        # Every plot adds 2 distributions, one from the true data and one from the predicted data
        bin_pos += 3 
        plt.errorbar(bin_pos[0], bin_means_true[0][iBin], yerr=stds_true[iBin], fmt = 'bo', capsize=5)
        plt.errorbar(bin_pos[1], bin_means_pred[0][iBin], yerr=stds_pred[iBin], fmt = 'ro', capsize=5)
        x_label_centers.append(np.mean(bin_pos))

    plt.ylabel('%s %s' % (predicted_feat, unit_dict[predicted_feat]), fontsize=15)
    plt.xlabel('True %s %s' % (binning_feat, unit_dict[binning_feat]), fontsize=15)
    plt.legend(['True data $\pm 1 \sigma$', 'Predicted data $\pm 1 \sigma$'], loc='upper left', fontsize='xx-large')
    ax.set_xlim(left=x_label_centers[0]-2, right=x_label_centers[-1]+2)
    
    plt.xticks(x_label_centers, bin_centers)

    if title is not None:
        plt.title(title, y=1.03, fontsize=20)

    return fig



def get_scatter_comparison_plots(model, training_data_dict, unit_dict, x_axis_feature, y_axis_feature, title=None,
                                y_min=None, y_max=None, x_min=None, x_max=None):

    ### Will make two subplots, the left one with predicted x and y features, the right one with true x and y
    ### features. If either the x or y feature is an input feature, and thus has no predicted feature, the left
    ### subplot will instead contain the true values for that feature.

    if not x_axis_feature in (training_data_dict['output_features'] + training_data_dict['input_features']):
        print('That x feature is not available. Choose between\n%s' %
              (', '.join(training_data_dict['output_features']) + '\nand\n' +
               ', '.join(training_data_dict['input_features'])))
        return
    if not y_axis_feature in (training_data_dict['output_features'] + training_data_dict['input_features']):
        print('That y feature is not available. Choose between\n%s' %
              (', '.join(training_data_dict['output_features']) + '\nand\n' +
               ', '.join(training_data_dict['input_features'])))
        return

    predicted_points = predict_test_points(model, training_data_dict)

    if x_axis_feature in training_data_dict['output_features']:
        left_x_data = predicted_points[:,training_data_dict['y_data_keys'][x_axis_feature]]
        right_x_data = training_data_dict['y_test'][:,training_data_dict['y_data_keys'][x_axis_feature]]
        left_x_label = 'Predicted %s %s' % (x_axis_feature, unit_dict[x_axis_feature])
        right_x_label = 'True %s %s' % (x_axis_feature, unit_dict[x_axis_feature])

    elif x_axis_feature in training_data_dict['input_features']:
        left_x_data = training_data_dict['x_test'][:,training_data_dict['x_data_keys'][x_axis_feature]]
        right_x_data = training_data_dict['x_test'][:,training_data_dict['x_data_keys'][x_axis_feature]]
        left_x_label = 'True %s %s' % (x_axis_feature, unit_dict[x_axis_feature])
        right_x_label = 'True %s %s' % (x_axis_feature, unit_dict[x_axis_feature])

    if y_axis_feature in training_data_dict['output_features']:
        left_y_data = predicted_points[:,training_data_dict['y_data_keys'][y_axis_feature]]
        right_y_data = training_data_dict['y_test'][:,training_data_dict['y_data_keys'][y_axis_feature]]
        left_y_label = 'Predicted %s %s' % (y_axis_feature, unit_dict[y_axis_feature])
        right_y_label = 'True %s %s' % (y_axis_feature, unit_dict[y_axis_feature])

    elif y_axis_feature in training_data_dict['input_features']:
        left_y_data = training_data_dict['x_test'][:,training_data_dict['x_data_keys'][y_axis_feature]]
        right_y_data = training_data_dict['x_test'][:,training_data_dict['x_data_keys'][y_axis_feature]]
        left_y_label = 'True %s %s' % (y_axis_feature, unit_dict[x_axis_feature])
        right_y_label = 'True %s %s' % (y_axis_feature, unit_dict[x_axis_feature])


    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(121)

    plt.plot(left_x_data, left_y_data, 'r.', markersize=2)
    plt.xlabel(left_x_label, fontsize=15)
    plt.ylabel(left_y_label, fontsize=15)
    xmin_1, xmax_1 = ax1.get_xlim()
    ymin_1, ymax_1 = ax1.get_ylim()


    ax2 = plt.subplot(122)
    plt.plot(right_x_data, right_y_data, 'b.', markersize=2)
    plt.xlabel(right_x_label, fontsize=15)
    plt.ylabel(right_y_label, fontsize=15)
    xmin_2, xmax_2 = ax2.get_xlim()
    ymin_2, ymax_2 = ax2.get_ylim()
    
    
    if ymin_1 <= ymin_2:
        if ymax_1 >= ymax_2:
            ax1.set_ylim(bottom=ymin_1, top=ymax_1)
            ax2.set_ylim(bottom=ymin_1, top=ymax_1)
        else:
            ax1.set_ylim(bottom=ymin_1, top=ymax_2)
            ax2.set_ylim(bottom=ymin_1, top=ymax_2)
    else:
        if ymax_1 >= ymax_2:
            ax1.set_ylim(bottom=ymin_2, top=ymax_1)
            ax2.set_ylim(bottom=ymin_2, top=ymax_1)
        else:
            ax1.set_ylim(bottom=ymin_2, top=ymax_2)
            ax2.set_ylim(bottom=ymin_2, top=ymax_2)
            
    if xmin_1 <= xmin_2:
        if xmax_1 >= xmax_2:
            ax1.set_xlim(left=xmin_1, right=xmax_1)
            ax2.set_xlim(left=xmin_1, right=xmax_1)
        else:
            ax1.set_xlim(left=xmin_1, right=xmax_2)
            ax2.set_xlim(left=xmin_1, right=xmax_2)
    else:
        if xmax_1 >= xmax_2:
            ax1.set_xlim(left=xmin_2, right=xmax_1)
            ax2.set_xlim(left=xmin_2, right=xmax_1)
        else:
            ax1.set_xlim(left=xmin_2, right=xmax_2)
            ax2.set_xlim(left=xmin_2, right=xmax_2)
            
    if y_min is not None:
        ax1.set_ylim(bottom=y_min)
        ax2.set_ylim(bottom=y_min)
    if y_max is not None:
        ax1.set_ylim(top=y_max)
        ax2.set_ylim(top=y_max) 
    if x_min is not None:
        ax1.set_xlim(left=x_min)
        ax2.set_xlim(left=x_min)
    if x_max is not None:
        ax1.set_xlim(right=x_max)
        ax2.set_xlim(right=x_max) 
    if title is not None:
        plt.suptitle(title, y=1.03, fontsize=20)

    plt.tight_layout()

    return fig
