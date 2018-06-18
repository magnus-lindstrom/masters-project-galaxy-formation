import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from data_processing import predict_points
from scipy import stats
import corner

def get_pred_vs_real_scatterplot(model, training_data_dict, unit_dict, data_keys, predicted_feat, redshifts='all', title=None,
                                 data_type='test', predicted_points=None, n_points=1000, n_columns=3):
    
    if not predicted_feat in training_data_dict['output_features']:
        print('That output feature is not available. Choose between\n%s' % 
              (', '.join(training_data_dict['output_features'])))
        return 
    if redshifts == 'all':
        unique_redshifts = np.unique(training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys']['Redshift']])
    else:
        for redshift in redshifts:
            if redshift not in training_data_dict['redshifts']:
                print('The redshift {} was not used for training'.format(redshift))
                return
        unique_redshifts = redshifts
     
    feat_nr = training_data_dict['y_data_keys'][predicted_feat]
    
    if predicted_points is None:
        predicted_points = predict_points(model, training_data_dict, data_type = data_type)
        
    n_fig_rows = int(np.ceil(len(unique_redshifts)/n_columns))
    if len(unique_redshifts) <= n_columns:
        n_fig_cols = len(unique_redshifts)
    else:
        n_fig_cols = n_columns
        
    fig = plt.figure(figsize=(4*n_fig_cols,4*n_fig_rows))
    ax_list = []

    global_xmin = float('Inf')
    global_xmax = -float('Inf')
    global_ymin = float('Inf')
    global_ymax = -float('Inf')
    
    for i_red, redshift in enumerate(unique_redshifts):
        relevant_inds = training_data_dict['original_data'][training_data_dict['{}_indices'.format(data_type)], 
                                                              training_data_dict['original_data_keys']['Redshift']] == redshift
        data_redshift = training_data_dict['y_{}'.format(data_type)][relevant_inds, :]
        pred_points_redshift = predicted_points[relevant_inds, :]

        ax = plt.subplot(n_fig_rows, n_fig_cols, i_red+1)

        true_handle = ax.scatter(data_redshift[:n_points, feat_nr], data_redshift[:n_points, feat_nr], c='b', s=8)
        pred_handle = ax.scatter(pred_points_redshift[:n_points, feat_nr], data_redshift[:n_points, feat_nr], c='r', s=8, alpha=0.3)
        ax.set_ylabel('$log(%s[%s])$' % (predicted_feat, unit_dict[predicted_feat]), fontsize=15)
        ax.set_xlabel('$log(%s_{DNN}[%s])$' % (predicted_feat, unit_dict[predicted_feat]), fontsize=15)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if xmin < global_xmin:
            global_xmin = xmin
        if xmax > global_xmax:
            global_xmax = xmax
        if ymin < global_ymin:
            global_ymin = ymin
        if ymax > global_ymax:
            global_ymax = ymax
  #      ax.legend(['Emerge', 'DNN'], loc='upper left')
        ax_list.append(ax)
    
    fig.subplots_adjust(hspace=0, wspace=0)
    
  #  for i_ax, ax in enumerate(ax_list):
  #      ax.set_xlim(left=global_xmin, right=global_xmax)
  #      ax.set_ylim(bottom=global_ymin, top=global_ymax)
  #      if i_ax % n_fig_cols is not 0:
     #       ax.set_yticks([])
   #         ax.set_ylabel('')
            
    for i_ax, ax in enumerate(ax_list):
        
        # enable upper ticks as well as right ticks
        ax.tick_params(axis='x', top=True)
        ax.tick_params(axis='y', right=True)
        # turn off x-labels for all but the last subplots
        if (len(ax_list) - (i_ax+1)) > n_fig_cols:   
            ax.set_xlabel('')
        if i_ax % n_fig_cols is not 0:
            ax.set_ylabel('')
        # set the lims to be the global max/min
        ax.set_ylim(bottom=global_ymin, top=global_ymax)
        ax.set_xlim(left=global_xmin, right=global_xmax)
        # make sure the ticks are on the inside and the numbers are on the outside of the plots
        ax.tick_params(axis="y",direction="in", pad=10)
        ax.tick_params(axis="x",direction="in", pad=10)
        # display redshift inside the plots
        ax.text(-5, 1.5, 'z = {:2.1f}'.format(unique_redshifts[i_ax]), fontsize=20) 
        
    # set one big legend outside the subplots
    legend = fig.legend( [true_handle, pred_handle], ['Emerge data $\pm 1 \sigma$', 'DNN data $\pm 1 \sigma$'], loc = (0.4, .1), 
               fontsize=30, markerscale=7)
 #   legend.legendHandles[0]._legmarker.set_markersize(6)
 #   legend.legendHandles[1]._legmarker.set_markersize(6)
    if title is not None:
        fig.suptitle(title, y=.93, fontsize=20)
    
    
    
    return fig
    
    

def get_real_vs_pred_boxplot(model, training_data_dict, unit_dict, data_keys, predicted_feat, binning_feat, redshifts='all', 
                             nBins=8, title=None, data_type='test', predicted_points=None, n_points=1000, n_columns=3):
    
    if not predicted_feat in training_data_dict['output_features']:
        print('Predicted feature not available (%s). Choose between\n%s' % 
              (predicted_feat, ', '.join(training_data_dict['output_features'])))
        return 
    pred_feat_nr = training_data_dict['y_data_keys'][predicted_feat]
        
    if redshifts == 'all':
        unique_redshifts = np.unique(training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys']['Redshift']])
    else:
        for redshift in redshifts:
            if redshift not in training_data_dict['redshifts']:
                print('The redshift {} was not used for training'.format(redshift))
                return
        unique_redshifts = redshifts
        
    if predicted_points is None:
        predicted_points = predict_points(model, training_data_dict, data_type = data_type)    
        
    n_fig_rows = int(np.ceil(len(unique_redshifts)/n_columns))
    if len(unique_redshifts) <= n_columns:
        n_fig_cols = len(unique_redshifts)
    else:
        n_fig_cols = n_columns
        
    fig = plt.figure(figsize=(4*n_fig_cols,4*n_fig_rows))
    ax_list = []
    xtick_list = []
    xtick_label_list = []
    
    global_ymin = float('Inf')
    global_ymax = -float('Inf')
    
    # set the bin edges for every subplot
    binning_feature_data_tot = training_data_dict['original_data'][training_data_dict['{}_indices'.format(data_type)], 
                                                              training_data_dict['original_data_keys'][binning_feat]]
    binned_feat_min_value = np.amin(binning_feature_data_tot)
    binned_feat_max_value = np.amax(binning_feature_data_tot)
    bin_edges = np.linspace(binned_feat_min_value, binned_feat_max_value, nBins+1)

    bin_centers = []
    for iBin in range(nBins):
        bin_center = (bin_edges[iBin] + bin_edges[iBin+1]) / 2
        bin_centers.append('{:.1f}'.format(bin_center))
    
    for i_red, redshift in enumerate(unique_redshifts):
    
        # get the indeces of the train/val/test data that have the current redshift
        relevant_inds = training_data_dict['original_data'][training_data_dict['{}_indices'.format(data_type)], 
                                                              training_data_dict['original_data_keys']['Redshift']] == redshift
        data_redshift = training_data_dict['y_{}'.format(data_type)][relevant_inds, pred_feat_nr]
        pred_points_redshift = predicted_points[relevant_inds, pred_feat_nr]
        binning_feature_data_redshift = binning_feature_data_tot[relevant_inds]

        

        # bin_means contain (0: mean of the binned values, 1: bin edges, 2: numbers pointing each example to a bin)
        bin_means_true = stats.binned_statistic(binning_feature_data_redshift, data_redshift, bins=bin_edges)
        bin_means_pred = stats.binned_statistic(binning_feature_data_redshift, pred_points_redshift.flatten(), 
                                                bins=bin_edges)  
  
        # separate the data, based on redshift, into separate lists 
        sorted_true_y_data = []
        sorted_pred_y_data = []
        for iBin in range(1,nBins+1):
            sorted_true_y_data.append(data_redshift[bin_means_true[2] == iBin])
            sorted_pred_y_data.append(pred_points_redshift[bin_means_pred[2] == iBin])

        # get standard deviations of the binned values
        stds_true = np.zeros((nBins))
        stds_pred = np.zeros((nBins))
        for iBin in range(nBins):
            stds_true[iBin] = np.std(sorted_true_y_data[iBin])
            stds_pred[iBin] = np.std(sorted_pred_y_data[iBin])

        ax = plt.subplot(n_fig_rows, n_fig_cols, i_red+1)

        bin_pos = np.array([-2,-1]) # (because this makes it work)
        x_label_centers = []
        for iBin in range(nBins):
            # Every plot adds 2 distributions, one from the Emerge data and one from the DNN data
            bin_pos += 3 
            true_handle = ax.errorbar(bin_pos[0], bin_means_true[0][iBin], yerr=stds_true[iBin], fmt = 'bo', capsize=5)
            pred_handle = ax.errorbar(bin_pos[1], bin_means_pred[0][iBin], yerr=stds_pred[iBin], fmt = 'ro', capsize=5)
            x_label_centers.append(np.mean(bin_pos))
            
        x_feat_name = get_print_name(binning_feat)
        y_feat_name = get_print_name(predicted_feat)

        ax.set_ylabel('log({}$[{}])$'.format(y_feat_name, unit_dict[predicted_feat]), fontsize=12)
        ax.set_xlabel('log({}$_{{Emerge}}[{}])$'.format(x_feat_name, unit_dict[binning_feat]), fontsize=12)
        ax.set_xlim(left=x_label_centers[0]-2, right=x_label_centers[-1]+2)

        # set the xticks to be where the bin centers are and with the x-value of the bin centers
        xtick_list.append(x_label_centers)
        xtick_label_list.append(bin_centers)
        
        # keep track of the max and min y-values to do a scaling of the y-axis after the global max/min has been found
        ymin, ymax = ax.get_ylim()
        if ymin < global_ymin:
            global_ymin = ymin
        if ymax > global_ymax:
            global_ymax = ymax
        
        # don't draw the tick values nor the y label on the inner plots, just the ones to the left
        if i_red % n_fig_cols is not 0:
            ax.set_yticklabels([])
            ax.set_ylabel('')
            
        ax_list.append(ax)
        
    for i_ax, ax in enumerate(ax_list):
        
        # enable upper ticks as well as right ticks
        ax.tick_params(axis='x', top=True)
        ax.tick_params(axis='y', right=True)
        # set all of the x-ticks
        ax.set_xticks(xtick_list[i_ax], minor=False)
        ax.set_xticklabels(xtick_label_list[i_ax])
        # turn off x-labels for all but the last subplots
        if (len(ax_list) - (i_ax+1)) > n_fig_cols:   
            ax.set_xlabel('')
        # set the ylims to be the global max/min
        ax.set_ylim(bottom=global_ymin, top=global_ymax)
        # make sure the ticks are on the inside and the numbers are on the outside of the plots
        ax.tick_params(axis="y",direction="in", pad=10)
        ax.tick_params(axis="x",direction="in", pad=10)
        # display redshift inside the plots
        if predicted_feat == 'Stellar_mass' and binning_feat == 'Halo_mass':
            ax.text(13, 7.5, 'z = {:2.1f}'.format(unique_redshifts[i_ax]), fontsize=20)
        elif predicted_feat == 'SFR' and binning_feat == 'Stellar_mass':
            ax.text(1, 2, 'z = {:2.1f}'.format(unique_redshifts[i_ax]), fontsize=20)

    # eliminate space between plots
    fig.subplots_adjust(hspace=0, wspace=0)
    # set one big legend outside the subplots
    fig.legend( [true_handle, pred_handle], ['Emerge data $\pm 1 \sigma$', 'DNN data $\pm 1 \sigma$'], loc = (0.4, .1), 
               fontsize=30)

    if title is not None:
        plt.title(title, x=1.5, y=4.1, fontsize=20)

    return fig


def get_halo_stellar_mass_plots(model, training_data_dict, unit_dict, redshifts='all', 
                                 title=None, n_redshifts_per_row=2, y_min=None, y_max=None, x_min=None, x_max=None, 
                                 data_type='test', predicted_points=None):

    ### Will make two subplots, the left one with predicted x and y features, the right one with true x and y
    ### features. If either the x or y feature is an input feature, and thus has no predicted feature, the left
    ### subplot will instead contain the true values for that feature.

    if predicted_points is None:
        predicted_points = predict_points(model, training_data_dict, data_type = data_type)
        
    if redshifts == 'all':
        unique_redshifts = np.unique(training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys']['Redshift']])
    else:
        for redshift in redshifts:
            if redshift not in training_data_dict['redshifts']:
                print('The redshift {} was not used for training'.format(redshift))
                return
        unique_redshifts = redshifts
        
    x_data = training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys']['Halo_mass']]
    true_y_data = training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys']['Stellar_mass']]
    predicted_y_data = predicted_points[:,training_data_dict['y_data_keys']['Stellar_mass']]

    x_feat_name = get_print_name('Halo_mass')
    y_feat_name = get_print_name('Stellar_mass')
 
    x_label = 'log({}$_{{Emerge}}[{}])$'.format(x_feat_name, unit_dict['Halo_mass'])
    y_label = 'log({}$[{}])$'.format(y_feat_name, unit_dict['Stellar_mass'])

     
    n_fig_rows = np.ceil(int(len(unique_redshifts) / n_redshifts_per_row))
    if n_redshifts_per_row == 2 and len(unique_redshifts) > 1:
        n_fig_columns = 4
    else:
        n_fig_columns = 2
    
    fig = plt.figure(figsize=(6*n_fig_columns,4*n_fig_rows))
    pred_ax_list = []
    true_ax_list = []
    ax_list = []
    global_xmin = float('Inf')
    global_xmax = -float('Inf')
    global_ymin = float('Inf')
    global_ymax = -float('Inf')

    for i_red, redshift in enumerate(unique_redshifts):
        
        # get the indeces of the train/val/test data that have the current redshift
        relevant_inds = training_data_dict['original_data'][training_data_dict['{}_indices'.format(data_type)], 
                                                              training_data_dict['original_data_keys']['Redshift']] == redshift
        x_data_redshift = x_data[relevant_inds]
        true_y_data_redshift = true_y_data[relevant_inds]
        predicted_y_data_redshift = predicted_y_data[relevant_inds]
        
        ax_pred = plt.subplot(n_fig_rows, n_fig_columns, i_red * 2 + 1)

        ax_pred.plot(x_data_redshift, predicted_y_data_redshift, 'r.', markersize=2)
   #     ax1.xlabel(predicted_x_label, fontsize=15)
    #    ax1.ylabel(predicted_y_label, fontsize=15)
        xmin, xmax = ax_pred.get_xlim()
        ymin, ymax = ax_pred.get_ylim()

        if xmin < global_xmin:
            global_xmin = xmin
        if xmax > global_xmax:
            global_xmax = xmax
        if ymin < global_ymin:
            global_ymin = ymin
        if ymax > global_ymax:
            global_ymax = ymax
            
        ax_true = plt.subplot(n_fig_rows, n_fig_columns, i_red * 2 + 2)
        ax_true.plot(x_data_redshift, true_y_data_redshift, 'b.', markersize=2)
   #     ax2.xlabel(true_x_label, fontsize=15)
   #     ax2.ylabel(true_y_label, fontsize=15)
        xmin, xmax = ax_true.get_xlim()
        ymin, ymax = ax_true.get_ylim()
        
        if xmin < global_xmin:
            global_xmin = xmin
        if xmax > global_xmax:
            global_xmax = xmax
        if ymin < global_ymin:
            global_ymin = ymin
        if ymax > global_ymax:
            global_ymax = ymax
        
        pred_ax_list.append(ax_pred)
        true_ax_list.append(ax_true)
        
    for i_ax, (ax_pred, ax_true) in enumerate(zip(pred_ax_list, true_ax_list)):
        
        # enable upper ticks as well as right ticks
        ax_pred.tick_params(axis='x', top=True)
        ax_pred.tick_params(axis='y', right=True)
        ax_true.tick_params(axis='x', top=True)
        ax_true.tick_params(axis='y', right=True)
        if (2*(len(pred_ax_list) - (i_ax+1))) < n_fig_columns:   
            ax_pred.set_xlabel(x_label, fontsize=15)
  #          ax_pred.set_ylabel(predicted_y_label, fontsize=15)
            ax_true.set_xlabel(x_label, fontsize=15)
  #          ax_true.set_ylabel(true_y_label, fontsize=15)
        if n_redshifts_per_row == 2:
            if i_ax % 2 is 0:
                ax_pred.set_ylabel(y_label, fontsize=15)
        # set the lims to be the global max/min
        ax_pred.set_xlim(left=global_xmin, right=global_xmax)
        ax_pred.set_ylim(bottom=global_ymin, top=global_ymax)
        ax_true.set_xlim(left=global_xmin, right=global_xmax)
        ax_true.set_ylim(bottom=global_ymin, top=global_ymax)
        # make sure the ticks are on the inside and the numbers are on the outside of the plots
        ax_pred.tick_params(axis="y",direction="in", pad=10)
        ax_pred.tick_params(axis="x",direction="in", pad=10)
        ax_true.tick_params(axis="y",direction="in", pad=10)
        ax_true.tick_params(axis="x",direction="in", pad=10)
        # display redshift inside the plots
        ax_pred.text(.73, .38, 'z = {:2.1f}'.format(unique_redshifts[i_ax]), fontsize=20, transform = ax_pred.transAxes,
                    horizontalalignment='center')
        ax_true.text(.73, .38, 'z = {:2.1f}'.format(unique_redshifts[i_ax]), fontsize=20, transform = ax_true.transAxes,
                    horizontalalignment='center')
        # set legends
        ax_pred.legend(['DNN'], loc=(.6,.2), fontsize=15, markerscale=7)
        ax_true.legend(['Emerge'], loc=(.6,.2), fontsize=15, markerscale=7)

    # eliminate space between plots
    fig.subplots_adjust(hspace=0, wspace=0)
    if title is not None:
        plt.suptitle(title, y=.92, fontsize=20)

    return fig


def get_stellar_mass_sfr_plots(model, training_data_dict, unit_dict, redshifts='all', 
                                 title=None, n_redshifts_per_row=2, y_min=None, y_max=None, x_min=None, x_max=None, 
                                 data_type='test', predicted_points=None):

    ### Will make two subplots, the left one with predicted x and y features, the right one with true x and y
    ### features. If either the x or y feature is an input feature, and thus has no predicted feature, the left
    ### subplot will instead contain the true values for that feature.

    if predicted_points is None:
        predicted_points = predict_points(model, training_data_dict, data_type = data_type)
        
    if redshifts == 'all':
        unique_redshifts = np.unique(training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys']['Redshift']])
    else:
        for redshift in redshifts:
            if redshift not in training_data_dict['redshifts']:
                print('The redshift {} was not used for training'.format(redshift))
                return
        unique_redshifts = redshifts
        
    true_x_data = training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys']['Stellar_mass']]
    true_y_data = training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys']['SFR']]
    predicted_x_data = predicted_points[:,training_data_dict['y_data_keys']['Stellar_mass']]
    predicted_y_data = predicted_points[:,training_data_dict['y_data_keys']['SFR']]
    
    x_feat_name = get_print_name('Stellar_mass')
    y_feat_name = get_print_name('SFR')
 
    true_x_label = 'log({}$_{{Emerge}}[{}])$'.format(x_feat_name, unit_dict['Stellar_mass'])
    predicted_x_label = 'log({}$_{{DNN}}[{}])$'.format(x_feat_name, unit_dict['Stellar_mass'])
    y_label = 'log({}$[{}])$'.format(y_feat_name, unit_dict['SFR'])

    n_fig_rows = np.ceil(int(len(unique_redshifts) / n_redshifts_per_row))
    if n_redshifts_per_row == 2 and len(unique_redshifts) > 1:
        n_fig_columns = 4
    else:
        n_fig_columns = 2
    
    fig = plt.figure(figsize=(6*n_fig_columns,4*n_fig_rows))
    pred_ax_list = []
    true_ax_list = []
    ax_list = []
    global_xmin = float('Inf')
    global_xmax = -float('Inf')
    global_ymin = float('Inf')
    global_ymax = -float('Inf')

    for i_red, redshift in enumerate(unique_redshifts):
        
        # get the indeces of the train/val/test data that have the current redshift
        relevant_inds = training_data_dict['original_data'][training_data_dict['{}_indices'.format(data_type)], 
                                                              training_data_dict['original_data_keys']['Redshift']] == redshift
        true_x_data_redshift = true_x_data[relevant_inds]
        true_y_data_redshift = true_y_data[relevant_inds]
        predicted_x_data_redshift = predicted_x_data[relevant_inds]
        predicted_y_data_redshift = predicted_y_data[relevant_inds]
        
        ax_pred = plt.subplot(n_fig_rows, n_fig_columns, i_red * 2 + 1)

        ax_pred.plot(predicted_x_data_redshift, predicted_y_data_redshift, 'r.', markersize=2)
   #     ax1.xlabel(predicted_x_label, fontsize=15)
    #    ax1.ylabel(predicted_y_label, fontsize=15)
        xmin, xmax = ax_pred.get_xlim()
        ymin, ymax = ax_pred.get_ylim()

        if xmin < global_xmin:
            global_xmin = xmin
        if xmax > global_xmax:
            global_xmax = xmax
        if ymin < global_ymin:
            global_ymin = ymin
        if ymax > global_ymax:
            global_ymax = ymax
            
        ax_true = plt.subplot(n_fig_rows, n_fig_columns, i_red * 2 + 2)
        ax_true.plot(true_x_data_redshift, true_y_data_redshift, 'b.', markersize=2)
   #     ax2.xlabel(true_x_label, fontsize=15)
   #     ax2.ylabel(true_y_label, fontsize=15)
        xmin, xmax = ax_true.get_xlim()
        ymin, ymax = ax_true.get_ylim()
        
        if xmin < global_xmin:
            global_xmin = xmin
        if xmax > global_xmax:
            global_xmax = xmax
        if ymin < global_ymin:
            global_ymin = ymin
        if ymax > global_ymax:
            global_ymax = ymax
        
        pred_ax_list.append(ax_pred)
        true_ax_list.append(ax_true)
        
    for i_ax, (ax_pred, ax_true) in enumerate(zip(pred_ax_list, true_ax_list)):
        
        # enable upper ticks as well as right ticks
        ax_pred.tick_params(axis='x', top=True)
        ax_pred.tick_params(axis='y', right=True)
        ax_true.tick_params(axis='x', top=True)
        ax_true.tick_params(axis='y', right=True)
        if (2*(len(pred_ax_list) - (i_ax+1))) < n_fig_columns:   
            ax_pred.set_xlabel(predicted_x_label, fontsize=15)
  #          ax_pred.set_ylabel(predicted_y_label, fontsize=15)
            ax_true.set_xlabel(true_x_label, fontsize=15)
  #          ax_true.set_ylabel(true_y_label, fontsize=15)
        if n_redshifts_per_row == 2:
            if i_ax % 2 is 0:
                ax_pred.set_ylabel(y_label, fontsize=15)
        # set the lims to be the global max/min
        ax_pred.set_xlim(left=global_xmin, right=global_xmax)
        ax_pred.set_ylim(bottom=global_ymin, top=global_ymax)
        ax_true.set_xlim(left=global_xmin, right=global_xmax)
        ax_true.set_ylim(bottom=global_ymin, top=global_ymax)
        # make sure the ticks are on the inside and the numbers are on the outside of the plots
        ax_pred.tick_params(axis="y",direction="in", pad=10)
        ax_pred.tick_params(axis="x",direction="in", pad=10)
        ax_true.tick_params(axis="y",direction="in", pad=10)
        ax_true.tick_params(axis="x",direction="in", pad=10)
        # display redshift inside the plots
        ax_pred.text(.1, .8, 'z = {:2.1f}'.format(unique_redshifts[i_ax]), fontsize=20, transform = ax_pred.transAxes)
        ax_true.text(.1, .8, 'z = {:2.1f}'.format(unique_redshifts[i_ax]), fontsize=20, transform = ax_true.transAxes)
        # set legends
        ax_pred.legend(['DNN'], loc=(.6,.2), fontsize=15, markerscale=7)
        ax_true.legend(['Emerge'], loc=(.6,.2), fontsize=15, markerscale=7)

    # eliminate space between plots
    fig.subplots_adjust(hspace=0, wspace=0)
    if title is not None:
        plt.suptitle(title, y=.92, fontsize=20)

    return fig

def get_scatter_comparison_plots_old(model, training_data_dict, unit_dict, x_axis_feature, y_axis_feature, title=None,
                                y_min=None, y_max=None, x_min=None, x_max=None, data_type='test',
                                predicted_points=None):

    ### Will make two subplots, the left one with predicted x and y features, the right one with true x and y
    ### features. If either the x or y feature is an input feature, and thus has no predicted feature, the left
    ### subplot will instead contain the true values for that feature.

    if predicted_points is None:
        predicted_points = predict_points(model, training_data_dict, data_type = data_type)
        
    true_x_data = training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys'][x_axis_feature]]
    true_y_data = training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys'][y_axis_feature]]
    true_x_label = 'True %s %s' % (x_axis_feature, unit_dict[x_axis_feature])
    true_y_label = 'True %s %s' % (y_axis_feature, unit_dict[y_axis_feature])
    if x_axis_feature in training_data_dict['output_features']:
        predicted_x_data = predicted_points[:,training_data_dict['y_data_keys'][x_axis_feature]]
        predicted_x_label = 'Predicted %s %s' % (x_axis_feature, unit_dict[x_axis_feature])
    else:
        predicted_x_data = true_x_data
        predicted_x_label = 'True %s %s' % (x_axis_feature, unit_dict[x_axis_feature])
        
    if y_axis_feature in training_data_dict['output_features']:
        predicted_y_data = predicted_points[:,training_data_dict['y_data_keys'][y_axis_feature]]
        predicted_y_label = 'Predicted %s %s' % (y_axis_feature, unit_dict[y_axis_feature])
    else:
        predicted_y_data = true_y_data
        predicted_y_label = 'True %s %s' % (y_axis_feature, unit_dict[y_axis_feature])

    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(121)

    plt.plot(predicted_x_data, predicted_y_data, 'r.', markersize=2)
    plt.xlabel(predicted_x_label, fontsize=15)
    plt.ylabel(predicted_y_label, fontsize=15)
    xmin_1, xmax_1 = ax1.get_xlim()
    ymin_1, ymax_1 = ax1.get_ylim()


    ax2 = plt.subplot(122)
    plt.plot(true_x_data, true_y_data, 'b.', markersize=2)
    plt.xlabel(true_x_label, fontsize=15)
    plt.ylabel(true_y_label, fontsize=15)
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
        plt.suptitle(title, y=1.1, fontsize=20)

    return fig


def get_real_vs_pred_same_fig(model, training_data_dict, unit_dict, x_axis_feature, y_axis_feature, title=None, data_type='test',
                             marker_size=5, y_min=None, y_max=None, x_min=None, x_max=None, predicted_points=None):
    
    
    if not y_axis_feature in training_data_dict['output_features']:
        print('y axis feature not available (%s). Choose between\n%s' % 
              (y_axis_feature, ', '.join(training_data_dict['output_features'])))
        return 
        
    x_data = training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys'][x_axis_feature]]
    if predicted_points is None:
        pred_y_data = predict_points(model, training_data_dict, data_type = data_type)[:,training_data_dict['y_data_keys']
                                                                                   [y_axis_feature]]
    else:
        pred_y_data = predicted_points[:,training_data_dict['y_data_keys'][y_axis_feature]]
        
    
    true_y_data = training_data_dict['original_data'][training_data_dict[data_type+'_indices'], 
                                                              training_data_dict['original_data_keys'][y_axis_feature]]
    x_label = 'True %s %s' % (x_axis_feature, unit_dict[x_axis_feature])
    y_label = '%s %s' % (y_axis_feature, unit_dict[y_axis_feature])
    
    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(111)

    plt.plot(x_data, true_y_data, 'b.', markersize=marker_size)
    plt.plot(x_data, pred_y_data, 'r.', markersize=marker_size)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if x_min is not None:
        ax.set_xlim(left=x_min)
    if x_max is not None:
        ax.set_xlim(right=x_max)

    plt.legend(['True data', 'Predicted data'], loc='upper left', fontsize='xx-large')
    
    if title is not None:
        plt.title(title, y=1.03, fontsize=20)
        
    return fig
    
def get_sfr_stellar_mass_contour(model, training_data_dict, unit_dict, title=None, data_type='test',
                                 y_min=None, y_max=None, x_min=None, x_max=None, predicted_points=None):
    
    if predicted_points is None:
        predicted_points = predict_points(model, training_data_dict, data_type = data_type)
        
        
    sfr_true = training_data_dict['original_data'][:, training_data_dict['original_data_keys']['SFR']]
    stellar_masses_true = training_data_dict['original_data'][:, training_data_dict['original_data_keys']['Stellar_mass']]
    
    min_true_sfr = np.amin(sfr_true)
    min_pred_sfr = np.amin(predicted_points[:,training_data_dict['y_data_keys']['SFR']])
    max_true_sfr = np.amax(sfr_true)
    max_pred_sfr = np.amax(predicted_points[:,training_data_dict['y_data_keys']['SFR']])
    
    min_true_stellar_mass = np.amin(stellar_masses_true)
    min_pred_stellar_mass = np.amin(predicted_points[:,training_data_dict['y_data_keys']['Stellar_mass']])
    max_true_stellar_mass = np.amax(stellar_masses_true)
    max_pred_stellar_mass = np.amax(predicted_points[:,training_data_dict['y_data_keys']['Stellar_mass']])
    
    if min_true_sfr <= min_pred_sfr:
        if max_true_sfr > max_pred_sfr:
            sfr_range = [min_true_sfr, max_true_sfr]
        else:
            sfr_range = [min_true_sfr, max_pred_sfr]
    else:
        if max_true_sfr > max_pred_sfr:
            sfr_range = [min_pred_sfr, max_true_sfr]
        else:
            sfr_range = [min_pred_sfr, max_pred_sfr]
    
    if min_true_stellar_mass <= min_pred_stellar_mass:
        if max_true_stellar_mass > max_pred_stellar_mass:
            stellar_mass_range = [min_true_stellar_mass, max_true_stellar_mass]
        else:
            stellar_mass_range = [min_true_stellar_mass, max_true_stellar_mass]
    else:
        if max_true_stellar_mass > max_pred_stellar_mass:
            stellar_mass_range = [min_pred_stellar_mass, max_true_stellar_mass]
        else:
            stellar_mass_range = [min_pred_stellar_mass, max_pred_stellar_mass]
        
        
        
        if np.amin(stellar_masses_true) <= np.amin(predicted_points[:,training_data_dict['y_data_keys']['Stellar_mass']]):
            min_sfr = np.amin(sfr_true)
    sfr_true = np.expand_dims(sfr_true, axis=1)
    stellar_masses_true = np.expand_dims(stellar_masses_true, axis=1)

    data = np.hstack((stellar_masses_true, sfr_true))

    fig1 = corner.corner(data, labels=['Stellar_mass {}'.format(unit_dict['Stellar_mass']),
                                         'SFR {}'.format(unit_dict['SFR'])], label_kwargs={"fontsize": 20},
                                         range=[stellar_mass_range, sfr_range])
    fig1.gca().annotate("True stellar mass to\nSFR contour plot.",
                          xy=(.78, .75), xycoords="figure fraction",
                          xytext=(0, 0), textcoords="offset points",
                          ha="center", va="center", fontsize=20)
    fig1.set_size_inches(12, 12)
    plt.tight_layout()
    plt.show()
    
    fig2 = corner.corner(predicted_points, labels=['Stellar_mass {}'.format(unit_dict['Stellar_mass']),
                                         'SFR {}'.format(unit_dict['SFR'])], label_kwargs={"fontsize": 20},
                                         range=[stellar_mass_range, sfr_range])
    fig2.gca().annotate("Predicted stellar mass to\nSFR contour plot.",
                          xy=(.78, .75), xycoords="figure fraction",
                          xytext=(0, 0), textcoords="offset points",
                          ha="center", va="center", fontsize=20)
    fig2.set_size_inches(12, 12)
    plt.tight_layout()
    plt.show()
    
    return [fig1, fig2]
    
    
def get_print_name(feature_name):
    
    if feature_name == 'Halo_mass':
        print_name = 'Halo mass'
    elif feature_name == 'SFR':
        print_name = 'SFR'
    elif feature_name == 'Stellar_mass':
        print_name = 'Stellar mass'
        
    return print_name
    
    
    
    
    
    
