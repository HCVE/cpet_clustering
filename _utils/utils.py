import os
import plotly
import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go


def write2excel(label, patient, sex, path, cluster_labels):
    workbook = xlsxwriter.Workbook(path + f'/Clustering_assignments_{sex}.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Patient ID')
    worksheet.write(0, 1, 'Cluster')
    for i in range(len(label)):
        worksheet.write(i + 1, 0, patient[i])
        worksheet.write(i + 1, 1, cluster_labels[label[i]])
    workbook.close()


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    legend_attributes = sorted(unique, key=lambda tup: tup[1])
    ax.legend(*zip(*legend_attributes))


def visualize_clustering_results(dataframe, clustering_results, combinations, sex, root_path, cluster_colours):
    # iterate through the given combinations to be plotted
    y_dim = int(np.ceil(np.sqrt(len(combinations))))
    x_dim = int(len(combinations) / y_dim)
    f_all, ax_all = plt.subplots(x_dim, y_dim, figsize=(25, 15))

    for count, comb in enumerate(combinations):
        rw, cl = int(count // y_dim), int(count % y_dim)
        print(f"{comb[0]} vs {comb[1]}")
        path = root_path + f"/{comb[0]}_vs_{comb[1]}_{sex}"
        # create the folder for each desired combination
        if not os.path.exists(path):
            os.makedirs(path)

        f1, ax = plt.subplots()
        for counter, c in enumerate(np.unique(clustering_results)):
            pairs = []
            positions = np.where(clustering_results == c)[0]
            measurements = dataframe["CPET Data"]

            # plot the individual traces for each cluster separately
            for j in positions:

                if comb[0] == "Time":
                    test_duration = measurements.iloc[j][comb[0]] - measurements.iloc[j][comb[0]].iloc[0]
                else:
                    test_duration = measurements.iloc[j][comb[0]]

                trace = go.Scatter(x=test_duration, y=measurements.iloc[j][comb[1]], mode='lines',
                                   opacity=0.5, marker=dict(color=cluster_colours[counter]),
                                   text='ID: ' + str(dataframe["Patient IDs"].iloc[j]))
                pairs.append(trace)
                ax.plot(test_duration, measurements.iloc[j][comb[1]], c=cluster_colours[counter],
                        alpha=0.6, label="Cluster " + str(c))
                ax_all[rw, cl].plot(test_duration,
                                    measurements.iloc[j][comb[1]],
                                    c=cluster_colours[counter], alpha=0.35, label="Cluster " + str(c))
            ax.set_xlabel(comb[0])
            ax.set_ylabel(comb[1])

            # fix the legend and save the figures
            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels)
            legend_without_duplicate_labels(ax)
            f1.savefig(path + f'/{comb[0]}_vs_{comb[1]}_all_clusters.png')
            f1.savefig(path + f'/{comb[0]}_vs_{comb[1]}_all_clusters.svg')
            plt.close(f1)

            # plot the desired variable combinations for each cluster separately
            layout = go.Layout(title='Cluster ' + str(c + 1), hovermode='closest',
                               xaxis=dict(title=comb[0]), yaxis=dict(title=comb[1]),
                               font=dict(size=25))
            fig = dict(data=pairs, layout=layout)
            plotly.offline.plot(fig, filename=path + f'/{comb[0]}_vs_{comb[1]}_Cluster_{c}.html',
                                auto_open=False)

        ax_all[rw, cl].set_xlabel(comb[0])
        ax_all[rw, cl].set_ylabel(comb[1])

    # fix the legend and save the figures
    handles, labels = ax_all[x_dim - 1, y_dim - 1].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax_all[x_dim - 1, y_dim - 1].legend(handles, labels)
    legend_without_duplicate_labels(ax_all[x_dim - 1, y_dim - 1])
    f_all.savefig(root_path + f'/all_combinations_{sex}.png')
    f_all.savefig(root_path + f'/all_combinations_{sex}.svg')
    plt.close(f_all)


def visualize_most_important_window(cpet_data, model, ids, res, cl_labels, cl_colours):
    plt.rcParams['font.size'] = 14
    combinations = [("V'O2", "HR"), ("Time", "PETO2"), ("Time", "PETCO2"), ("Time", "RER")]
    cl_labels_n = dict((v, k) for k, v in cl_labels.items())
    combinations_dictionary = {"HR": 0, "V'O2": 1, "RER": 2, "PETO2": 3, "PETCO2": 4, "Time": 5}

    medoid_indices = model.clustering_model.medoid_indices_
    patients = [ids[i] for i in medoid_indices]
    centroids = model.medoids_

    time = [cpet_data[cpet_data["Patient IDs"] == i]["CPET Data"].iloc[0]["Time"].to_numpy() for i in patients]
    padded_time = np.zeros((5, centroids.shape[1], 1))
    padded_time[:] = np.nan
    for i in range(len(time)):
        time[i] = time[i] - time[i][0]
        padded_time[i, :len(time[i]), 0] = time[i]

    centroids = np.concatenate((centroids, padded_time), axis=2)

    y_dim = int(np.ceil(np.sqrt(len(combinations))))
    x_dim = int(len(combinations) / y_dim)
    f, ax = plt.subplots(x_dim, y_dim, figsize=(25, 15))
    for n_comb, comb in enumerate(combinations):
        rw, col = int(n_comb // y_dim), int(n_comb % y_dim)
        for counter, key in enumerate(res.keys()):
            ax[rw, col].plot(centroids[cl_labels_n[counter + 1], :, combinations_dictionary[comb[0]]],
                             centroids[cl_labels_n[counter + 1], :, combinations_dictionary[comb[1]]],
                             linewidth=1.5, c=cl_colours[counter + 1], label=f"Cluster {counter + 1}")
            for j in res[key]:
                j = np.squeeze(np.array(j))
                if len(j) > 0:
                    c = centroids[cl_labels_n[counter + 1]]
                    c = c[~np.isnan(c[:, 0]), :]
                    if j[-1] > len(c):
                        j = np.array([len(c)-1-k for k in range(len(j))])

                    ax[rw, col].plot(centroids[cl_labels_n[counter + 1], j, combinations_dictionary[comb[0]]],
                                     centroids[cl_labels_n[counter + 1], j, combinations_dictionary[comb[1]]],
                                     linewidth=3.5, c=cl_colours[counter + 1])

        ax[rw, col] = axis_configuration_interpretability(ax=ax[rw, col], data_category=comb)
        handles, labels = ax[rw, col].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax[rw, col].legend(handles, labels)
    plt.savefig(f"C:/Users/vagge/Desktop/distinctive_window.png")
    plt.savefig(f"C:/Users/vagge/Desktop/distinctive_window.svg")
    plt.show()


def axis_configuration_interpretability(ax, data_category):
    if data_category[0] == "V'O2" and data_category[1] == "HR":
        ax.set_xlabel(f" O\u2082 Uptake (ml/min)")
        ax.set_ylabel(f" Heart rate (beats/min)")
        ax.set_xlim(200, 3500)
        ax.set_ylim(60, 185)
    elif data_category[0] == "Time" and data_category[1] == "PETO2":
        ax.set_xlabel(f" Duration of test (sec)")
        ax.set_ylabel(f"End-tidal O\u2082 (mm Hg)")
        ax.set_xlim(-50, 800)
        ax.set_ylim(95, 125)
    elif data_category[0] == "Time" and data_category[1] == "PETCO2":
        ax.set_xlabel(f" Duration of test (sec)")
        ax.set_ylabel(f"End-tidal CO\u2082 (mm Hg)")
        ax.set_xlim(-50, 800)
        ax.set_ylim(28, 44)
    elif data_category[0] == "Time" and data_category[1] == "RER":
        ax.set_xlabel(f" Duration of test (sec)")
        ax.set_ylabel(f"Respiratory exchange ratio")
        ax.set_xlim(-50, 800)
        ax.set_ylim(0.6, 1.3)
    else:
        raise ValueError(f"Invalid keywords. '{data_category[0]}' and {data_category[1]} are not recognized.")
    return ax
