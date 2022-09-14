import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from model import decision_boundary, one_pass

def animate_model(model, intermediate_model, cuda_flag, df, indb, indr):
    if cuda_flag:
        model = model.cuda()
        spit_out_intermediate = intermediate_model.cuda()
    
    
    # Define model training
    loss_criterion = nn.MSELoss() # BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Define necessary lines to plot a grid, i.e the input space
    # Or object on which we apply the geometric transformation
    grids = [np.column_stack((np.linspace(-1,1, 100), k*np.ones(100)/10.)) for k in range(-10,11)] +\
                [np.column_stack((k*np.ones(100)/10.,np.linspace(-1,1, 100))) for k in range(-10,11) ]

    # Stuff below is matplotlib mostly, to define the plot
    fig = plt.figure()
    plt.ion()

    ax = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1, fig=fig)
    ax_loss = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1, fig=fig)
    
    # Below we use the split intermediate to extract the 2dimensional transformed features right before sigmoid
    if cuda_flag:
        orig_vals = intermediate_model(Variable(torch.Tensor(df[['x','y']].values)).cuda()).data.cpu().numpy()
    else:
        orig_vals = intermediate_model(Variable(torch.Tensor(df[['x','y']].values))).data.numpy()

        # Here we extract weights and bias term right before sigmoid
    decision_x, decision_y = decision_boundary(model)
    line, = ax.plot(decision_x,decision_y,color='black')
    lineb, = ax.plot(orig_vals[indb,0].ravel(), orig_vals[indb,1].ravel(), marker='.', color='b')
    liner, = ax.plot(orig_vals[indr,0].ravel(), orig_vals[indr,1].ravel(), marker='.', color='r')
    loss_line, = ax_loss.plot([],[], c='g')
    grid_lines = []
    loss_list = []

    # Below we process the grid that we defined in lines 14-15, the transformation of the grid
    # will help us visiualise what is going on with the space
    for grid in grids:
        if cuda_flag:
            vals = intermediate_model(Variable(torch.Tensor(np.array(grid))).cuda()).data.cpu().numpy()
        else:
            vals = intermediate_model(Variable(torch.Tensor(np.array(grid)))).data.numpy()
        l, = ax.plot(vals[:,0],vals[:,1], color='grey', alpha=.5)
        grid_lines.append(l)

    all_lines = tuple([line, lineb, liner, *grid_lines])
    
    # Below we iterate 10000 times over one_pass of model training, where the parameters are optimised
    # using the batch size of N, in this case 256 samples of data.
    for it in range(10000):
        # iterate over model training
        loss = one_pass(model, loss_criterion, optimizer, df, N=256)

        if it % 100 == 0:
            # Append model loss to the data for plotting
            loss_list.append(loss)
            # Update the decision boundary plot
            line.set_data(*decision_boundary(model))
            # Extract the transformed features
            if cuda_flag:
                vals = intermediate_model(Variable(torch.Tensor(df[['x','y']].values)).cuda()).data.cpu().numpy()
            else:
                vals = intermediate_model(Variable(torch.Tensor(df[['x','y']].values))).data.numpy()
            lineb.set_data(vals[indb,0], vals[indb,1])
            liner.set_data(vals[indr,0], vals[indr,1])
            loss_line.set_data([i for i in range(len(loss_list))], loss_list)

            # Again, process the gridlines through the network and extract intermediate features
            for k in range(len(grid_lines)):
                ln = grid_lines[k]
                grid = grids[k]
                if cuda_flag:
                    vals = intermediate_model(Variable(torch.Tensor(np.array(grid))).cuda()).data.cpu().numpy()
                else:
                    vals = intermediate_model(Variable(torch.Tensor(np.array(grid)))).data.numpy()
                ln.set_data(vals[:,0],vals[:,1])

            # Matplotlib stuff that I dont understand, but it seems to make it work
            ax.autoscale_view(True,True,True)
            ax.relim()
            ax_loss.autoscale_view(True,True,True)
            ax_loss.relim()
            plt.draw()
            fig.canvas.draw()