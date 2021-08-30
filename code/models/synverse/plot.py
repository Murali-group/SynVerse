def plot_loss(model, edge_type, pos_edges_split_dict,neg_edges_split_dict,\
              edge_type_wise_number_of_subtypes, idx_2_cell_line, train_or_val, min_loss, epoch):
    total_loss = 0
    for edge_sub_type in range(edge_type_wise_number_of_subtypes[edge_type]):
        cell_line_wise_loss = 0
        for split_idx in range(len(pos_edges_split_dict[edge_type][edge_sub_type])):
            pos_edges = pos_edges_split_dict[edge_type][edge_sub_type][split_idx].to(dev)
            neg_edges = neg_edges_split_dict[edge_type][edge_sub_type][split_idx].to(dev)
            batch_wise_pos_pred, batch_wise_neg_pred, loss = val(model, pos_edges,
                                                                       neg_edges, edge_type,
                                                                       edge_sub_type)
            cell_line_wise_loss += loss
        total_loss += cell_line_wise_loss.to('cpu').detach().item()

        if edge_type == 'drug_drug':
            cell_line = idx_2_cell_line[edge_sub_type]
            cell_line_wise_title = train_or_val + '_loss_' + cell_line
            wandb.log({cell_line_wise_title: cell_line_wise_loss}, step=wandb_step)

    # if total_val_loss.to('cpu').detach().numpy()[0] < min_val_loss.to('cpu').detach().numpy()[0]:
    if total_loss < min_loss:
        print(total_loss, min_loss)
        min_loss = total_loss

        print('current minimum ' + train_or_val+ ' loss = ', min_loss, ' at epoch: ', epoch)

    wandb.log({train_or_val+ '_loss_'+ edge_type: total_loss}, step=wandb_step)