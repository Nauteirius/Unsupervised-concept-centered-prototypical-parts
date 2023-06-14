import time
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from helpers import list_of_distances, make_one_hot
import matplotlib.pyplot as plt


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, device=torch.device("cuda")):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''

    
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    
    for data, label in dataloader:
        image = data['IMG']
        batch_size = image.shape[0]
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances, prototype_activations, activation_masks = model(input)
            
            activation_masks = F.interpolate(activation_masks, (224, 224))
            activation_masks_cpu = activation_masks.cpu().detach().numpy()

    
    
    
            part_0 = rgb_to_grayscale(data['PART_0'].to(device))
            part_1 = rgb_to_grayscale(data['PART_1'].to(device))
            part_2 = rgb_to_grayscale(data['PART_2'].to(device))
            part_3 = rgb_to_grayscale(data['PART_3'].to(device))
            
            prototype_activations = prototype_activations.reshape(batch_size, 4, 20).detach()


#             print("-"*50)
 
#             # Define the grid layout for visualization
#             rows = 8
#             cols = 10

#             # Create a figure and axes for the grid
#             fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

#             # Flatten the axes array for easier iteration
#             axes = axes.flatten()

#             print("SHAPE")
#             print(activation_masks_cpu[0].shape)
#             # Iterate over the activation masks and plot them
#             for i, mask in enumerate(activation_masks_cpu[0]):
#                 # Normalize the mask values to [0, 1]
#                 # normalized_mask = (mask - mask.min()) / (mask.max() - mask.min())

#                 # Plot the mask on the corresponding axis
#                 axes[i].imshow(mask)
#                 axes[i].axis('off')

#             # Remove any extra subplots if there are fewer masks than grid cells
#             if len(activation_masks[0]) < len(axes):
#                 for j in range(len(activation_masks), len(axes)):
#                     fig.delaxes(axes[j])

#             # Adjust spacing and layout
#             fig.tight_layout()
#             # Display the plot
#             plt.show()
            
            part_0_activation_masks = activation_masks[:, :20]
            part_0_activations = part_0 * part_0_activation_masks
            max_part_0_activations = torch.amax(part_0_activations, dim=(2, 3))
            part_0_loss = torch.sum(torch.abs(max_part_0_activations - prototype_activations[:, 0]))

            part_1_activation_masks = activation_masks[:, 20:40]
            part_1_activations = part_1 * part_1_activation_masks
            max_part_1_activations = torch.amax(part_1_activations, dim=(2, 3))
            part_1_loss = torch.sum(torch.abs(max_part_1_activations - prototype_activations[:, 1]))

            part_2_activation_masks = activation_masks[:, 40:60]
            part_2_activations = part_2 * part_2_activation_masks
            max_part_2_activations = torch.amax(part_2_activations, dim=(2, 3))
            part_2_loss = torch.sum(torch.abs(max_part_2_activations - prototype_activations[:, 2]))

            part_3_activation_masks = activation_masks[:, 60:80]
            part_3_activations = part_3 * part_3_activation_masks
            max_part_3_activations = torch.amax(part_3_activations, dim=(2, 3))
            part_3_loss = torch.sum(torch.abs(max_part_3_activations - prototype_activations[:, 3]))

            total_loss = part_0_loss + part_1_loss + part_2_loss + part_3_loss
                        
            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            # total_loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\t accu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
    log('\t total parts loss: \t{0}'.format(total_loss))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\twarm')
    


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')