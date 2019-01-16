from collections import OrderedDict

import torch
import pandas as pd
import numpy as np
import cv2

screen = [416, 416]
GRID_SIZE = 13
def transform_coord(coord, cell):
    cy = cell % GRID_SIZE
    cx = cell // GRID_SIZE
    w = coord[2] * screen[0]
    h = coord[3] * screen[1]
    x = (coord[0] + cx)*(screen[0] / GRID_SIZE ) - (w/2)
    y = (coord[1] + cy)*(screen[1] / GRID_SIZE ) - (h/2)
    return (x, y), w, h

def get_abs_coord(box):
    box[2], box[3] = abs(box[2]), abs(box[3])
    x1 = (box[0] - box[2]/2) - 1
    y1 = (box[1] - box[3]/2) - 1
    x2 = (box[0] + box[2]/2) - 1
    y2 = (box[1] + box[3]/2) - 1
    return x1, y1, x2, y2


def sanity_fix(box):
    if box[0] > box[2]:
        box[0], box[2] = box[2], box[0]

    if box[1] > box[3]:
        box[1], box[3] = box[3], box[1]

    return box


def pred_corner_coord(prediction):
    # Get indices of non-zero confidence bboxes
    ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()

    box = prediction[ind_nz[0], ind_nz[1]]

    box_a = box.new(box.shape)
    box_a[:, 0] = (box[:, 0] - box[:, 2] / 2)
    box_a[:, 1] = (box[:, 1] - box[:, 3] / 2)
    box_a[:, 2] = (box[:, 0] + box[:, 2] / 2)
    box_a[:, 3] = (box[:, 1] + box[:, 3] / 2)
    box[:, :4] = box_a[:, :4]

    prediction[ind_nz[0], ind_nz[1]] = box
    return prediction


def write(x, batches, results, colors, classes):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


def process_csv(csv_file, test_fraction, mode='train'):
    df = pd.read_csv(csv_file)
    data_dict = OrderedDict()
    n = int(len(df['Name'].unique()) * (1-test_fraction))
    name = df['Name'].unique()[n]
    name_idx = list(df['Name']).index(name)
    low = 0 if mode == 'train' else name_idx
    high = name_idx if mode == 'train' else len(df)
    for i in range(low, high):
        img_filename = df.iloc[i, 0]
        img_w = df.iloc[i, 1]
        img_h = df.iloc[i, 2]

        img_sq_a = max(img_w, img_h)
        delta_x = (img_sq_a - img_w) // 2
        delta_y = (img_sq_a - img_h) // 2

        xmin = df.iloc[i, 3] + delta_x
        ymin = df.iloc[i, 4] + delta_y
        xmax = df.iloc[i, 5] + delta_x
        ymax = df.iloc[i, 6] + delta_y

        box = [0, 0, 0, 0]
        box[0] = float(xmin + xmax) / 2
        box[1] = float(ymin + ymax) / 2
        box[2] = float(xmax - xmin)
        box[3] = float(ymax - ymin)
        try:
            data_dict[img_filename]['bbox'].append(box)
        except:
            data_dict[img_filename] = {'bbox': [box], 'img_dim': [img_w, img_h]}

    return data_dict


def get_gt(img_dim, grid_size, boxes):
    img_w, img_h = img_dim
    img_sq_a = max(img_w, img_h)
    # delta_x = (img_sq_a - img_w) / 2
    # delta_y = (img_sq_a - img_h) / 2
    grid_a = img_sq_a / grid_size

    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # x += delta_x
    # y += delta_y
    cell_x = np.floor(x / grid_a)
    offset_x = (x / grid_a) - cell_x
    cell_y = np.floor(y / grid_a)
    offset_y = (y / grid_a) - cell_y
    rel_w = w / img_sq_a
    rel_h = h / img_sq_a
    idx = cell_x * grid_size + cell_y
    tbox = np.zeros((grid_size**2, 4), dtype='float32')
    tprob = np.zeros((grid_size**2, 1), dtype='int64')
    offset_x = offset_x.reshape(-1, 1)
    offset_y = offset_y.reshape(-1, 1)
    rel_w = rel_w.reshape(-1, 1)
    rel_h = rel_h.reshape(-1, 1)

    for i in range(grid_size**2):
        # one cell might have many ground truth boxes
        box_ind = np.where(idx.reshape(-1) == i)[0]
        if len(box_ind) > 0:
            boxes_in_cell_i = np.concatenate([offset_x[box_ind], offset_y[box_ind], rel_w[box_ind], rel_h[box_ind]], axis=-1)
            # only 1 ground truth boxes can be taken at max
            boxes_in_cell_i = boxes_in_cell_i[:1]
            boxes_in_cell_i = boxes_in_cell_i.reshape(-1)
            tbox[i, :] = boxes_in_cell_i
            tprob[i, 0] = 1

    return tprob.reshape(grid_size, grid_size, 1), tbox.reshape(grid_size, grid_size, 4)
