from itertools import combinations
from PIL import Image
import numpy as np
import torch
import cv2

def CLR2Gray(src_img, rgb_array): # To support PIL format
    dst_img = Image.new('L', (src_img.size[0], src_img.size[1]), 255)
    pix_src_img = src_img.load()
#    max_tmp = rgb_array[0] * 255 + rgb_array[1] * 255 + rgb_array[2] * 255
    for w_i in range (src_img.size[0]):
        for h_i in range (src_img.size[1]):
            pix_src_Tuple = pix_src_img[w_i, h_i]
            tmp = int(rgb_array[0] * pix_src_Tuple[0] + rgb_array[1] * pix_src_Tuple[1] +\
                  rgb_array[2] * pix_src_Tuple[2])
#            tmp = int(tmp/max_tmp * 255)
            dst_img.putpixel((w_i, h_i), tmp)
    return dst_img




def LocateCrop(img):
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
    radius = round(img.size[1] / 2)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("binary.jpg", binary)
    w = img.size[0]
    pix_count_array = np.zeros(w, dtype=int)
    for w_i in range(0, w):
        pix_count_array[w_i] = np.sum(binary[:, w_i])
    top_k_idx = pix_count_array.argsort()[::-1][0:200]
    mean_center = int(np.mean(top_k_idx))
#    print("Center of tealeaf is : ", mean_center)
    margin = 50
    x0 = mean_center - radius - margin
    if x0 < 0:
        x0 = 0
    x1 = mean_center + radius + margin
    if x1 > img.size[0]:
        x1 = img.size[0]
    y0 = 0
    y1 = img.size[1]
    crop_array = (x0, y0, x1, y1)
    return crop_array

def pdist(vectors):
    v1 = vectors[0]
    v2 = vectors[1]
    v3 = vectors[2]
    vectors = torch.stack((v1, v2, v3), 0)
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
#    distance_matrix[0] = -2 * vectors_0.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
#        dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(*embeddings)


        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        print("embedding is ", len(embeddings), ", ", len(embeddings[0]))
        distance_matrix = pdist(embeddings)
#        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        labels = labels[0]
        labels_0 = labels[0].numpy()
        labels_1 = labels[1].numpy()
        labels_2 = labels[2].numpy()
        labels = (labels_0, labels_1, labels_2)
        print(labels)
#        labels = []
#        labels[0] = labels_0
#        labels[1] = labels_1
#        labels[2] = labels_2
#        labels.append(labels_0)
#        labels.append(labels_1)
#        labels.append(labels_2)
        triplets = []

#        for label in set(labels):
        for label in labels:
            print("label ", label)
            label_mask = (labels == label)
            print(label_mask)
            label_indices = np.where(label_mask)[0]
            print("label_indices: ", label_indices)
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            print(negative_indices)
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            print("anchor_positives: ", anchor_positives)
            anchor_positives = np.array(anchor_positives)
            print(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            print("ap_distances: ", ap_distances)
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)
