"""
   GET INFORMATION FROM NEURAL NETWORK
     Reference: https://github.com/ravidziv/IDNNs
"""
import numpy as np
import torch


# give probabilities to discretized values of the outputs of layers
def extract_probs(label):
    """
        calculate the probabilities of the given data and labels p(y)
    """
    # pys = np.sum(label, axis=0) / float(label.shape[0])
    label = label.cpu().detach().numpy()
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize)))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    return torch.tensor(pys1), torch.tensor(unique_inverse_y)


def calc_entropy_for_T_KDE(data, pxs, device, cov, bigHS=False):
    size = data.shape[0]
    dim = data.shape[1]
    pa = torch.zeros(size, size)
    for i in range(size):
        pa[i] = torch.sum(torch.pow((data - data[i]), 2), dim=tuple(np.arange(1, len(data.shape))))
    distance = pa.sqrt()

    if len(data.shape) == 2:
        lamda = 0.01
    else:
        lamda = 0.04
    sigma = np.sqrt(lamda * cov)

    num = torch.zeros(3)
    freq = torch.zeros(3)
    for i in range(3):
        num[i] = (((distance < ((i + 1) * sigma)).sum()) - size) * 1.0 / size
        freq[i] = num[i] * 1.0 / (size - 1)

    prob = torch.exp(-pa / (2 * lamda * cov))
    pts = torch.sum(prob, 1) / size
    H2 = - torch.log2(pts).sum() / size
    H2 = pxs * H2
    if bigHS:
        _, idx = torch.sort(pts)
        idx_for_HS = idx[:10]
        return H2, idx_for_HS
    else:
        return H2, lamda, num, freq


def calc_pz(zi, z, cov):
    size = z.shape[0]
    pa = torch.sum(torch.pow(zi - z, 2), dim=tuple(np.arange(1, len(z.shape))))
    pts = 1 / size * torch.sum(torch.exp(-pa / (2 * cov)))
    return pts


def create_fake_data(distrib, size):
    fake_data = torch.zeros(size)
    for i in range(size[0]):
        fake_data[i] = (torch.rand(size[1:]) < distrib.cpu()).float()
    return fake_data


def calc_information_sampling_KDE(data, pys1, unique_inverse_y, device, TC, fake, bigHS=False):
    data = data.to(device)
    size = data.shape[0]

    cov = np.prod(np.array(data.shape[1:])).item()

    if bigHS:
        H2, idx = calc_entropy_for_T_KDE(data, 1, device, cov, bigHS=bigHS)
    else:
        H2, lamda, num, freq = calc_entropy_for_T_KDE(data, 1, device, cov)
    IX = H2

    PYs = pys1.T

    H2Y_array = torch.zeros(PYs.shape[0])
    for i in range(PYs.shape[0]):
        index = unique_inverse_y == i
        H2Y_array[i], _, _, _ = calc_entropy_for_T_KDE(data[index, :], PYs[i], device, cov, TC)
    H2Y = torch.sum(H2Y_array)
    IY = H2 - H2Y

    params = {}
    params['local_IXT'] = IX.cpu().detach().numpy()
    params['local_ITY'] = IY.cpu().detach().numpy()
    params['cover_num'] = num.cpu().detach().numpy()
    params['cover_freq'] = freq.cpu().detach().numpy()
    params['lamda'] = lamda
    if bigHS:
        params['bigHS_ind'] = idx

    if TC:
        distrib = torch.sum(data, axis=0) / size
        fake_data = create_fake_data(distrib, data.shape).to(device)
        if fake:
            cov_tmp = torch.zeros([size])
            for i in range(size):
                if len(data.shape) == 4:
                    pa = torch.sum(torch.sum(torch.sum(torch.pow(fake_data - data[i], 2), 1), 1), 1)
                elif len(data.shape) == 3:
                    pa = torch.sum(torch.sum(torch.pow(fake_data - data[i], 2), 1), 1)
                elif len(data.shape) == 2:
                    pa = torch.sum(torch.pow(fake_data - data[i], 2), 1)
                pb, _ = torch.sort(pa)
                cov_tmp[i] = torch.mean(pb)
            cov_fake = torch.mean(cov_tmp)
        #######################################
        pa = torch.zeros([size])
        pf = torch.zeros([size])
        for i in range(size):
            pa[i] = calc_pz(data[i], data, torch.tensor([cov[0], 1e-5]).max())
            if fake:
                pf[i] = calc_pz(data[i], fake_data, torch.tensor([cov_fake, 1e-5]).max())
            else:
                pf[i] = calc_pz(data[i], fake_data, torch.tensor([cov[0], 1e-5]).max())
        TC = torch.log2(pa / pf).sum() / size

        CL = 0
        for act_rate in distrib.flatten():
            act_rate = torch.tensor([act_rate, 1e-5]).max()
            CL += -torch.log2(act_rate)

        params['local_TC'] = TC.cpu().detach().numpy()
        params['local_CL'] = CL.cpu().detach().numpy()
    return params


def calc_information_for_epoch_KDE(ws_iter_index, device, TC=False, fake=False, bigHS=False):
    pys1, unique_inverse_y = extract_probs(ws_iter_index[-1])
    pys1 = pys1.to(device)
    unique_inverse_y = unique_inverse_y.to(device)
    params = np.array(
        [calc_information_sampling_KDE(ws_iter_index[i], pys1, unique_inverse_y, device, TC=TC, fake=fake, bigHS=bigHS)
         for i in range(len(ws_iter_index) - 1)]
    )

    return params
