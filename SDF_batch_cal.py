import numpy as np
import torch 
from scipy.spatial.distance import cdist
import sys
import cv2
import math

def arc_center_radius(p1, p2, p3):
    '''
    calculate the center and radius of a batch of arc defined by three points
    '''
    mid1 = (p1 + p2) / 2  
    mid2 = (p2 + p3) / 2  


    slope1 = (p2[:, 1] - p1[:, 1]) / (p2[:, 0] - p1[:, 0])  
    slope2 = (p3[:, 1] - p2[:, 1]) / (p3[:, 0] - p2[:, 0])  

    perp_slope1 = -1 / slope1  
    perp_slope2 = -1 / slope2  

    perp_slope1[(p2[:, 0] - p1[:, 0]) == 0] = 0  
    perp_slope2[(p3[:, 0] - p2[:, 0]) == 0] = 0

    b1 = mid1[:, 1] - perp_slope1 * mid1[:, 0]  
    b2 = mid2[:, 1] - perp_slope2 * mid2[:, 0]  


    A = torch.stack([-perp_slope1, torch.ones_like(perp_slope1)], dim=1)  # (batch_size, 2)
    B = torch.stack([-perp_slope2, torch.ones_like(perp_slope2)], dim=1)  # (batch_size, 2)


    A_inv = torch.linalg.pinv(torch.stack([A, B], dim=1))  #  (batch_size, 2, 2)
    centers = A_inv @ torch.stack([b1, b2], dim=1).unsqueeze(-1)  #  (batch_size, 2, 1)
    centers = centers.squeeze(-1)  # (batch_size, 2)

    radii = torch.sqrt(torch.sum((centers - p1) ** 2, dim=1))  #  (batch_size,)

    return centers, radii


def cross2d(v1, v2):
    '''cross for 2d vecs shaped like [batch, num ,2]'''
    corss = v1[:, :, 0]*v2[:, :, 1]-v1[:, :, 1]*v2[:, :, 0]
    return corss


def arc_sdf(points, arc):
    """ calculate the sdf of a batch of arc defined by three poitns"""
    start_point, mid_point, end_point  = arc[:, :2],arc[:, 2:4],arc[:, 4:6]
    # clock wise or counter clock wise assert
    start_mid_vec = start_point.unsqueeze(0) - mid_point.unsqueeze(0)
    mid_end_vec = end_point.unsqueeze(0) - mid_point.unsqueeze(0)
    center, radius = arc_center_radius(start_point, mid_point, end_point)
    num_points = points.shape[0]
    batch_size = start_point.shape[0]
    OP = points.unsqueeze(1) - center.unsqueeze(0)
    OA = start_point.unsqueeze(0) - center.unsqueeze(0)
    OB = mid_point.unsqueeze(0) - center.unsqueeze(0)
    OC = end_point.unsqueeze(0) - center.unsqueeze(0)
    AP = points.unsqueeze(1) - start_point.unsqueeze(0)
    CP = points.unsqueeze(1) - end_point.unsqueeze(0)

    crb_a = cross2d(OB.repeat(num_points, 1, 1), OA.repeat(num_points, 1, 1))
    crb_c = cross2d(OB.repeat(num_points, 1, 1), OC.repeat(num_points, 1, 1))
    crp_a = cross2d(OP, OA.repeat(num_points, 1, 1))
    crp_c = cross2d(OP, OC.repeat(num_points, 1, 1))
    cra_c = cross2d(OA.repeat(num_points, 1, 1), OC.repeat(num_points, 1, 1))
    cba_c = torch.sum(start_mid_vec.repeat(num_points, 1, 1)*mid_end_vec.repeat(num_points, 1, 1), -1)

    ACP = torch.min(torch.norm(AP, dim=-1), torch.norm(CP, dim=-1))
    PD = torch.abs(radius.unsqueeze(0) - torch.norm(OP, dim=-1))
    sign = torch.zeros_like(ACP)

    mask1 = (cba_c <= 0).bool() & (((crb_a * crp_a) >= 0).bool() & ((crb_c * crp_c) >= 0).bool()).bool()
    sign[mask1] = 1

    mask2 = (cba_c > 0).bool() & (cra_c > 0).bool() & ~((crp_a < 0).bool() & (crp_c > 0).bool())
    sign[mask2] = 1

    mask3 = (cba_c > 0).bool() & (cra_c <= 0).bool() & ~((crp_a > 0).bool() & (crp_c < 0).bool())
    sign[mask3] = 1
    dist = torch.cat([ACP.unsqueeze(-1), PD.unsqueeze(-1)], dim=-1)
    sign = sign.long()
    eye_mat = torch.eye(2).to(sign)
    sign_one_hot = eye_mat[sign.long()].float()
    dist = dist

    dist = dist.view(batch_size * num_points, 2, 1)
    sign_one_hot = sign_one_hot.view(batch_size * num_points, 1, 2)
    sdf = torch.bmm(sign_one_hot, dist).flatten(end_dim=1)

    sdf = sdf.view(num_points, batch_size)
    sdf, _ = torch.min(sdf, -1)
    return sdf


def line_sdf(points, line_points):
    """SDF calculation of a groupof lines"""
    start_point = line_points[:, :2]
    end_point   = line_points[:, 2:]

    PA = points.unsqueeze(1) - start_point.unsqueeze(0)
    AB = end_point.unsqueeze(0) - start_point.unsqueeze(0)
    
    # import pdb;pdb.set_trace()
    t = torch.sum(PA * AB, dim=-1) / torch.sum(AB * AB, dim=-1)
    t = torch.clamp(t, 0, 1)

    CP = PA - t.unsqueeze(-1) * AB
    sdf = torch.norm(CP, dim=-1)
    sdf, _ = torch.min(sdf, dim=-1)
    return sdf


def points_sdf(eval_points, bound_points, format='torch', clamp=True, clamp_delta=0.1):
    '''SDF calculation of a group of points'''
    dist = torch.cdist(eval_points, bound_points)
    min_dist, _ = torch.min(dist, dim=-1)
    return min_dist


def circle_sdf(points, circle):
    """SDF calculation of a group of circles"""
    center, radius = circle[:, :2], circle[:, 2]
    sdf = torch.abs(torch.norm(points.unsqueeze(1) - center.unsqueeze(0), dim=-1) - radius)
    sdf, _ = torch.min(sdf, -1)
    return sdf


def create_mesh_points(num_x, num_y, x_range=(0.0, 1.0), y_range=(0.0, 1.0)):
    '''
    creat eval grid points
    '''
    x = torch.linspace(x_range[0], x_range[1], num_x)
    y = torch.linspace(y_range[0], y_range[1], num_y)
    xx, yy = torch.meshgrid(x, y)
    points = torch.stack([xx.flatten(), yy.flatten()], axis=-1) 
    return points

if __name__ == "__main__":

    points = create_mesh_points(256, 256).cuda()

    # points 
    seq_gt = torch.tensor([
        [0.5, 0.5], 
        [0.9, 0.9], 
        [0.9, 0.0],
        [0.0, 0.9]]).cuda()
    sdf_gt = points_sdf(points, seq_gt).cuda()
    
    # lines
    seg_gt = torch.tensor([
        [0,   0,   0.5, 0.3],
        [0.5, 0.3, 0.8, 0.7],
        [0.8, 0.7, 0.3, 0.2]]).cuda()
    sdf_gt = line_sdf(points, seg_gt).cuda()

    # circle
    seg_gt = torch.tensor([
        [0,   0,   0.5],
        [0.5, 0.3, 0.8],
        [0.8, 0.7, 0.3]]).cuda()
    sdf_gt = circle_sdf(points, seg_gt).cuda()

    # arc
    seg_gt = torch.tensor([
        # [0, 0,   0.5, 0.5, 0.5*(1+1/math.sqrt(2)), 0.5/math.sqrt(2)],
        # [0, 0,   0.5, 0.5, 1, 0],
        # [0, 0.5, 0.5*(1+1/math.sqrt(2)), 0.5*(1+1/math.sqrt(2)), 0.5, 0]
        [0, 0.5, 0.5, 0, 1, 0.5],
    ]).cuda()
    sdf_gt = arc_sdf(points=points, arc=seg_gt).cuda()


    points = points.detach().cpu().numpy()
    delta = 1
    sdf_gt = torch.clamp(sdf_gt, -delta, delta)
    import matplotlib.pyplot as plt
    plt.scatter(points[:,0], points[:,1], c=sdf_gt.detach().cpu().numpy())
    plt.savefig("test1.png")
    sdf_gt = sdf_gt.reshape((256, 256))
    sdf_gt = sdf_gt.permute(1, 0)
    sdf_gt = ((delta - sdf_gt)/delta)*255
    img = sdf_gt.detach().cpu().numpy()
    cv2.imwrite('test.png', img)
