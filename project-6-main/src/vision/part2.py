import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable

class NerfModel(nn.Module):
    
    def __init__(self, in_channels: int, filter_size: int=256):
        """This network will have a total of 8 fully connected layers. The activation function will be ReLU

        The number of input features to layer 5 will be a bit different. Refer to the docstring for the forward pass.
        Do not include an activation after layer 8 in the Sequential block. Layer 8's should output 4 features.

        Args
        ---
        in_channels (int): the number of input features from 
            the data
        filter_size (int): the number of in/out features for all layers. Layers 1 (because of in_channels), 5, and 8 are
            a bit different.
        """
        super().__init__()

        self.fc_layers_group1: nn.Sequential = None  # For layers 1-3
        self.layer_4: nn.Linear = None
        self.fc_layers_group2: nn.Sequential = None  # For layers 5-8
        self.loss_criterion = None

        ##########################################################################
        # Student code begins here
        ##########################################################################
        
        # raise NotImplementedError('`init` function in `NerfModel` needs to be implemented')

        self.fc_layers_group1 = nn.Sequential(
            nn.Linear(in_channels, filter_size),
            nn.ReLU(True),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(True),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(True),
        )

        self.layer_4 = nn.Linear(filter_size, filter_size)

        self.fc_layers_group2 = nn.Sequential(nn.Linear(filter_size * 2, filter_size), nn.ReLU(True), nn.Linear(filter_size, filter_size), nn.ReLU(True), nn.Linear(filter_size, filter_size), nn.ReLU(True), nn.Linear(filter_size, 4))

        self.loss_criterion = nn.MSELoss()

        def initialize(m):

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        self.apply(initialize)

        ##########################################################################
        # Student code ends here
        ##########################################################################
  
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass of the model. 
        
        NOTE: The input to layer 5 should be the concatenation of post-activation values from layer 4 with 
        post-activation values from layer 3. Therefore, be extra careful about how self.layer_4 is used and what 
        the specified input size to layer 5 should be. The output from layer 5 and the dimensions thereafter should be
        filter_size.
        
        Args
        ---
        x (torch.Tensor): input of shape 
            (batch_size, in_channels)
        
        Returns
        ---
        rgb (torch.Tensor): The predicted rgb values with 
            shape (batch_size, 3)
        sigma (torch.Tensor): The predicted density values with shape (batch_size)
        """
        rgb = None
        sigma = None

        ##########################################################################
        # Student code begins here
        ##########################################################################
       
        # raise NotImplementedError('`forward` function in `NerfModel` needs to be implemented')

        o1 = self.fc_layers_group1(x)
        o4 = F.relu(self.layer_4(o1), True)
        cat45 = torch.cat([o4, o1], dim=1)
        o8 = self.fc_layers_group2(cat45)
        rgb = torch.sigmoid(o8[..., :3])
        sigma = F.relu(o8[..., 3], True)

        ##########################################################################
        # Student code ends here
        ##########################################################################

        return rgb, sigma

def get_rays(height: int, width: int, intrinsics: torch.Tensor, tform_cam2world: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).
    
    Args
    ---
    height (int): 
        the height of an image.
    width (int): the width of an image.
    intrinsics (torch.Tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    
    Returns
    ---
    ray_origins (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the centers of
        each ray. Note that desipte that all ray share the same origin, 
        here we ask you to return the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the
        direction of each ray.
    """
    device = tform_cam2world.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

    ##########################################################################
    # Student code begins here
    ##########################################################################
    
    # raise NotImplementedError('`get_rays()` function needs to be implemented')
    device = intrinsics.device
    x_f, y_f = intrinsics[0,0], intrinsics[1,1]
    x_temp, y_temp = intrinsics[0,2], intrinsics[1,2]

    i = torch.arange(height, device=device)
    j = torch.arange(width, device=device)
    i_mesh, j_mesh = torch.meshgrid(i, j, indexing='ij')

    drc = torch.stack([(j_mesh - x_temp) / x_f, (i_mesh - y_temp) / y_f, torch.ones_like(i_mesh)], dim=-1)
    mat = tform_cam2world[:3, :3]
    t = tform_cam2world[:3, 3]
    ray_directions = drc @ mat.T
    ray_origins = t.view(1,1,3).expand(height, width, 3)

    ##########################################################################
    # Student code ends here
    ##########################################################################

    return ray_origins, ray_directions

def sample_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize:bool = True
) -> tuple[torch.tensor, torch.tensor]:
    """Sample 3D points on the given rays. The near_thresh and far_thresh
    variables indicate the bounds of sampling range.
    
    Args
    ---
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    query_points (torch.Tensor): Query 3D points along each ray
        (shape: :math:`(height, width, num_samples, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    """
    device = ray_origins.device
    height, width = ray_origins.shape[:2]
    depth_values = torch.zeros((height, width, num_samples), device=device) # placeholder
    query_points = torch.zeros((height, width, num_samples, 3), device=device) # placeholder
    
    ##########################################################################
    # Student code begins here
    ##########################################################################
    
    # raise NotImplementedError('`sample_points_from_rays()` function needs to be implemented')

    device = ray_origins.device
    H, W = ray_origins.shape[:2]
    
    if randomize:
        index = torch.arange(num_samples, device=device, dtype=torch.float32)
        values_t = index / num_samples
        values_z = near_thresh * (1- values_t) + far_thresh * values_t
        mids = 0.5 * (values_z[:-1] + values_z[1:])
        high = torch.cat([mids, values_z[-1:]], dim=0)
        lower = torch.cat([values_z[:1], mids], dim=0)
        random_t = torch.rand((H, W, num_samples), device=device)
        depth_values = lower + (high - lower) * random_t

    else:
        index = torch.arange(num_samples, dtype=torch.float32, device=device)
        values_t = index / num_samples
        values_z = near_thresh * (1 -values_t) + far_thresh * values_t
        depth_values = values_z.view(1, 1, num_samples).expand(H, W,num_samples)

    query_points = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * depth_values.unsqueeze(-1)


    ##########################################################################
    # Student code ends here
    ##########################################################################
    
    return query_points, depth_values

def cumprod_exclusive(x: torch.tensor) -> torch.tensor:
    """ Helper function that computes the cumulative product of the input tensor, excluding the current element
    Example:
    > cumprod_exclusive(torch.tensor([1,2,3,4,5]))
    > tensor([ 1,  1,  2,  6, 24])
    
    Args:
    -   x: Tensor of length N
    
    Returns:
    -   cumprod: Tensor of length N containing the cumulative product of the tensor
    """

    cumprod = torch.cumprod(x, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod

def compute_compositing_weights(sigma: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
    """This function will compute the compositing weight for each query point.

    Args
    ---
    sigma (torch.Tensor): Volume density at each query location (X, Y, Z)
        (shape: :math:`(height, width, num_samples)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    
    Returns:
    weights (torch.Tensor): Rendered compositing weight of each sampled point 
        (shape: :math:`(height, width, num_samples)`).
    """

    device = depth_values.device
    weights = torch.ones_like(sigma, device=device) # placeholder

    ##########################################################################
    # Student code begins here
    ##########################################################################
    
    # raise NotImplementedError('`compute_compositing_weights()` function needs to be implemented')
    delts = depth_values[..., 1:] - depth_values[..., :-1]
    delt_infa = 1e10 * torch.ones_like(delts[..., :1])
    delts = torch.cat([delts, delt_infa], dim=-1)
    alpha = 1. - torch.exp(-sigma * delts)
    T = cumprod_exclusive(torch.exp(-sigma * delts))
    w = alpha * T
    weights = w
    ##########################################################################
    # Student code ends here
    ##########################################################################

    return weights

def get_minibatches(inputs: torch.Tensor, csize: int = 1024 * 32) -> list[torch.Tensor]:
    """Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `csize`.
    """
    return [inputs[i:i + csize] for i in range(0, inputs.shape[0], csize)]

def render_image_nerf(height: int, width: int, intrinsics: torch.tensor, tform_cam2world: torch.tensor,
                      near_thresh: float, far_thresh: float, depth_samples_per_ray: int,
                      encoding_function: Callable, model:NerfModel, rand:bool=False) \
                      -> tuple[torch.Tensor, torch.Tensor]:
    """ This function will utilize all the other rendering functions that have been implemented in order to sample rays,
    pass those rays to the NeRF model to get color and density predictions, and then use volume rendering to create
    an image of this view. 

    Hints: 
    ---
    It is a good idea to "flatten" the height/width dimensions of the data when passing to the NeRF (maintain the color
    channel dimension) and then "unflatten" the outputs. 
    To avoid running into memory limits, it's recommended to use the given get_minibatches() helper function to 
    divide up the input into cs. For each minibatch, supply them to the model and then concatenate the corresponding
    output vectors from each minibatch to form the complete outpute vectors. 
    
    Args
    ---
    height (int): 
        the pixel height of an image.
    width (int): the pixel width of an image.
    intrinsics (torch.tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    depth_samples_per_ray (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    encoding_function (Callable): The function used to encode the query points (e.g. positional encoding)
    model (NerfModel): The NeRF model that will be used to render this image
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    rgb_predicted (torch.tensor): 
        A tensor of shape (height, width, num_channels) with the color info at each pixel.
    depth_predicted (torch.tensor): A tensor of shape (height, width) containing the depth from the camera at each pixel.
    """

    rgb_predicted, depth_predicted = None, None

    ##########################################################################
    # Student code begins here
    ##########################################################################
    
    #raise NotImplementedError('`render_image_nerf()` function needs to be implemented')
    
    ray_o, ray_d = get_rays(height, width, intrinsics, tform_cam2world)
    pts, values_z = sample_points_from_rays(ray_o, ray_d, near_thresh, far_thresh, depth_samples_per_ray, randomize=rand)
    H, W, N = values_z.shape

    pts_flat = pts.reshape(-1, 3)
    enc = encoding_function(pts_flat)
    rgb_chunks = []
    sigma_chunks = []

    for chunk in get_minibatches(enc):
        out_rgb_c, out_sig_c = model(chunk)
        rgb_chunks.append(out_rgb_c)
        sigma_chunks.append(out_sig_c)

    flatt_rgb = torch.cat(rgb_chunks, dim=0)
    flatt_sigma = torch.cat(sigma_chunks, dim=0)
    rgb = flatt_rgb.view(H, W, N, 3)
    sigma = flatt_sigma.view(H, W, N)

    weights = compute_compositing_weights(sigma, values_z)

    rgb_predicted = torch.sum(weights.unsqueeze(-1) * rgb, dim=2)
    depth_predicted = torch.sum(weights * values_z, dim=2)

    ##########################################################################
    # Student code ends here
    ##########################################################################

    return rgb_predicted, depth_predicted