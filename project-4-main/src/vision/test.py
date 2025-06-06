#!/usr/bin/python3

import logging
import os
import pdb
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data


# from mseg.utils.mask_utils_detectron2 import Visualizer
# from mseg.utils.resize_util import resize_img_by_short_side

# from mseg_semantic.utils.cv2_video_utils import VideoWriter, VideoReader
# from mseg_semantic.utils import dataset, transform, config

import src.vision.cv2_transforms as transform
from src.vision.avg_meter import AverageMeter
from src.vision.part2_dataset import SemData
from src.vision.part5_pspnet import PSPNet
from src.vision.part4_segmentation_net import SimpleSegmentationNet
from src.vision.trainer import DEFAULT_ARGS
from src.vision.utils import load_class_names, get_imagenet_mean_std, get_logger, normalize_img
from src.vision.accuracy_calculator import AccuracyCalculator

"""
Modified from https://github.com/mseg-dataset/mseg-semantic/blob/master/mseg_semantic/tool/inference_task.py

Given a specified task, run inference on it using a pre-trained network.
Used for demos, and for testing on an evaluation dataset.

Note: "base size" should be the length of the shorter side of the desired
inference image resolution. "base_size" is a very important parameter and will
affect results significantly.

model_taxonomy='test_dataset', eval_taxonomy='test_dataset'
    --> means evaluating `oracle` model -- trained and tested on same
		dataset (albeit on separate splits)
"""

_ROOT = Path(__file__).resolve().parent.parent.parent


logger = get_logger()



def create_test_loader(args) -> Tuple[torch.utils.data.dataloader.DataLoader, List[Tuple[str,str]]]:
    """Create a Pytorch dataloader from a dataroot and list of relative paths.
    
    Args:
        args: CfgNode object
        use_batched_inference: whether to process images in batch mode
    
    Returns:
        test_loader
        data_list: list of 2-tuples (relative rgb path, relative label path)
    """
    # no resizing on the fly using OpenCV and also normalize images on the fly
    test_transform = transform.Compose([transform.ToTensor()])

    test_data = SemData(
        split=args.split,
        data_root=args.data_root,
        data_list_fpath=args.test_list,
        transform=test_transform
    )

    data_list = test_data.data_list
    
    # limit batch size to 1
    batch_size = 1
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    return test_loader, data_list


def resize_by_scaled_short_side(image: np.ndarray, base_size: int, scale: float) -> np.ndarray:
    """Equivalent to ResizeShort(), but functional, instead of OOP paradigm, and w/ scale param.

    Args:
        image: Numpy array of shape ()
        scale: scaling factor for image

    Returns:
        image_scaled:
    """
    h, w, _ = image.shape
    short_size = round(scale * base_size)
    new_h = short_size
    new_w = short_size
    # Preserve the aspect ratio
    if h > w:
        new_h = round(short_size / float(w) * h)
    else:
        new_w = round(short_size / float(h) * w)
    image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image_scaled


def pad_to_crop_sz(
    image: np.ndarray, crop_h: int, crop_w: int, mean: Tuple[float, float, float]
) -> Tuple[np.ndarray, int, int]:
    """
    Network input should be at least crop size, so we pad using mean values if
    provided image is too small. No rescaling is performed here.

    We use cv2.copyMakeBorder to copy the source image into the middle of a
    destination image. The areas to the left, to the right, above and below the
    copied source image will be filled with extrapolated pixels, in this case the
    provided mean pixel intensity.

    Args:
        image:
        crop_h: integer representing crop height
        crop_w: integer representing crop width

    Returns:
        image: Numpy array of shape (crop_h x crop_w) representing a
               square image, with short side of square is at least crop size.
         pad_h_half: half the number of pixels used as padding along height dim
         pad_w_half" half the number of pixels used as padding along width dim
    """
    orig_h, orig_w, _ = image.shape
    pad_h = max(crop_h - orig_h, 0)
    pad_w = max(crop_w - orig_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            src=image,
            top=pad_h_half,
            bottom=pad_h - pad_h_half,
            left=pad_w_half,
            right=pad_w - pad_w_half,
            borderType=cv2.BORDER_CONSTANT,
            value=mean,
        )
    return image, pad_h_half, pad_w_half


def imread_rgb(img_fpath: str) -> np.ndarray:
    """
    Returns:
        RGB 3 channel nd-array with shape H * W * 3
    """
    bgr_img = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(rgb_img)
    return rgb_img


class InferenceTask:
    def __init__(
        self,
        args,
        base_size: int,
        crop_h: int,
        crop_w: int,
        input_file: str,
        model_taxonomy: str,
        eval_taxonomy: str,
        scales: List[float],
        device: torch.device = torch.device('cuda'),
    ) -> None:
        """
        We always use the ImageNet mean and standard deviation for normalization.
        mean: 3-tuple of floats, representing pixel mean value
        std: 3-tuple of floats, representing pixel standard deviation

        'args' should contain at least 5 fields (shown below).
        See brief explanation at top of file regarding taxonomy arg configurations.

        Args:
            args: experiment configuration arguments
            base_size: shorter side of image
            crop_h: integer representing crop height, e.g. 473
            crop_w: integer representing crop width, e.g. 473
            input_file: could be absolute path to .txt file, .mp4 file, or to a directory full of jpg images
            model_taxonomy: taxonomy in which trained model makes predictions
            eval_taxonomy: taxonomy in which trained model is evaluated
            scales: floats representing image scales for multi-scale inference
            device: device to run inference on
        """
        self.args = args

        # Required arguments:
        assert isinstance(self.args.save_folder, str)
        assert isinstance(self.args.dataset, str)
        assert isinstance(self.args.print_freq, int)
        assert isinstance(self.args.num_model_classes, int)
        assert isinstance(self.args.model_path, str)
        self.num_model_classes = self.args.num_model_classes

        self.base_size = base_size
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.input_file = input_file
        self.model_taxonomy = model_taxonomy
        self.eval_taxonomy = eval_taxonomy
        self.scales = scales
        self.device = device

        self.mean, self.std = get_imagenet_mean_std()
        self.model = self.load_model(args)
        self.softmax = nn.Softmax(dim=1)

        self.gray_folder = None  # optional, intended for dataloader use
        self.data_list = None  # optional, intended for dataloader use

        if model_taxonomy == "universal" and eval_taxonomy == "universal":
            # See note above.
            # no conversion of predictions required
            self.num_eval_classes = self.num_model_classes

        elif model_taxonomy == "test_dataset" and eval_taxonomy == "test_dataset":
            # no conversion of predictions required
            self.num_eval_classes = len(load_class_names(args.dataset))

        if self.args.arch == "PSPNet":
            assert isinstance(self.args.zoom_factor, int)

        # `id_to_class_name_map` only used for visualizing universal taxonomy
        self.id_to_class_name_map = {i: classname for i, classname in enumerate(load_class_names(args.dataset))}

        # indicate which scales were used to make predictions
        # (multi-scale vs. single-scale)
        self.scales_str = "ms" if len(args.scales) > 1 else "ss"

    def load_model(self, args):
        """Load Pytorch pre-trained model from disk of type torch.nn.DataParallel.

        Note that `args.num_model_classes` will be size of logits output.

        Args:
            args:

        Returns:
            model
        """
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        if args.arch == "PSPNet":
            model = PSPNet(
                layers=args.layers,
                num_classes=args.classes,
                zoom_factor=args.zoom_factor,
                criterion=criterion,
                pretrained=False,
                use_ppm=args.use_ppm
            )
        elif args.arch == "SimpleSegmentationNet":
            model = SimpleSegmentationNet(
                pretrained=True,
                num_classes=args.classes,
                criterion=criterion
            )
        

        # logger.info(model)
        if self.device.type == 'cuda':
            cudnn.benchmark = True

        if os.path.isfile(args.model_path):
            logger.info(f"=> loading checkpoint '{args.model_path}'")
            checkpoint = torch.load(args.model_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            logger.info(f"=> loaded checkpoint '{args.model_path}'")
        else:
            raise RuntimeError(f"=> no checkpoint found at '{args.model_path}'")
        model = model.to(self.device)
        return model

    def execute(self) -> None:
        """
        Execute the demo, i.e. feed all of the desired input through the
        network and obtain predictions. Gracefully handles .txt,
        or video file (.mp4, etc), or directory input.
        """
        logger.info(">>>>>>>>>>>>>> Start inference task >>>>>>>>>>>>>")
        self.model.eval()

        if self.input_file is None and self.args.dataset != "default":
            # evaluate on a train or test dataset
            test_loader, self.data_list = create_test_loader(self.args)
            self.execute_on_dataloader(test_loader)
            logger.info("<<<<<<<<< Inference task completed <<<<<<<<<")
            return

        suffix = self.input_file[-4:]
        is_img = suffix in [".png", ".jpg"]

        if is_img:
            self.render_single_img_pred()
        else:
            logger.info("Error: Unknown input type")

        logger.info("<<<<<<<<<<< Inference task completed <<<<<<<<<<<<<<")

    # def render_single_img_pred(self, min_resolution: int = 1080):
    #     """Since overlaid class text is difficult to read below 1080p, we upsample predictions."""
    #     in_fname_stem = Path(self.input_file).stem
    #     output_gray_fpath = f"{in_fname_stem}_gray.jpg"
    #     output_demo_fpath = f"{in_fname_stem}_overlaid_classes.jpg"
    #     logger.info(f"Write image prediction to {output_demo_fpath}")

    #     rgb_img = imread_rgb(self.input_file)
    #     pred_label_img = self.execute_on_img(rgb_img)

    #     # avoid blurry images by upsampling RGB before overlaying text
    #     if np.amin(rgb_img.shape[:2]) < min_resolution:
    #         rgb_img = resize_img_by_short_side(rgb_img, min_resolution, "rgb")
    #         pred_label_img = resize_img_by_short_side(pred_label_img, min_resolution, "label")

    #     metadata = None
    #     frame_visualizer = Visualizer(rgb_img, metadata)
    #     overlaid_img = frame_visualizer.overlay_instances(
    #         label_map=pred_label_img, id_to_class_name_map=self.id_to_class_name_map
    #     )
    #     imageio.imwrite(output_demo_fpath, overlaid_img)
    #     imageio.imwrite(output_gray_fpath, pred_label_img)

    def execute_on_img(self, image: np.ndarray) -> np.ndarray:
        """
        Rather than feeding in crops w/ sliding window across the full-res image, we
        downsample/upsample the image to a default inference size. This may differ
        from the best training size.

        For example, if trained on small images, we must shrink down the image in
        testing (preserving the aspect ratio), based on the parameter "base_size",
        which is the short side of the image.

        Args:
            image: Numpy array representing RGB image

        Returns:
            gray_img: prediction, representing predicted label map
        """
        h, w, _ = image.shape
        is_single_scale = len(self.scales) == 1

        if is_single_scale:
            # single scale, do addition and argmax on CPU
            image_scaled = resize_by_scaled_short_side(image, self.base_size, self.scales[0])
            prediction = torch.Tensor(self.scale_process_cuda(image_scaled, h, w))

        else:
            # multi-scale, prefer to use fast addition on the GPU
            prediction = np.zeros((h, w, self.num_eval_classes), dtype=float)
            prediction = torch.Tensor(prediction)
            prediction = prediction.to(self.device)
            for scale in self.scales:
                image_scaled = resize_by_scaled_short_side(image, self.base_size, scale)
                scaled_prediction = torch.Tensor(self.scale_process_cuda(image_scaled, h, w))
                scaled_prediction = scaled_prediction.to(self.device)
                prediction = prediction + scaled_prediction

        prediction /= len(self.scales)
        prediction = torch.argmax(prediction, axis=2)
        prediction = prediction.data.cpu().numpy()
        gray_img = np.uint8(prediction)
        return gray_img


    def execute_on_dataloader(self, test_loader: torch.utils.data.dataloader.DataLoader) -> None:
        """Run a pretrained model over each batch in a dataloader.

        Args:
             test_loader:
        """
        if self.args.save_folder == "default":
            self.args.save_folder = f"{_ROOT}/temp_files/{self.args.model_name}_{self.args.dataset}_universal_{self.scales_str}/{self.args.base_size}"

        os.makedirs(self.args.save_folder, exist_ok=True)
        gray_folder = os.path.join(self.args.save_folder, "gray")
        self.gray_folder = gray_folder

        data_time = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        os.makedirs(self.gray_folder, exist_ok=True)

        for i, (input, _) in enumerate(test_loader):
            logger.info(f"On image {i}")
            data_time.update(time.time() - end)

            # determine path for grayscale label map
            image_path, _ = self.data_list[i]
            image_name = Path(image_path).stem

            gray_path = os.path.join(self.gray_folder, image_name + ".png")
            if Path(gray_path).exists():
                continue

            # convert Pytorch tensor -> Numpy, then feedforward
            input = np.squeeze(input.numpy(), axis=0)
            image = np.transpose(input, (1, 2, 0))
            gray_img = self.execute_on_img(image)

            batch_time.update(time.time() - end)
            end = time.time()
            cv2.imwrite(gray_path, gray_img)

            # todo: update to time remaining.
            if ((i + 1) % self.args.print_freq == 0) or (i + 1 == len(test_loader)):
                logger.info(
                    "Test: [{}/{}] "
                    "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                        i + 1, len(test_loader), data_time=data_time, batch_time=batch_time
                    )
                )

    def scale_process_cuda(self, image: np.ndarray, raw_h: int, raw_w: int, stride_rate: float = 2 / 3) -> np.ndarray:
        """First, pad the image. If input is (384x512), then we must pad it up to shape
        to have shorter side "scaled base_size".

        Then we perform the sliding window on this scaled image, and then interpolate
        (downsample or upsample) the prediction back to the original one.

        At each pixel, we increment a counter for the number of times this pixel
        has passed through the sliding window.

        Args:
            image: Array, representing image where shortest edge is adjusted to base_size
            raw_h: integer representing native/raw image height on disk, e.g. for NYU it is 480
            raw_w: integer representing native/raw image width on disk, e.g. for NYU it is 640
            stride_rate: stride rate of sliding window operation

        Returns:
            prediction: Numpy array representing predictions with shorter side equal to self.base_size
        """
        resized_h, resized_w, _ = image.shape
        padded_image, pad_h_half, pad_w_half = pad_to_crop_sz(image, self.crop_h, self.crop_w, self.mean)
        new_h, new_w, _ = padded_image.shape
        stride_h = int(np.ceil(self.crop_h * stride_rate))
        stride_w = int(np.ceil(self.crop_w * stride_rate))
        grid_h = int(np.ceil(float(new_h - self.crop_h) / stride_h) + 1)
        grid_w = int(np.ceil(float(new_w - self.crop_w) / stride_w) + 1)

        prediction_crop = torch.zeros((self.num_eval_classes, new_h, new_w))
        count_crop = torch.zeros((new_h, new_w))

        prediction_crop = prediction_crop.to(self.device)
        count_crop = count_crop.to(self.device)

        # loop w/ sliding window, obtain start/end indices
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                # height indices are s_h to e_h (start h index to end h index)
                # width indices are s_w to e_w (start w index to end w index)
                s_h = index_h * stride_h
                e_h = min(s_h + self.crop_h, new_h)
                s_h = e_h - self.crop_h
                s_w = index_w * stride_w
                e_w = min(s_w + self.crop_w, new_w)
                s_w = e_w - self.crop_w
                image_crop = padded_image[s_h:e_h, s_w:e_w].copy()
                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[:, s_h:e_h, s_w:e_w] += self.net_process(image_crop)

        prediction_crop /= count_crop.unsqueeze(0)
        # disregard predictions from padded portion of image
        prediction_crop = prediction_crop[:, pad_h_half : pad_h_half + resized_h, pad_w_half : pad_w_half + resized_w]

        # CHW -> HWC
        prediction_crop = prediction_crop.permute(1, 2, 0)
        prediction_crop = prediction_crop.data.cpu().numpy()

        # upsample or shrink predictions back down to scale=1.0
        prediction = cv2.resize(prediction_crop, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
        return prediction

    def net_process(self, image: np.ndarray, flip: bool = True) -> torch.Tensor:
        """Feed input through the network.

        In addition to running a crop through the network, we can flip
        the crop horizontally, run both crops through the network, and then
        average them appropriately. Afterwards, apply softmax, then convert
        the prediction to the label taxonomy.

        Args:
            image:
            flip: boolean, whether to average with flipped patch output

        Returns:
            probs: Pytorch tensor representing network predicting in evaluation taxonomy
                (not necessarily the model taxonomy)
        """
        input = torch.from_numpy(image.transpose((2, 0, 1))).float()
        normalize_img(input, self.mean, self.std)
        input = input.unsqueeze(0)

        input = input.to(self.device)
        if flip:
            # add another example to batch dimension, that is the flipped crop
            input = torch.cat([input, input.flip(3)], 0)
        with torch.no_grad():
            logits, _, _, _ = self.model(input)
        _, _, h_i, w_i = input.shape
        _, _, h_o, w_o = logits.shape
        if (h_o != h_i) or (w_o != w_i):
            logits = F.interpolate(logits, (h_i, w_i), mode="bilinear", align_corners=True)

        # model & eval tax match, so no conversion needed
        assert self.model_taxonomy in ["universal", "test_dataset"]
        # todo: determine when .cuda() needed here
        probs = self.softmax(logits)

        if flip:
            # take back out the flipped crop, correct its orientation, and average result
            probs = (probs[0] + probs[1].flip(2)) / 2
        else:
            probs = probs[0]
        # output = output.data.cpu().numpy()
        # convert CHW to HWC order
        # output = output.transpose(1, 2, 0)
        # output = output.permute(1,2,0)

        return probs



def test_model(args):
    """ """
    args.split = "test"

    # dataset_name = args.dataset

    # if len(args.scales) > 1:
    #     scale_type = 'ms' # multi-scale
    # else:
    #     scale_type = 'ss' # single-scale

    # model_results_root = f'{Path(args.model_path).parent}/{Path(args.model_path).stem}'
    # args.save_folder = f'{model_results_root}/{args.dataset}/{args.base_size}/{scale_type}/'

    args.save_folder = f'{Path(args.model_path).stem}/{args.dataset}/{args.base_size}/'

    class_names = load_class_names(args.dataset)
    args.num_model_classes = len(class_names)
    num_eval_classes = args.num_model_classes

    # logger.info(args)

    # args.print_freq = 100

    itask = InferenceTask(
        args=args,
        base_size=args.base_size,
        crop_h=args.test_h,
        crop_w=args.test_w,
        input_file=None,
        model_taxonomy="test_dataset",
        eval_taxonomy="test_dataset",
        scales=args.scales,
        device=args.device
    )
    itask.execute()

    logger.info(">>>>>>>>> Calculating accuracy from cached results >>>>>>>>>>")

    excluded_ids = [] # no classes are excluded from evaluation of the test sets
    _, test_data_list = create_test_loader(args)
    ac = AccuracyCalculator(
        args=args,
        data_list=test_data_list,
        dataset_name=args.dataset,
        class_names=class_names,
        save_folder=args.save_folder,
        eval_taxonomy="test_dataset",
        num_eval_classes=num_eval_classes,
        excluded_ids=excluded_ids
    )
    ac.compute_metrics()


if __name__ == '__main__':
    args = DEFAULT_ARGS
    args.model_path = "/Users/johnlambert/Downloads/s21_cs6476_proj6/proj6_code/exp/camvid/pspnet50/model/train_epoch_100.pth"

    test_model(args)



# # if __name__ == '__main__':
# # 	pass
