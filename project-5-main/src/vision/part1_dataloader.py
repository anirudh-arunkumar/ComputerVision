import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class Argoverse(Dataset):

    def load_path_with_classes(self, split: str, data_root: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Builds (path, class) pairs by pulling all txt files found under
        the data_root directory under the given split. Also builds a list
        of all labels we have provided data for.

        Each of the classes have a total of 200 point clouds numbered from 0 to 199.
        We will be using point clouds 0-169 for the train split and point clouds 
        170-199 for the test split. This gives us a 85/15 train/test split.

        Args:
        -   split: Either train or test. Collects (path, label) pairs for the specified split
        -   data_root: Root directory for training and testing data
        
        Output:
        -   pairs: List of all (path, class) pairs found under data_root for the given split 
        -   class_list: List of all classes present in the dataset *sorted in alphabetical order*
        """

        pairs = []
        class_list = []

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`load_path_with_classes` function in "
        #     + "`part1_dataloader.py` needs to be implemented"
        # )

        for c_name in sorted(os.listdir(data_root)):
            c_directory = os.path.join(data_root, c_name)

            if not os.path.isdir(c_directory):
                continue
            class_list.append(c_name)

            for f in os.listdir(c_directory):

                if not f.endswith(".txt"):
                    continue
                
                index = int(os.path.splitext(f)[0])
                in_train = index <= 169

                if (split == "train" and in_train) or (split == "test" and not in_train):
                    f_path = os.path.join(c_directory, f)
                    pairs.append((f_path, c_name))
        

        ############################################################################
        # Student code end
        ############################################################################

        return pairs, class_list


    def get_class_dict(self, class_list: List[str]) -> Dict[str, int]:
        """
        Creates a mapping from classes to labels. For example, [Animal, Car, Bus],
        would map to {Animal:0, Bus:1, Car:2}. *Note: for consistency, we sort the
        input classes in alphabetical order before creating the mapping (gradescope)
        tests will probably fail if you forget to do so*

        Args:
        -   class_list: List of classes to create mapping from

        Output: 
        -   classes: dictionary containing the class to label mapping
        """

        classes = dict()

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`get_class_dict` function in "
        #     + "`part1_dataloader.py` needs to be implemented"
        # )

        for i, c, in enumerate(sorted(class_list)):
            classes[c] = i

        ############################################################################
        # Student code end
        ############################################################################

        return classes
    

    def __init__(self, split: str, data_root: str, pad_size: int) -> None:
        """
        Initializes the dataset. *Hint: Use the functions above*

        Args:
        -   split: Which split to pull data for. Either train or test
        -   data_root: The root of the directory containing all the data
        -   pad_size: The number of points each point cloud should contain when
                      when we access them. This is used in the pad_points function.

        Variables:
        -   self.instances: List of (path, class) pairs
        -   class_dict: Mapping from classes to labels
        -   pad_size: Number of points to pad each point cloud to
        """
        super().__init__()
        
        file_label_pairs, classes = self.load_path_with_classes(split, data_root)
        self.instances = file_label_pairs
        self.class_dict = self.get_class_dict(classes)
        self.pad_size = pad_size


    def get_points_from_file(self, path: str) -> torch.Tensor:
        """
        Returns a tensor containing all of the points in the given file

        Args:
        -   path: Path to the file that we should extract points from

        Output:
        -   pts: A tensor of shape (N, 3) where N is the number of points in the file
        """

        pts = []

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`get_points_from_file` function in "
        #     + "`part1_dataloader.py` needs to be implemented"
        # )

        with open(path, 'r') as file:
            _ = file.readline()
            for l in file:
                
                l = l.strip()

                if not l:
                    continue
                    
                a, b, c = l.split()
                
                pts.append([float(a), float(b), float(c)])
        pts = torch.tensor(pts, dtype=torch.float32)

        

        ############################################################################
        # Student code end
        ############################################################################

        return pts

    def pad_points(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Pads pts to have pad_size points in it. Let p1 be the first point in 
        the tensor. We want to pad pts by adding p1 to the end of pts until 
        it has size (pad_size, 3). 

        Args:
        -   pts: A tensor of shape (N, 3) where N is the number of points in the tensor

        Output: 
        -   pts_full: A tensor of shape (pad_size, 3)
        """

        pts_full = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`pad_points` function in "
        #     + "`part1_dataloader.py` needs to be implemented"
        # )

        if pts.shape[0] >= self.pad_size:
            return pts[:self.pad_size]
        
        n = pts.shape[0]
        first = pts[0:1]
        pad = self.pad_size - n
        pts_pad = first.repeat(pad, 1)

        return torch.cat([pts, pts_pad], dim = 0)

        ############################################################################
        # Student code end
        ############################################################################
        
        return pts_full

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the (points, label) pair at the given index.

        Hint: 
        1) get info from self.instances
        2) use get_points_from_file and pad_points

        Args:
        -   i: Index to retrieve

        Output:
        -   pts: Points contained in the file at the given index
        -   label: Tensor containing the label of the point cloud at the given index
        """

        pts = None
        label = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`__getitem__` function in "
        #     + "`part1_dataloader.py` needs to be implemented"
        # )
        pth, c = self.instances[i]

        pts = self.get_points_from_file(path=pth)
        pts = self.pad_points(pts=pts)
        label = torch.tensor(self.class_dict[c], dtype=torch.long)


        

        ############################################################################
        # Student code end
        ############################################################################

        return pts, label

    def __len__(self) -> int:
        """
        Returns number of examples in the dataset

        Output: 
        -    l: Length of the dataset
        """
        
        l = None

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError(
        #     "`__len__` function in "
        #     + "`part1_dataloader.py` needs to be implemented"
        # )
        l = len(self.instances)

        ############################################################################
        # Student code end
        ############################################################################

        return l
