import os
import json
import numpy as np
import mindspore.dataset as ds

np.random.seed(4399)


class ShapeNetpartDataset:
    def __init__(self, root_path, split='train', num_points=1024, normal_channel=False, class_choice=None):
        self.path = root_path
        self.num_points = num_points
        self.normal_channel = normal_channel
        self.catfile = os.path.join(self.path, 'synsetoffset2category.txt')
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.path, 'train_test_split', 'shuffled_train_file_list.json'), 'r')as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.path, 'train_test_split', 'shuffled_val_file_list.json'), 'r')as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.path, 'train_test_split', 'shuffled_test_file_list.json'), 'r')as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.path, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Existing..' % split)
                exit(1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}
        self.cache_size = 20000

    def __getitem__(self, index):
        # print("**********************")
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = self._pc_normalize(point_set[:, 0:3])

        if len(seg) >= self.num_points:
            point_set = point_set[:self.num_points, :]
            seg = seg[:self.num_points]
        else:
            numbers = self.num_points - len(seg)
            zeros = np.array(point_set[0, :]).reshape(1, -1)
            zeros = zeros.repeat(numbers, axis=0)
            point_set = point_set[:self.num_points, :]
            seg_zeros = np.array(seg[0]).repeat(numbers, axis=0)
            seg = seg[:self.num_points]
            point_set = np.concatenate((point_set, zeros), axis=0)
            seg = np.concatenate((seg, seg_zeros), axis=0)

        # cls = np.expand_dims(cls, axis=1)
        # if not self.normal_channel:
        #     cls = cls.repeat(3, axis=1).astype(np.float32)
        # else:
        #     cls = cls.repeat(6, axis=1).astype(np.float32)
        # point_set = np.concatenate((point_set, cls), axis=0)

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)

    def _pc_normalize(self, data):
        centroid = np.mean(data, axis=0)
        data = data - centroid
        m = np.max(np.sqrt(np.sum(data ** 2, axis=1)))
        data /= m
        return data


if __name__ == "__main__":
    import open3d as o3d
    root_path = '/home/user/cmj/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    dataset_generator = ShapeNetpartDataset(root_path=root_path,split="train",normal_channel=True)

    data = dataset_generator[0]
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "classes", "label"], shuffle=False)
    dataset = dataset.batch(8, drop_remainder=True)
    for data in dataset.create_dict_iterator():
        pointcloud = data["data"].asnumpy()
        classes = data["classes"].asnumpy()
        # pc = pointcloud[0,:1024,:]
        # print(pc)
        # # pcd = o3d.geometry.PointCloud()
        # # pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
        # # o3d.visualization.draw_geometries([pcd])
        label = data["label"].asnumpy()
        print(pointcloud.shape)
        print(classes.shape)
        print(label.shape)
        # sys.exit()
    print('data size:', dataset.get_dataset_size())
