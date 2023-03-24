from .src.infer_dataset import InferDataset

def select_dataset(dataset_name, image_size=512):
    if dataset_name == 'casia1':
        dataset_rootpath = '/data/sunzhihao/season2206/dataset/CASIA1.0'

        labelfile_path = '/data/sunzhihao/season23/workspace/FindManipulation/datasets/labelfiles/casia1.txt'
        
        infer_dataset = InferDataset(dataset_rootpath=dataset_rootpath, labelfile_paths=[labelfile_path], image_size=image_size)
        
        return infer_dataset
    
    else:
        print(f'[Error] Dataset {dataset_name} is not supported!')
        exit()