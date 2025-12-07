# utils/data.py
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms
from avalanche.benchmarks import nc_benchmark
from torchvision import datasets as datasets
from avalanche.benchmarks.datasets import CUB200,TinyImagenet
from avalanche.benchmarks import SplitTinyImageNet
import torch
from torchvision.datasets.folder import default_loader



class SubsetImageFolder(torch.utils.data.Dataset):
    def __init__(self, paths, targets, transform):
        self.samples = list(zip(paths, targets))
        self.targets = list(targets)  # <-- Add this line
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)
    

    
def create_transforms(dataset_name):
    """
    Create transforms for the given dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (train_transform, test_transform)
    """
    if dataset_name == "cifar100":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name == "cub200":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif dataset_name == "fgvc_aircraft":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif dataset_name == "tinyimagenet":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif dataset_name == "imagenet-r":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif dataset_name == "imagenet-a":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_transform, test_transform

def create_benchmark(config):
    """
    Create an Avalanche continual learning benchmark from the configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Avalanche scenario
    """
    dataset_name = config['dataset_name']
    print(f"Creating benchmark for {dataset_name}")
    num_tasks = config['training']['num_tasks']
    data_path = config['data_path']
    seed = config['seed']
    print(f"Creating benchmark for {dataset_name} with {num_tasks} tasks")
    train_transform, test_transform = create_transforms(dataset_name)
    
    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_path, 
            train=True, 
            download=True, 
            transform=train_transform
        )
        
        test_dataset = datasets.CIFAR100(
            root=data_path, 
            train=False, 
            download=True, 
            transform=test_transform
        )
    elif dataset_name == "cub200":
        train_dataset = CUB200(
            data_path, 
            train=True, 
            download=True, 
            transform=train_transform
        )
        
        test_dataset = CUB200(
            data_path, 
            train=False, 
            download=True, 
            transform=test_transform
        )
        
        # Convert to list for Avalanche compatibility
        train_dataset.targets = [target for target in train_dataset.targets]
        test_dataset.targets = [target for target in test_dataset.targets]

    

    elif dataset_name == "fgvc_aircraft":
        train_dataset = datasets.FGVCAircraft(
            data_path, 
            split='train', 
            transform=train_transform, 
            download=True,
            annotation_level="variant"
        )
        
        test_dataset = datasets.FGVCAircraft(
            data_path, 
            split='test', 
            transform=test_transform, 
            download=True,
            annotation_level="variant"
        )
        
        # Convert labels for Avalanche compatibility
        train_dataset.targets = [cls for cls in train_dataset._labels]
        test_dataset.targets = [cls for cls in test_dataset._labels]
    

    elif dataset_name == "imagenet-r":
        from torchvision.datasets import ImageFolder
        full_dataset = torchvision.datasets.ImageFolder(data_path, transform=train_transform)

        # Split samples and create new datasets
        paths, targets = zip(*full_dataset.samples)
        train_paths, test_paths, train_targets, test_targets = train_test_split(
            paths, targets, test_size=0.3, stratify=targets, random_state=seed
        )

        from torchvision.datasets.folder import default_loader



        train_dataset = SubsetImageFolder(train_paths, train_targets, train_transform)
        test_dataset  = SubsetImageFolder(test_paths, test_targets, test_transform)
        

        scenario = nc_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=num_tasks,
            shuffle=True,
            seed=seed,
            task_labels=False
        )
        return scenario
    
    elif dataset_name == "imagenet-a":
        from torchvision.datasets import ImageFolder
        full_dataset = ImageFolder(data_path, transform=train_transform)

        # Split samples and create new datasets
        paths, targets = zip(*full_dataset.samples)
        train_paths, test_paths, train_targets, test_targets = train_test_split(
            paths, targets, test_size=0.3, stratify=targets, random_state=seed
        )

        train_dataset = SubsetImageFolder(train_paths, train_targets, train_transform)
        test_dataset  = SubsetImageFolder(test_paths, test_targets, test_transform)

        scenario = nc_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=num_tasks,
            shuffle=True,
            seed=seed,
            task_labels=False
        )
        return scenario


    elif dataset_name == "tinyimagenet":
       
    
        scenario = SplitTinyImageNet(
            n_experiences=num_tasks,
            dataset_root=data_path,
            seed=seed,
            return_task_id=False,
            train_transform=train_transform,
            eval_transform=test_transform,
            shuffle=True,
        )
        return scenario
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create Avalanche scenario
    scenario = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=num_tasks,
        shuffle=True,
        seed=seed,
        task_labels=False
    )
    
    return scenario