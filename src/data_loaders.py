import os

import numpy as np

def get_data_loaders(val_split, test_split, batch_size=32, verbose=True):
    num_workers = 0
    random_state = 10
    n_splits = 1

    # Listing the data
    # Cats
    print("LISTING DATA")
    input_dir = "dataset/temp/cats"
    images = [os.path.join(input_dir, image) for image in os.listdir(input_dir)]
    cat_images = np.array(images)  # transform to numpy
    cat_labels = ['cat'] * len(cat_images)

    # Dogs
    input_dir2 = "dataset/temp/dogs"
    images2 = [os.path.join(input_dir2, image) for image in os.listdir(input_dir2)]
    dog_images = np.array(images2)  # transform to numpy
    dog_labels = ['dog'] * len(dog_images)

    # Panda
    input_dir3 = "dataset/temp/panda"
    images3 = [os.path.join(input_dir3, image) for image in os.listdir(input_dir3)]
    panda_images = np.array(images3)  # transform to numpy
    panda_labels = ['panda'] * len(panda_images)

    # Appending lists
    images = np.append(np.append(cat_images, dog_images), panda_images)
    labels = cat_labels + dog_labels + panda_labels
    labels = np.array(labels)

    # Formatting the labs as ints
    classes = np.unique(labels).flatten()
    labels_int = np.zeros(labels.size, dtype=np.int64)

    # Convert string labels to integers
    for index, class_name in enumerate(classes):
        labels_int[labels == class_name] = index

    if verbose:
        print("Number of images in the dataset:", len(images))
        for index, class_name in enumerate(classes):
            print("Number of images in class ", class_name,
                  ":", (labels_int == index).sum())

    # Splitting the data in dev and test sets
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_split, random_state=random_state)
    sss.get_n_splits(images, labels_int)
    dev_index, test_index = next(sss.split(images, labels_int))

    dev_images = images[dev_index]
    dev_labels = labels_int[dev_index]

    test_images = images[test_index]
    test_labels = labels_int[test_index]

    # Splitting the data in train and val sets
    val_size = int(val_split * images.size)
    val_split = val_size / dev_images.size
    sss2 = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=val_split, random_state=random_state)
    sss2.get_n_splits(dev_images, dev_labels)
    train_index, val_index = next(sss2.split(dev_images, dev_labels))

    train_images = images[train_index]
    train_labels = labels_int[train_index]

    val_images = images[val_index]
    val_labels = labels_int[val_index]

    if verbose:
        print("Train set:", train_images.size)
        print("Val set:", val_images.size)
        print("Test set:", test_images.size)

    # Transforms
    torchvision_transform_train = transforms.Compose(
        [transforms.Resize((args.unified_image_width, args.unified_image_height)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.ToTensor()])

    # Datasets
    train_dataset_unorm = CustomDataset(
        train_images, train_labels, transform=torchvision_transform_train)

    # Get training set stats
    trainloader_unorm = torch.utils.data.DataLoader(
        train_dataset_unorm, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    mean_train, std_train = get_dataset_stats(trainloader_unorm)

    if verbose:
        print("Statistics of training set")
        print("Mean:", mean_train)
        print("Std:", std_train)

    torchvision_transform = transforms.Compose(
        [transforms.Resize((args.unified_image_width, args.unified_image_height)),
         transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean_train, std=std_train)])

    torchvision_transform_test = transforms.Compose(
        [transforms.Resize((args.unified_image_width, args.unified_image_height)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean_train, std=std_train)])

    # Get the train/val/test loaders
    train_dataset = CustomDataset(
        train_images, train_labels, transform=torchvision_transform)
    val_dataset = CustomDataset(val_images, val_labels, transform=torchvision_transform)
    test_dataset = CustomDataset(
        test_images, test_labels, transform=torchvision_transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader