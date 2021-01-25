import logging
from .MNISTDataset import*
from .CIFAR10Dataset import *
from .IVCFilter import *
from .PneumoniaDataset import *


from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

mnist_tsfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

cifar10_tsfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def build_ae_dataset(dataset_name, data_path, ae_batch_size, normal_class):
    logger = logging.getLogger()
    logger.info("Build AutoEncoder's dataset for {}".format(dataset_name))
    
    assert dataset_name in ['mnist', 'cifar10', 'ivc-filter', 'rsna']
    
    if dataset_name == "mnist":
#         normal_class = [0]
        normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test = get_mnist_data(normal_class)
        
        train_set = MNISTLabelDataset(normal_x_train, normal_y_train, mnist_tsfms)
        validate_set = MNISTLabelDataset(normal_x_val, normal_y_val, mnist_tsfms)
        test_set = MNISTLabelDataset(normal_x_val+outlier_x_test, normal_y_val+outlier_y_test, mnist_tsfms)
        
        ae_dataloaders = {'train': DataLoader(train_set, batch_size = ae_batch_size, shuffle = True, num_workers = 1),
                       'val': DataLoader(validate_set, batch_size = ae_batch_size, num_workers = 1),
                       'test': DataLoader(test_set, batch_size = ae_batch_size, num_workers = 1)}
        ae_dataset_sizes = {'train': len(train_set), 'val': len(validate_set), 'test': len(test_set)}

    elif dataset_name == "cifar10":
#         normal_class = [0]
        normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test = get_cifar10_data(normal_class)

        train_set = CIFAR10LabelDataset(normal_x_train, normal_y_train, cifar10_tsfms)
        validate_set = CIFAR10LabelDataset(normal_x_val, normal_y_val, cifar10_tsfms)
        test_set = CIFAR10LabelDataset(normal_x_val+outlier_x_test, normal_y_val+outlier_y_test, cifar10_tsfms)
               
        ae_dataloaders = {'train': DataLoader(train_set, batch_size = ae_batch_size, shuffle = True, num_workers = 1),
                       'val': DataLoader(validate_set, batch_size = ae_batch_size, num_workers = 1),
                       'test': DataLoader(test_set, batch_size = ae_batch_size, num_workers = 1)}
        ae_dataset_sizes = {'train': len(train_set), 'val': len(validate_set), 'test': len(test_set)}
        
        
    elif dataset_name == "ivc-filter":
        trn_img, trn_lbl, tst_img, tst_lbl = get_ivc_anomaly_dataset(normal_class)        
        train_set = IVCFilter_Dataset(trn_img[0:int(0.8*len(trn_img))], trn_lbl[0:int(0.8*len(trn_img))])
        validate_set = IVCFilter_Dataset(trn_img[int(0.8*len(trn_img))+1:], trn_lbl[int(0.8*len(trn_img))+1:])
        test_set = IVCFilter_Dataset(trn_img[int(0.8*len(trn_img))+1:]+tst_img, trn_lbl[int(0.8*len(trn_img))+1:]+tst_lbl)
        
        ae_dataloaders = {'train': DataLoader(train_set, batch_size = ae_batch_size, shuffle = True, num_workers = 1),
                          'val': DataLoader(validate_set, batch_size = ae_batch_size, num_workers = 1),
                          'test': DataLoader(test_set, batch_size = ae_batch_size, num_workers = 1)}
        ae_dataset_sizes = {'train': len(train_set), 'val': len(validate_set), 'test':len(test_set)}
        
    elif dataset_name == "rsna":
        #normal: 0, opacity: 1, other: 2
        normal_path, opacity_path, other_path = get_rsna_data()
       
        all_path = [normal_path, opacity_path, other_path]
        
        new_normal_path = []
        for i in normal_class:
            new_normal_path = new_normal_path + all_path[i]
            
        new_abnormal_path = []    
        for i in range(0, len(all_path)):
            if i not in normal_class:
                new_abnormal_path = new_abnormal_path + all_path[i]
        
        traindata = new_normal_path[0:int(0.8*(len(new_normal_path)))]
        valdata = new_normal_path[int(0.8*(len(new_normal_path)))+1:]
        
        train_set = PneumoniaDataset(traindata, [0]*len(traindata))
        validate_set = PneumoniaDataset(valdata, [0]*len(valdata))
        test_set = PneumoniaDataset(valdata+new_abnormal_path, [0]*len(valdata)+[1]*len(new_abnormal_path))
        
        ae_dataloaders = {'train': DataLoader(train_set, batch_size = ae_batch_size, shuffle = True, num_workers = 1),
                          'val': DataLoader(validate_set, batch_size = ae_batch_size, num_workers = 1),
                          'test': DataLoader(test_set, batch_size = ae_batch_size, num_workers = 1)}
        ae_dataset_sizes = {'train': len(train_set), 'val': len(validate_set), 'test':len(test_set)}
        
    return ae_dataloaders, ae_dataset_sizes



def build_tend_dateset(dataset_name, data_path, tend_batch_size, normal_class):
    logger = logging.getLogger()
    logger.info("Build TEND's dataset for {}".format(dataset_name))
    
    assert dataset_name in ['mnist', 'cifar10', 'ivc-filter', 'rsna']
    
    if dataset_name == "mnist":
#         normal_class = [0]
        normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test = get_mnist_data(normal_class)
        cls_train_set = MNISTtfmDataset(normal_x_train, normal_y_train, mnist_tsfms)
        cls_validate_set = MNISTLabelDataset(normal_x_val+outlier_x_test, normal_y_val+outlier_y_test, mnist_tsfms)
        cls_dataloaders = {'train': DataLoader(cls_train_set, batch_size = tend_batch_size, shuffle = True, num_workers = 1),
                           'val': DataLoader(cls_validate_set, batch_size = tend_batch_size, num_workers = 1)}
        cls_dataset_sizes = {'train': len(cls_train_set), 'val': len(cls_validate_set)}
        
    elif dataset_name == "cifar10":
#         normal_class = [0]
        normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test = get_cifar10_data(normal_class)
        cls_train_set = CIFAR10TfmDataset(normal_x_train, normal_y_train, cifar10_tsfms)
        cls_validate_set = CIFAR10LabelDataset(normal_x_val+outlier_x_test, normal_y_val+outlier_y_test, cifar10_tsfms)
        cls_dataloaders = {'train': DataLoader(cls_train_set, batch_size = tend_batch_size, shuffle = True, num_workers = 1),
                           'val': DataLoader(cls_validate_set, batch_size = tend_batch_size, num_workers = 1)}
        cls_dataset_sizes = {'train': len(cls_train_set), 'val': len(cls_validate_set)}
        
    elif dataset_name == "ivc-filter":
        trn_img, trn_lbl, tst_img, tst_lbl = get_ivc_anomaly_dataset(normal_class) 
        cls_train_set = IVCFilterTfm_Dataset(trn_img[0:int(0.8*len(trn_img))], trn_lbl[0:int(0.8*len(trn_img))])
        cls_validate_set = IVCFilter_Dataset(trn_img[int(0.8*len(trn_img))+1:]+tst_img, trn_lbl[int(0.8*len(trn_img))+1:]+tst_lbl)
        
        cls_dataloaders = {'train': DataLoader(cls_train_set, batch_size = tend_batch_size, shuffle = True, num_workers = 1),
                           'val': DataLoader(cls_validate_set, batch_size = tend_batch_size, num_workers = 1)}
        cls_dataset_sizes = {'train': len(cls_train_set), 'val': len(cls_validate_set)}
    
    elif dataset_name == "rsna":
        normal_path, opacity_path, other_path = get_rsna_data()
       
        all_path = [normal_path, opacity_path, other_path]
        
        new_normal_path = []
        for i in normal_class:
            new_normal_path = new_normal_path + all_path[i]
            
        new_abnormal_path = []    
        for i in range(0, len(all_path)):
            if i not in normal_class:
                new_abnormal_path = new_abnormal_path + all_path[i]
        
        
        traindata = new_normal_path[0:int(0.8*(len(new_normal_path)))]
        valdata = new_normal_path[int(0.8*(len(new_normal_path)))+1:]

        cls_train_set = PneumoniaTfmDataset(traindata, [0]*len(traindata))
        cls_validate_set = PneumoniaDataset(valdata+new_abnormal_path, [0]*len(valdata)+[1]*len(new_abnormal_path))
        
        cls_dataloaders = {'train': DataLoader(cls_train_set, batch_size = tend_batch_size, shuffle = True, num_workers = 1),
                           'val': DataLoader(cls_validate_set, batch_size = tend_batch_size, num_workers = 1)}
        cls_dataset_sizes = {'train': len(cls_train_set), 'val': len(cls_validate_set)}

    return cls_dataloaders, cls_dataset_sizes



