from torch.utils.data.dataset import Dataset
import random
class BaseDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class MAMLDataset(Dataset):
    def __init__(self, label_dict,n_way,k_shot,n_query):
        super().__init__()
        self.label_dict=label_dict
        self.n_way=n_way
        self.k_shot=k_shot
        self.n_query=n_query
        self.n_sample=0
        for key,value in self.label_dict.items():
            self.n_sample+=value.shape[0]


    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.get_one_task()

    def get_one_task(self):
        choice_label=random.sample(list(self.label_dict.keys()),self.n_way)

        data_set=[]
        for label in choice_label:
            data_set.append((label,random.sample(list(self.label_dict[label]),self.k_shot+self.n_query)))

        support_set=[]
        query_set=[]
        for label,data in data_set:
            support_set.append((label,data[:self.k_shot]))
            query_set.append((label,data[self.k_shot:]))
        random.shuffle(support_set)
        random.shuffle(query_set)
        support_images=[]
        support_labels=[]
        query_images=[]
        query_labels=[]
        for label,images in support_set:
            support_images.extend(images)
            support_labels.extend([label]*len(images))
        for label,images in query_set:
            query_images.extend(images)
            query_labels.extend([label]*len(images))
        return support_images,support_labels,query_images,query_labels

