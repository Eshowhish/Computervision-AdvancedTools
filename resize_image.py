#%%
import matplotlib.pyplot as plt
import pickle
import numpy as np
import PIL
import random




train_list = []
train_label_list = []

test_list = []
test_label_list = []

val_list = []
val_label_list = []

for i in range(12500):
    try:
        img_cat = PIL.Image.open("kagglecatsanddogs_3367a/PetImages/Cat/"+str(i)+'.jpg').convert('RGB')
        img_dog = PIL.Image.open("kagglecatsanddogs_3367a/PetImages/Dog/"+str(i)+'.jpg').convert('RGB')
        img_cat_1 = img_cat.copy().resize((128, 128))
        img_dog_1 = img_dog.copy().resize((128, 128))
    

        if i < 8000: 
            train_list.append(np.array(img_cat_1))
            train_label_list.append(1)
            img_cat_2 = img_cat.copy().resize((random.randint(128, 224), random.randint(128, 224)))
            train_list.append(np.array(img_cat_2))
            train_label_list.append(1)
            
            train_list.append(np.array(img_dog_1))
            train_label_list.append(0)
            img_dog_2 = img_dog.copy().resize((random.randint(128, 224), random.randint(128, 224)))
            train_list.append(np.array(img_dog_2))
            train_label_list.append(0)
        elif 8000 <= i < 10000:
            img_cat_2 = img_cat.copy().resize((random.randint(128, 224), random.randint(128, 224)))
            val_list.append(np.array(img_cat_2))
            val_label_list.append(1)
            img_dog_2 = img_dog.copy().resize((random.randint(128, 224), random.randint(128, 224)))
            val_list.append(np.array(img_dog_2))
            val_label_list.append(0)

        else:
            test_list.append(np.array(img_cat_1))
            test_label_list.append(1)
            test_list.append(np.array(img_dog_1))
            test_label_list.append(0)

    except:
        print('error !!' + str(i))


#%%
dic = {'train_data' : train_list, 'train_label' : train_label_list, 
        'test_data' : test_list, 'test_label':test_label_list,
        'val_data' : val_list, 'val_label' : val_label_list}


#%%
with open('cat_dog_dataset_val_2.plk', 'wb') as f:
    pickle.dump(dic, f)
# %%
