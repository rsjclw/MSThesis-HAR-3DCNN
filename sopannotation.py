from glob import glob
import numpy as np
dataset_path = 'sop_dataset/'

# class_file = dataset_path+'label_map_sop.txt'
# class_names = []
# f = open(class_file, 'r')
# for line in f:
#     class_names.append(line[:-1])
# f.close()
# annotation = []
# class_idx = 0
# for c in class_names:
#     for idx in glob(dataset_path+c+'/*'):
#         annotation.append([idx[12:], class_idx])
#     class_idx += 1
# annotation = np.array(annotation)
# np.random.shuffle(annotation)
# # 696
# # 232
# # 232
# print(annotation)
# print((annotation[:696]).shape, (annotation[696:928]).shape, (annotation[928:1160]).shape)
# np.savetxt(dataset_path+'sop_annotation.txt', annotation, delimiter=',', fmt='%s')

annotation = np.genfromtxt(dataset_path+'sop_annotation.txt', delimiter=',', dtype=str)
for i in annotation:
    print(dataset_path + i[0], int(i[1]))
print((annotation[:696]).shape, (annotation[696:928]).shape, (annotation[928:1160]).shape)

for i in range(8):
    print('class_'+str(i)+'_train:', len(np.where(annotation[:696,1] == str(i))[0]))
    print('class_'+str(i)+'_val:',len(np.where(annotation[696:928,1] == str(i))[0]))
    print('class_'+str(i)+'_test:',len(np.where(annotation[928:1160,1] == str(i))[0]))