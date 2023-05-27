from PIL import Image  
import os
import os.path
from torchvision import transforms
import json
def save_format(image_path,augument_from,cls):
    return {
        "image_path":image_path,
        "augument_from_path":augument_from,
        "class":cls
    }
def get_factor(annotations, cls, target_samples):
    class_count = sum(1 for annot in annotations if annot['class'] == cls)
    factor = max(target_samples // class_count - 1, 0)
    return factor

def get_rare(annotations, threshold=0.2):
    # Count the number of samples per class
    class_counts = {}
    for annot in annotations:
        cls = annot['class']
        if cls not in class_counts:
            class_counts[cls] = 0
        class_counts[cls] += 1
     
    # Calculate the proportions of each class
    total_samples = len(annotations)
    class_proportions = {cls: count / total_samples for cls, count in class_counts.items()}
     
    # Find the rare classes whose proportions are less than the threshold
    rare_classes = [cls for cls, proportion in class_proportions.items() if proportion < threshold]
    
    print("Rare Class condition in train dataset:")
    for cls in rare_classes:
        print(f"class {cls} with proportion {class_proportions[cls]} samples number: {class_counts[cls]}")
    return rare_classes

def generate_data_augument(data_path="",threshold=0.2,sample_number=500):
    annotations = json.load(open(os.path.join(data_path, 
                                                       'annotations', "train.json")))
    rare_classes=get_rare(annotations=annotations,threshold=threshold)

    # Create transform method to augument images
    augument_transform=transforms.Compose([
            transforms.RandomRotation(60),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
    # Identify the indices of the minority class samples and
    #   calculate their augmentation factors
    augument_annotation=[]
    for cls in rare_classes:
        indices = [i for i, annot in enumerate(annotations) if annot['class'] == cls]
        
        # DEFINE[factor]: for each image in rare class, we will generate {factor}
        #   augumented images and stored them in data_path/auugument
        factor = get_factor(annotations, cls, sample_number)
        print(f"Augument factor for class {cls} is {factor}")
        for rare_index in indices:
            for _ in range(factor):
                image_path=annotations[rare_index]['image_path']
                img=Image.open(image_path)
                img_transformed=augument_transform(img)
                new_path=os.path.join(data_path,'images,'f"aug_{os.path.basename(image_path)}")
                img_transformed.save(new_path)
                augument_annotation.append(save_format(
                    image_path=new_path,
                    augument_from=image_path,
                    cls=annotations[rare_index]['class']
                ))
    with open(os.path.join(data_path,"annotations","augument.json"),'w') as f:
        json.dump(augument_annotation,f)
            

