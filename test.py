import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from train import *

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    maxdisp = 32
    feature_extractor =feature_extraction()
    model = MYmodel(maxdisp,feature_extractor)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


# 加载模型
model = load_checkpoint("C:\\Users\\kk\\Desktop\\data\\model\\shape\\model_weights_self256_shape.pth")
if torch.cuda.is_available():
    model = model.cuda()


def preserve_aspect_ratio_resize(image, target_size):
    # Compute the aspect ratio of the image and target size
    img_aspect = image.width / image.height
    target_aspect = target_size[0] / target_size[1]

    # Compare aspect ratios
    if img_aspect > target_aspect:
        # Image is wider than target, resize by width
        resize = transforms.Resize((int(target_size[1] * img_aspect), target_size[0]))
    else:
        # Image is taller or equal to target, resize by height
        resize = transforms.Resize((target_size[1], int(target_size[0] / img_aspect)))
    return resize(image)

transform = transforms.Compose([
    transforms.Lambda(lambda img: preserve_aspect_ratio_resize(img, target_size=(224, 224))),
    transforms.CenterCrop((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.500, 0.495, 0.501], std=[0.295, 0.294, 0.297]),
])

def predict_depth(image_path1, image_path2):
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    image1 = transform(image1).unsqueeze(0)
    image2 = transform(image2).unsqueeze(0)
    
    if torch.cuda.is_available():
        image1 = image1.cuda()
        image2 = image2.cuda()
    
    with torch.no_grad():
        predicted_depth = model(image1, image2).item()
    return predicted_depth

def test_and_show(image_path1, image_path2):
    ture_depth= input("please Enter the ture depth: ")
    predicted_depth = predict_depth(image_path1, image_path2)
    try:
        ture_depth=float(ture_depth)
        if ture_depth != 0:
            loss = (abs(predicted_depth-ture_depth))/ture_depth
            loss_str = f"{loss:.4f}"      
        else:
            accuracy_str = "***"
    except ValueError:
        accuracy_str ="***"

    plt.figure(figsize=(10, 5))
    Left_image = Image.open(image_path1)
    #Left_canny_image = cv2.Canny(cv2.imread(image_path1,cv2.IMREAD_GRAYSCALE),50,100)
    Right_image = Image.open(image_path2)
   # Right_canny_image = cv2.Canny(cv2.imread(image_path2,cv2.IMREAD_GRAYSCALE),50,100)
    
    plt.subplot(1, 2, 1)
    plt.imshow(Left_image)
    plt.title("Left_image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(Right_image)
    plt.title("Right_image")

    ###plt.subplot(2, 2, 3)
    ###plt.imshow(Left_canny_image)
    ###plt.title("Left_canny_image")

    ###plt.subplot(2, 2, 4)
    ###plt.imshow(Right_canny_image)
    ###plt.title("Right_canny_image")

    plt.suptitle(f"Predicted Depth: {predicted_depth:.4f} Real Depth :{ture_depth: .4f} \n Loss: {loss_str}", fontsize=16)
    plt.subplots_adjust(hspace=1)
    plt.show()

# 测试和显示结果
test_image1 = "path_to_test_image1.jpg"
test_image2 = "path_to_test_image2.jpg"
test_and_show('C:\\Users\\kk\\Desktop\\data\\image_shape\\test_data\\L\\manhole_13_9_20230627_160443.jpg', 'C:\\Users\\kk\\Desktop\\data\\image_shape\\test_data\\R\\manhole_13_9_20230627_160443.jpg')
 