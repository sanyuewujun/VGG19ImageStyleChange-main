import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from modeling import VGG

def image_loader(image_name, imsize):
    """图像预处理函数"""
    try:
        loader = transforms.Compose([
            transforms.Resize(imsize),  # 调整图像大小
            transforms.ToTensor()       # 转换为张量
        ])
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)  # 添加批次维度
        return image.to(device, torch.float)
    except FileNotFoundError:
        print(f"Error: The file {image_name} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def imshow(tensor, title=None):
    """图像显示函数"""
    try:
        unloader = transforms.ToPILImage()  # 转换回PIL图像
        image = tensor.cpu().clone()        # 克隆张量以避免修改原始图像
        image = image.squeeze(0)            # 移除批次维度
        image = unloader(image)
        image.show()
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # 暂停以更新图表
    except Exception as e:
        print(f"An error occurred while displaying the image: {e}")

def gram_matrix(input):
    """计算风格损失函数"""
    a, b, c, d = input.size()  # a=batch size(=1)
    features = input.view(a * b, c * d)  # 重塑为2D
    G = torch.mm(features, features.t())  # 计算Gram矩阵
    return G.div(a * b * c * d)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载主体图像和迁移风格图像
content_img = image_loader("VGG19ImageStyleChange-main\hoshinoai.jpg", 512)
style_img = image_loader("VGG19ImageStyleChange-main\picasso.jpg", 512)

# 如果图像加载失败，退出程序
if content_img is None or style_img is None:
    print("Exiting due to image loading errors.")
    exit()

# 初始化生成图像
input_img = content_img.clone()

# 定义模型
model = VGG().to(device).eval()

# 定义优化器
optimizer = optim.LBFGS([input_img.requires_grad_()])

# 定义内容损失和风格损失的权重
content_weight = 1
style_weight = 1000000

def get_content_loss(gen_features, content_features):
    """定义内容损失函数"""
    return F.mse_loss(gen_features[2], content_features[2])

def get_style_loss(gen_features, style_features):
    """定义风格损失函数"""
    loss = 0
    for gen, style in zip(gen_features, style_features):
        loss += F.mse_loss(gram_matrix(gen), gram_matrix(style))
    return loss

# 优化循环
num_steps = 1
for step in range(num_steps):
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        gen_features = model(input_img)
        content_features = model(content_img)
        style_features = model(style_img)
        content_loss = get_content_loss(gen_features, content_features)
        style_loss = get_style_loss(gen_features, style_features)
        loss = content_weight * content_loss + style_weight * style_loss
        loss.backward()
        if step % 50 == 0:
            print(f"Step [{step}/{num_steps}], Content Loss: {content_loss}, Style Loss: {style_loss}")
        return loss.item()
    optimizer.step(closure)

# 显示结果
output = input_img.data.clamp_(0, 1)
imshow(output, title='Output Image')
