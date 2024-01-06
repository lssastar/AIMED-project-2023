from skimage import io, transform
from skimage.metrics import structural_similarity as compare_ssim
import os
import numpy as np

# 加载参考图片
reference_image_path = 'D:\\Data\\ADNI2T1\\ADNI_009_S_0751_MR_localizer__br_raw_20120906170433106_1_S166392_I331767.png'
reference_image = io.imread(reference_image_path, as_gray=True)

# 获取参考图像的尺寸
ref_shape = reference_image.shape
reference_image = transform.resize(reference_image, ref_shape)

# 设定 SSIM 阈值，高于此值的图片将被删除
ssim_threshold = 0.5  # 可根据需要调整

image_dir = 'D:\\Data\\ADNI2T1\\classes\\SMC'
os.chdir(image_dir)

# 检查目录中的每个文件
for filename in os.listdir('.'):
    if filename.endswith('.png') and filename != reference_image_path:
        current_image = io.imread(filename, as_gray=True)

        # 调整当前图像的尺寸以匹配参考图像
        current_image = transform.resize(current_image, ref_shape)
        
        # 计算当前图片与参考图片的 SSIM
        ssim_value = compare_ssim(reference_image, current_image, data_range=1.0)
        # print(f'Checking "{filename}"...')
        # print(f'SSIM: {ssim_value:.2f}')
        # 如果 SSIM 高于阈值，则删除文件
        if ssim_value > ssim_threshold:
            os.remove(filename)
            print(f'Deleted "{filename}" due to high structural similarity.')

reference_image_path2 = 'D:\\Data\\ADNI2T1\\ADNI_941_S_5124_MR_localizer__br_raw_20130410084410495_1_S186457_I366198.png'
reference_image2 = io.imread(reference_image_path2, as_gray=True)

# 获取参考图像的尺寸
ref_shape = reference_image2.shape
reference_image2 = transform.resize(reference_image2, ref_shape)

# 检查目录中的每个文件
for filename in os.listdir('.'):
    if filename.endswith('.png') and filename != reference_image_path2:
        current_image = io.imread(filename, as_gray=True)
        
        # 调整当前图像的尺寸以匹配参考图像
        current_image = transform.resize(current_image, ref_shape)

        # 计算当前图片与参考图片的 SSIM
        ssim_value = compare_ssim(reference_image2, current_image, data_range=1.0)
        # print(f'Checking "{filename}"...')
        # print(f'SSIM: {ssim_value:.2f}')
        # 如果 SSIM 高于阈值，则删除文件
        if ssim_value > ssim_threshold:
            os.remove(filename)
            print(f'Deleted "{filename}" due to high structural similarity.')

reference_image_path2 = 'D:\\Data\\ADNI2T1\\ADNI_009_S_4337_MR_localizer__br_raw_20121114085911177_1_S174457_I346474.png'
reference_image2 = io.imread(reference_image_path2, as_gray=True)

# 获取参考图像的尺寸
ref_shape = reference_image2.shape
reference_image2 = transform.resize(reference_image2, ref_shape)

# 检查目录中的每个文件
for filename in os.listdir('.'):
    if filename.endswith('.png') and filename != reference_image_path2:
        current_image = io.imread(filename, as_gray=True)
        
        # 调整当前图像的尺寸以匹配参考图像
        current_image = transform.resize(current_image, ref_shape)

        # 计算当前图片与参考图片的 SSIM
        ssim_value = compare_ssim(reference_image2, current_image, data_range=1.0)
        # print(f'Checking "{filename}"...')
        # print(f'SSIM: {ssim_value:.2f}')
        # 如果 SSIM 高于阈值，则删除文件
        if ssim_value > ssim_threshold:
            os.remove(filename)
            print(f'Deleted "{filename}" due to high structural similarity.')