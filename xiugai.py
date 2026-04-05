from ultralytics import YOLO
import cv2
import numpy as np

# -------------------------- 1. 加载训练好的模型 --------------------------
# 替换为你的best.pt文件路径（确保模型包含Continuous welding类别）
model = YOLO(r"C:\\Users\\86156\\Desktop\\程序汇总\\YOLOv8-model-improvement-P2cecasun\\best.pt")  

# -------------------------- 2. 加载待检测的图片 --------------------------
# 替换为你的测试图片路径（支持jpg/png等格式）
img_path = "00014.jpg"  
image = cv2.imread(img_path)
if image is None:
    raise ValueError(f"无法加载图片，请检查路径是否正确：{img_path}")

# -------------------------- 3. 执行检测 --------------------------
# conf：置信度阈值（可根据需要调整，比如0.5）
results = model(image, conf=0.25)  

# -------------------------- 4. 绘制检测框并显示两位小数置信度 --------------------------
# 定义类别名称映射（如果模型输出的是英文/数字，可在这里统一调整显示）
# 示例：如果模型中连焊的类别ID是1，可手动映射；若模型已输出Continuous welding则无需修改
class_name_mapping = {
    "Continuous welding": "Continuous welding",  # 连焊（保持英文显示）
    # 可添加其他类别映射，比如：
    # "Missing welding": "Missing welding",  # 漏焊
    # "多焊": "Excessive welding"  # 中文转英文（如果需要）
}

# 获取检测结果的基本信息
result = results[0]
for box in result.boxes:
    # 获取边界框坐标（四舍五入为整数）
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # 获取置信度并保留两位小数（核心：确保两位小数显示）
    conf = float(box.conf[0])
    conf_formatted = f"{conf:.2f}"  # 强制两位小数，如0.9→0.90
    
    # 获取类别名称并映射（确保Continuous welding正确显示）
    cls_id = int(box.cls[0])
    original_cls_name = result.names[cls_id] if result.names else f"Class_{cls_id}"
    display_cls_name = class_name_mapping.get(original_cls_name, original_cls_name)
    
    # 绘制检测框（红色，线宽2，突出Continuous welding类别）
    if display_cls_name == "Continuous welding":
        box_color = (0, 0, 255)  # 连焊用红色框
    else:
        box_color = (0, 255, 0)  # 其他类别用绿色框
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
    
    # 绘制标签背景（黑色半透明）
    label = f"{display_cls_name}: {conf_formatted}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    label_y1 = y1 - 10 if y1 - 10 > 10 else y1 + 20  # 避免标签超出图片边界
    cv2.rectangle(image, (x1, label_y1 - label_size[1] - 5), 
                  (x1 + label_size[0] + 5, label_y1 + 5), (0, 0, 0), -1)
    
    # 绘制标签文字（白色）
    cv2.putText(image, label, (x1 + 2, label_y1), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# -------------------------- 5. 显示和保存结果 --------------------------
# 调整显示窗口大小（可选，适配不同分辨率图片）
display_img = cv2.resize(image, (1000, 800))  # 放大窗口，方便查看小目标
cv2.imshow("PCB焊点缺陷检测结果（Continuous welding重点标注）", display_img)

# 保存检测后的图片（建议保存，用于论文案例分析）
save_path = "detected_Continuous_welding_result.jpg"
cv2.imwrite(save_path, image)
print(f"检测结果已保存至：{save_path}")

# 等待按键关闭窗口（按任意键退出）
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------- 6. 控制台打印详细检测结果 --------------------------
print("\n===== 检测结果详情 =====")
print(f"检测到的目标总数：{len(result.boxes)}")
continuous_welding_count = 0  # 统计连焊数量
for box in result.boxes:
    conf = float(box.conf[0])
    cls_name = result.names[int(box.cls[0])] if result.names else f"Class_{int(box.cls[0])}"
    display_cls_name = class_name_mapping.get(cls_name, cls_name)
    
    # 统计连焊数量
    if display_cls_name == "Continuous welding":
        continuous_welding_count += 1
    
    print(f"类别：{display_cls_name}，置信度：{conf:.2f}")

print(f"\n检测到的连焊（Continuous welding）数量：{continuous_welding_count}")