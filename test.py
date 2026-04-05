import os

def check_actual_structure():
    """检查实际的数据集结构"""
    
    base_path = r"C:\Users\86156\Desktop\PCB\DATA\data3"
    
    print("📁 检查实际的数据集结构...")
    print(f"基础路径: {base_path}")
    
    if not os.path.exists(base_path):
        print("❌ 基础路径不存在!")
        return
    
    print(f"\n📂 基础目录内容:")
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            print(f"   📁 {item}/")
            # 显示子目录内容
            try:
                sub_items = os.listdir(item_path)
                for sub_item in sub_items[:5]:  # 只显示前5个
                    sub_item_path = os.path.join(item_path, sub_item)
                    if os.path.isdir(sub_item_path):
                        print(f"      📁 {sub_item}/")
                    else:
                        print(f"      📄 {sub_item}")
                if len(sub_items) > 5:
                    print(f"      ... 还有 {len(sub_items) - 5} 个文件")
            except:
                print(f"      (无法访问)")
        else:
            print(f"   📄 {item}")
    
    # 搜索图像文件
    print(f"\n🔍 搜索图像文件...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                rel_path = os.path.relpath(os.path.join(root, file), base_path)
                image_files.append(rel_path)
    
    print(f"找到 {len(image_files)} 个图像文件:")
    for img in image_files[:10]:  # 显示前10个
        print(f"   📷 {img}")
    if len(image_files) > 10:
        print(f"   ... 还有 {len(image_files) - 10} 个图像文件")

if __name__ == '__main__':
    check_actual_structure()