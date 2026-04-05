import torch
import gc

def optimize_memory():
    """优化内存使用"""
    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 垃圾回收
    gc.collect()
    
    # 设置PyTorch内存优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    print("内存优化完成")

def check_memory():
    """检查内存状态"""
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU内存缓存: {torch.cuda.memory_cached()/1024**3:.2f} GB")
    else:
        print("使用CPU训练")

if __name__ == '__main__':
    optimize_memory()
    check_memory()