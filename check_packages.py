import sys
print("Python版本:", sys.version)
print("Python安装路径:", sys.executable)
print("\nPython模块搜索路径:")
for p in sys.path:
    print("-", p)

print("\n检查必要的包:")

packages = [
    "pyroomacoustics", 
    "numpy", 
    "librosa", 
    "matplotlib",
    "tensorflow",
    "torch"
]

for package in packages:
    try:
        module = __import__(package)
        try:
            version = module.__version__
            location = module.__file__
            print(f"√ {package} 已安装 (版本: {version})")
            print(f"  位置: {location}")
        except AttributeError:
            print(f"√ {package} 已安装")
    except ImportError:
        print(f"× {package} 未安装") 