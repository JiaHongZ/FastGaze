import os
import subprocess

# 定义模型文件前缀和日志保存目录
model_prefix = r"F:\project\FastGaze_master\train_10-10-2024-23-17-49\gazeformer_3E_3D_32_512d_"
log_dir = r"F:\project\FastGaze_master\train_10-10-2024-23-17-49"
log_file = os.path.join(log_dir, "combined_test_log.txt")
dataset_dir = r"G:\scanpath_data\Gazeformer-CVPR-2023\dataset"
img_ftrs_dir = r"G:\scanpath_data\Gazeformer-CVPR-2023\dataset\image_features"
script = "python test_clip.py"

# 创建或清空日志文件
with open(log_file, "w") as log:
    log.write("Test log initiated.\n")

# 循环加载 10.pkg 到 500.pkg，步长为 10
for i in range(10, 201, 10):
    # 当前的模型文件路径
    model_file = f"{model_prefix}{i}.pkg"

    # 输出当前测试的模型信息到日志
    with open(log_file, "a") as log:
        log.write(f"Testing with {model_file}...\n")
        print(f"Testing with {model_file}...")

    # 构建命令行参数
    command = [
        "python", "test_clip.py",
        "--trained_model", model_file,
        "--max_len", "7",
        "--sc_mask", "True",
        "--sc_ior", "True",
        "--num_encoder", "3",
        "--num_decoder", "3",
        "--lm_hidden_dim", "512",
        "--img_hidden_dim", "2048",
        "--dataset_dir", dataset_dir,
        "--img_ftrs_dir", img_ftrs_dir,
        "--cuda", "0"
    ]

    # 执行测试命令，并将输出追加到日志文件中
    with open(log_file, "a") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode == 0:
                log.write(f"Finished testing with {model_file}\n")
            else:
                log.write(f"Error occurred while testing {model_file}. Exit code: {result.returncode}\n")
        except Exception as e:
            log.write(f"Failed to run command for {model_file}. Error: {str(e)}\n")

    with open(log_file, "a") as log:
        log.write("---------------------------------------\n")

print(f"All tests completed. Logs are saved in {log_file}.")
