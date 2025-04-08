from PIL import Image

import torch.nn as nn
import torch

from typing import List
import traceback # 添加导入

from DeQAScore.src.model.builder import load_pretrained_model

from DeQAScore.src.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from DeQAScore.src.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


class Scorer(nn.Module):
    def __init__(self, pretrained="zhiyuanyou/DeQA-Score-Mix3", device="cuda:1"):
        super().__init__()
        # 确定目标设备，优先使用传入的 device，否则根据 CUDA 可用性选择 cuda 或 cpu
        target_device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[Scorer __init__] Initializing Scorer. Target device requested: {device}, determined as: {target_device}")

        try:
            # 加载模型，确保将设备作为字符串传递给 load_pretrained_model
            # 注意: load_pretrained_model 内部可能也会尝试移动模型，但我们之后会再次强制移动
            print(f"[Scorer __init__] Loading pretrained model '{pretrained}'...")
            tokenizer, model, image_processor, _ = load_pretrained_model(
                pretrained,
                None,           # model_base 通常为 None
                "deqa",         # model_name
                device=str(target_device) # 将 torch.device 转为字符串传递
            )
            # 检查模型加载后初始状态（可能在 CPU 或部分在 GPU）
            initial_device = next(model.parameters(), None)
            if initial_device is not None:
                print(f"[Scorer __init__] Model loaded via load_pretrained_model. Initial parameter device (example): {initial_device.device}")
            else:
                 print("[Scorer __init__] Model loaded, but has no parameters.")

            # --- 强制将模型及其所有参数和缓冲区移动到目标设备 ---
            print(f"[Scorer __init__] Explicitly moving model to {target_device}...")
            model = model.to(target_device)

            # --- 验证模型所有部分是否都在目标设备上 ---
            print(f"[Scorer __init__] Verifying model component devices after moving to {target_device}...")
            all_on_target = True
            issues = []
            # 检查参数
            for name, param in model.named_parameters():
                if param.device != target_device:
                    issues.append(f"  Parameter '{name}' is on {param.device}")
                    all_on_target = False
            # 检查缓冲区
            for name, buf in model.named_buffers():
                if buf.device != target_device:
                    issues.append(f"  Buffer '{name}' is on {buf.device}")
                    all_on_target = False

            if all_on_target:
                print(f"[Scorer __init__] All model parameters and buffers verified successfully on {target_device}.")
            else:
                print(f"[Scorer __init__] !! WARNING: Device placement issues detected after moving model to {target_device} !!")
                for issue in issues:
                    print(issue)
                # 根据需要，可以选择在这里抛出错误或继续
                # raise RuntimeError("Failed to move all model components to the target device.")
            # --- 结束验证 ---

            self.model = model
            self.tokenizer = tokenizer
            self.image_processor = image_processor

            # 准备 prompt 和 preferential_ids (这部分在 CPU 上操作)
            prompt = "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"
            # 获取 token IDs (通常在 CPU 完成)
            self.preferential_ids_ = [id_[1] for id_ in tokenizer(["excellent","good","fair","poor","bad"])["input_ids"]]
            print(f"[Scorer __init__] Preferential token IDs obtained.")

            # --- 确保 weight_tensor 和 input_ids 在目标设备上 ---
            # 应该在模型确认在目标设备之后进行
            # .half() 可以在 .to() 之前或之后，通常先转换类型
            weight_tensor_cpu = torch.Tensor([5., 4., 3., 2., 1.])
            self.weight_tensor = weight_tensor_cpu.half().to(target_device)
            print(f"[Scorer __init__] weight_tensor created and moved to {self.weight_tensor.device}. Expected: {target_device}")

            # 使用 tokenizer_image_token 生成 input_ids (在 CPU 上完成)
            input_ids_cpu = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            # 然后将结果移动到目标设备
            self.input_ids = input_ids_cpu.unsqueeze(0).to(target_device)
            print(f"[Scorer __init__] input_ids created and moved to {self.input_ids.device}. Expected: {target_device}")
            # --- 结束确保 ---

            print(f"[Scorer __init__] Scorer initialization on device {target_device} completed successfully.")

        except Exception as e:
            print(f"[Scorer __init__] !!! FAILED to initialize Scorer on device {target_device} !!!")
            traceback.print_exc()
            # 重新抛出异常，因为初始化失败意味着 Scorer 无法使用
            raise RuntimeError(f"Scorer initialization failed on device {target_device}") from e

    def expand2square(self, pil_img, background_color):
        """将图像扩展为正方形，用指定背景色填充。"""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def forward(self, image: List[Image.Image]):
        """
        对一批 PIL 图像进行 DeQA 评分。

        Args:
            image (List[Image.Image]): 输入的 PIL 图像列表。

        Returns:
            torch.Tensor: 每个图像对应的 DeQA 分数张量，形状为 (batch_size,)。
        """
        # 1. 确定期望的设备 (应该与模型参数一致)
        # 在每次 forward 调用时检查，以防模型设备意外改变
        try:
            expected_device = next(self.model.parameters()).device
        except StopIteration:
             # 如果模型没有参数，则使用 Scorer 初始化时确定的设备
             expected_device = self.input_ids.device # 或者 self.weight_tensor.device
             print(f"[Scorer forward] Warning: Model has no parameters. Using fallback device: {expected_device}")

        # print(f"[Scorer forward] Starting forward pass. Expected device: {expected_device}") # 调试日志

        # 2. 图像预处理 (在 CPU 上操作 PIL Image)
        try:
            # 计算背景色（从 image_processor 的均值转换）
            background_color = tuple(int(x * 255) for x in self.image_processor.image_mean)
            image = [self.expand2square(img, background_color) for img in image]
        except Exception as e:
            print("[Scorer forward] Error during expand2square preprocessing.")
            traceback.print_exc()
            raise RuntimeError("Image preprocessing failed in Scorer") from e

        # 3. 使用 torch.inference_mode() 避免梯度计算和内存占用
        with torch.inference_mode():
            # 3.1 图像张量化和移动到设备
            try:
                # image_processor.preprocess 通常返回 CPU 张量
                image_tensor_cpu = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
                # --- 显式转换类型并移动到目标设备 ---
                image_tensor = image_tensor_cpu.half().to(expected_device)
                # print(f"[Scorer forward] image_tensor device: {image_tensor.device}") # 调试日志
            except Exception as e:
                print(f"[Scorer forward] Error during image_processor.preprocess or moving image tensor to {expected_device}.")
                traceback.print_exc()
                raise RuntimeError("Image tensor processing/moving failed") from e

            # 3.2 准备 input_ids (根据 batch size 重复)
            try:
                 batch_input_ids = self.input_ids.repeat(image_tensor.shape[0], 1)
            except Exception as e:
                print("[Scorer forward] Error repeating input_ids.")
                traceback.print_exc()
                raise RuntimeError("Failed to prepare batch_input_ids") from e

            # --- 4. 添加断言进行设备检查 (在调用模型前) ---
            try:
                assert self.input_ids.device == expected_device, f"Input IDs device mismatch: {self.input_ids.device} vs {expected_device}"
                assert batch_input_ids.device == expected_device, f"Repeated Input IDs device mismatch: {batch_input_ids.device} vs {expected_device}"
                assert image_tensor.device == expected_device, f"Image tensor device mismatch: {image_tensor.device} vs {expected_device}"
                model_param_device = next(self.model.parameters()).device
                assert model_param_device == expected_device, f"Model parameter device mismatch: {model_param_device} vs {expected_device}"
                # 检查 weight_tensor
                assert self.weight_tensor.device == expected_device, f"Weight tensor device mismatch: {self.weight_tensor.device} vs {expected_device}"
            except AssertionError as ae:
                 print(f"[Scorer forward] !!! Device Assertion Failed before model call !!!")
                 print(f"    Assertion Error: {ae}")
                 print(f"    Current devices: input_ids={self.input_ids.device}, batch_input_ids={batch_input_ids.device}, image_tensor={image_tensor.device}, model_param={next(self.model.parameters()).device}, weight_tensor={self.weight_tensor.device}")
                 print(f"    Expected device: {expected_device}")
                 raise ae # 重新抛出断言错误
            # --- 结束断言 ---

            # 5. 模型推理
            try:
                # print(f"[Scorer forward] Calling self.model with input_ids: {batch_input_ids.shape} ({batch_input_ids.device}), images: {image_tensor.shape} ({image_tensor.device})")
                output_logits = self.model(
                            input_ids=batch_input_ids,
                            images=image_tensor
                        )["logits"] # 获取所有 logits

                # 提取所需 preferential_ids 对应的 logits
                # 假设 preferential_ids_ 是 CPU 列表，可以直接用于索引 GPU 张量
                output_logits_preferential = output_logits[:, -1, self.preferential_ids_] # 取最后一个 token 的 logits，再取特定 id 的

            except RuntimeError as e:
                print(f"[Scorer forward] !!! RuntimeError during self.model call !!!")
                print(f"    Input shapes: input_ids={batch_input_ids.shape}, images={image_tensor.shape}")
                print(f"    Input devices: input_ids={batch_input_ids.device}, images={image_tensor.device}, model={next(self.model.parameters()).device}")
                print(f"    Error Message: {e}")
                traceback.print_exc()
                # 检查是否有设备不匹配的特定信息
                if "Expected all tensors to be on the same device" in str(e):
                    print("[Scorer forward] Suggestion: Check the internal implementation of the loaded model for potential hardcoded CPU tensors or operations.")
                raise e # 重新抛出异常，让上层处理

            except Exception as e:
                 print(f"[Scorer forward] !!! Unexpected Error during self.model call !!!")
                 traceback.print_exc()
                 raise e

            # 6. 计算最终分数
            try:
                # weight_tensor 应该已经在正确的设备上 (已断言检查)
                # softmax 和矩阵乘法应该在 logits 所在的设备上进行
                final_scores = torch.softmax(output_logits_preferential, dim=-1) @ self.weight_tensor
                # print(f"[Scorer forward] Final scores calculated. Device: {final_scores.device}") # 调试日志

                # --- 断言检查最终分数的设备 ---
                assert final_scores.device == expected_device, f"Final scores device mismatch: {final_scores.device} vs {expected_device}"

            except Exception as e:
                print("[Scorer forward] Error calculating final scores from logits.")
                print(f"    Logits shape: {output_logits_preferential.shape}, device: {output_logits_preferential.device}")
                print(f"    Weight tensor shape: {self.weight_tensor.shape}, device: {self.weight_tensor.device}")
                traceback.print_exc()
                raise RuntimeError("Score calculation failed") from e

            return final_scores


# 主程序入口示例 (保持不变, 用于独立测试 Scorer)
if __name__ == "__main__":
    # 使用示例:
    try:
        # 尝试在 GPU 0 上初始化
        print("Testing Scorer initialization on cuda:0...")
        scorer = Scorer(pretrained="zhiyuanyou/DeQA-Score-Mix3", device="cuda:0")
        scorer.eval() # 设置为评估模式
        print("Scorer initialized successfully on cuda:0.")

        # 创建一个白色 PIL 图像列表作为测试输入
        print("Creating dummy PIL images for testing...")
        dummy_images = [Image.new('RGB', (224, 224), color = 'white') for _ in range(2)] # 创建 2 个图像
        print(f"Created {len(dummy_images)} dummy images.")

        # 使用 Scorer 进行评分
        print("Performing forward pass...")
        with torch.no_grad(): # 确保在无梯度环境下运行
             scores = scorer(dummy_images)
        print(f"Forward pass completed. Calculated scores: {scores}")
        print(f"Scores tensor device: {scores.device}")

        # 可以在不同的设备上测试
        if torch.cuda.device_count() > 1:
             print("\nTesting Scorer initialization on cuda:1...")
             try:
                 scorer_gpu1 = Scorer(pretrained="zhiyuanyou/DeQA-Score-Mix3", device="cuda:1")
                 scorer_gpu1.eval()
                 print("Scorer initialized successfully on cuda:1.")
                 with torch.no_grad():
                     scores_gpu1 = scorer_gpu1(dummy_images)
                 print(f"Calculated scores on cuda:1: {scores_gpu1}")
                 print(f"Scores tensor device: {scores_gpu1.device}")
             except Exception as e:
                  print(f"Failed to test on cuda:1: {e}")
        else:
             print("\nSkipping cuda:1 test as only one GPU is available or CUDA is not fully configured.")

        # 测试 CPU
        print("\nTesting Scorer initialization on cpu...")
        try:
            scorer_cpu = Scorer(pretrained="zhiyuanyou/DeQA-Score-Mix3", device="cpu")
            scorer_cpu.eval()
            print("Scorer initialized successfully on cpu.")
            with torch.no_grad():
                scores_cpu = scorer_cpu(dummy_images)
            print(f"Calculated scores on cpu: {scores_cpu}")
            print(f"Scores tensor device: {scores_cpu.device}")
        except Exception as e:
             print(f"Failed to test on cpu: {e}")


    except Exception as e:
        print("\nAn error occurred during the Scorer test.")
        traceback.print_exc()

