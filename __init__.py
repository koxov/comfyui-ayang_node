import os
import io
import base64
import traceback
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None  # 延迟报错，在调用时提示安装依赖
    OpenAIError = Exception  # 兼容异常处理


def _pil_to_base64_data_url(img: Image.Image, format: str = "jpeg") -> str:
    """将PIL图像转换为base64编码的data URL"""
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=format)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format};base64,{img_str}"


def _decode_image_from_openrouter_response(completion) -> Tuple[List[Image.Image], str]:
    """解析OpenRouter响应中的图片数据，返回PIL列表或错误信息"""
    try:
        response_dict = completion.model_dump()
        images_list = response_dict.get("choices", [{}])[0].get("message", {}).get("images", [])
        
        if not isinstance(images_list, list):
            return [], "模型回复格式错误：images不是列表类型"

        out_pils = []
        for image_info in images_list:
            base64_url = image_info.get("image_url", {}).get("url")
            if not base64_url:
                continue
                
            # 提取base64数据
            if "base64," in base64_url:
                base64_data = base64_url.split("base64,")[1]
            else:
                base64_data = base64_url

            try:
                img_bytes = base64.b64decode(base64_data)
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                out_pils.append(pil)
            except Exception as e:
                return [], f"图片解码失败: {str(e)}"

        if out_pils:
            return out_pils, ""
        return [], f"模型回复中未包含有效图片数据。\n\n--- 完整响应 ---\n{completion.model_dump_json(indent=2)}"
        
    except Exception as e:
        try:
            raw = completion.model_dump_json(indent=2)
        except Exception:
            raw = "<无法序列化响应>"
        return [], f"解析响应出错: {str(e)}\n\n--- 完整响应 ---\n{raw}"


def _tensor_to_pils(image) -> List[Image.Image]:
    """将ComfyUI的IMAGE张量转换为PIL图像列表"""
    if isinstance(image, dict) and "images" in image:
        image = image["images"]
        
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"期望输入为torch.Tensor，实际得到{type(image)}")
        
    # 处理单张图片的情况
    if image.ndim == 3:
        image = image.unsqueeze(0)
        
    # 验证张量维度
    if image.ndim != 4:
        raise ValueError(f"图像张量维度错误，期望4维，实际{image.ndim}维")

    # 转换为PIL图像
    arr = (image.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)  # [B,H,W,3]
    return [Image.fromarray(arr[i], mode="RGB") for i in range(arr.shape[0])]


def _pils_to_tensor(pils: List[Image.Image]) -> torch.Tensor:
    """将PIL图像列表转换为ComfyUI的IMAGE张量"""
    if not pils:
        return torch.zeros((0, 64, 64, 3), dtype=torch.float32)

    np_imgs = []
    for pil in pils:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil, dtype=np.uint8)  # [H,W,3]
        np_imgs.append(arr)
        
    batch = np.stack(np_imgs, axis=0).astype(np.float32) / 255.0  # [B,H,W,3]
    return torch.from_numpy(batch)


class OpenRouterImageGenerator:
    """
    使用OpenRouter生成图像的节点
    支持单提示词生成，可传入多张参考图，新增随机种控制功能
    """
    CATEGORY = "OpenRouter"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")  # 移除seed_used输出
    RETURN_NAMES = ("image", "status")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),  # 改为必填参数
                # 新增随机种控制参数
                "seed_mode": (["random", "fixed", "increase", "decrease"], {"default": "random"}),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
            },
            "optional": {
                "model": ("STRING", {"multiline": False, 
                                    "default": "google/gemini-2.5-flash-image-preview:free"}),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "api_key_1": ("STRING", {"multiline": False, "default": ""}),
                "api_key_2": ("STRING", {"multiline": False, "default": ""}),
                "api_key_3": ("STRING", {"multiline": False, "default": ""}),
                "api_key_4": ("STRING", {"multiline": False, "default": ""}),
                "api_key_5": ("STRING", {"multiline": False, "default": ""}),
                "image_format": (["jpeg", "png"], {"default": "jpeg"}),
            },
        }

    def _get_seed(self, seed_mode: str, base_seed: int) -> int:
        """根据种子模式计算最终使用的随机种"""
        if seed_mode == "random":
            return random.randint(0, 0xFFFFFFFF)
        elif seed_mode == "fixed":
            return base_seed
        elif seed_mode == "increase":
            return (base_seed + 1) % (0xFFFFFFFF + 1)  # 循环递增
        elif seed_mode == "decrease":
            return (base_seed - 1) % (0xFFFFFFFF + 1)  # 循环递减
        return base_seed

    def _call_openrouter(
        self,
        api_key: str,
        pil_refs: List[Image.Image],
        prompt_text: str,
        model: str,
        seed: int,  # 新增种子参数
        image_format: str = "jpeg"
    ) -> Tuple[List[Image.Image], str, bool]:
        """调用OpenRouter API生成图像，增加种子参数"""
        if OpenAI is None:
            return [], "请先安装openai库: pip install openai", False
            
        if not api_key:
            return [], "API Key不能为空", False

        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            headers = {}  # 移除site_url和site_name相关的header设置

            # 构建提示内容，加入种子信息以保证生成的可重复性
            full_prompt = (f"请根据参考图和以下提示词生成新图片，无需描述图片。"
                          f"提示词：'{prompt_text}' "
                          f"随机种子：{seed}（请使用此种子确保生成结果的可重复性）")
            content_items = [{"type": "text", "text": full_prompt}]

            # 添加参考图片
            for i, pil in enumerate(pil_refs):
                try:
                    data_url = _pil_to_base64_data_url(pil, format=image_format)
                    content_items.append({
                        "type": "image_url", 
                        "image_url": {"url": data_url}
                    })
                except Exception as e:
                    return [], f"参考图{i+1}转换失败: {str(e)}", False

            # 调用API
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_items}],
            )

            # 解析响应
            pils, err = _decode_image_from_openrouter_response(completion)
            if err:
                return [], err, False
            if not pils:
                return [], "未从模型收到图片数据", False
            return pils, "", False

        except OpenAIError as e:
            # 处理OpenAI特定错误
            err_msg = f"API调用错误: {str(e)}"
            status_code = getattr(e, 'status_code', 0)
            # 判断可重试状态码
            retryable_codes = {401, 403, 429, 502, 503, 504}
            retryable = status_code in retryable_codes
            return [], err_msg, retryable
        except Exception as e:
            tb = traceback.format_exc()
            return [], f"生成图片时出错: {tb}", False

    def generate(
        self,
        api_key: str,
        image,
        prompt: str = "",
        model: str = "google/gemini-2.5-flash-image-preview:free",
        # 新增随机种参数
        seed_mode: str = "random",
        base_seed: int = 0,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        image5: Optional[torch.Tensor] = None,
        api_key_1: str = "",
        api_key_2: str = "",
        api_key_3: str = "",
        api_key_4: str = "",
        api_key_5: str = "",
        image_format: str = "jpeg",
    ):
        """主生成函数，增加随机种处理逻辑"""
        # 计算当前使用的随机种
        current_seed = self._get_seed(seed_mode, base_seed)
        
        # 处理参考图片
        try:
            pils_main = _tensor_to_pils(image)
            if not pils_main:
                return image, "错误：参考图像不能为空"
        except Exception as e:
            return image, f"输入图像解析失败：{str(e)}"

        # 收集所有参考图
        pils_refs: List[Image.Image] = [pils_main[0]]
        for opt_img in (image2, image3, image4, image5):
            if opt_img is None:
                continue
            try:
                pils_opt = _tensor_to_pils(opt_img)
                if pils_opt:
                    pils_refs.append(pils_opt[0])
            except Exception as e:
                # 仅警告，不中断执行
                print(f"可选图像解析警告: {str(e)}")

        # 收集可用API Key
        api_keys = [k for k in [api_key_1, api_key_2, api_key_3, api_key_4, api_key_5] if k.strip()]
        if not api_keys and api_key.strip():
            api_keys = [api_key]
        if not api_keys:
            return image, "错误：未提供有效API Key，请至少填写一个"

        key_index = 0
        def next_key() -> str:
            nonlocal key_index
            key = api_keys[key_index % len(api_keys)]
            key_index += 1
            return key

        # 验证提示词
        prompt_text = prompt.strip()
        if not prompt_text:
            return image, "错误：请输入提示词"

        # 生成逻辑
        attempts = len(api_keys)
        last_err = ""
        success_pils: List[Image.Image] = []

        for attempt in range(attempts):
            used_key = next_key()
            # 调用时传入随机种
            out_pils, err, retryable = self._call_openrouter(
                used_key, pils_refs, prompt_text, model, current_seed, image_format
            )
            
            if out_pils:
                success_pils = out_pils
                break
            if not retryable:
                return image, f"使用Key #{attempt+1}发生不可重试错误: {err}"
            last_err = err

        if not success_pils:
            return image, f"所有API Key尝试失败: {last_err}"

        # 准备输出，状态信息中包含当前使用的种子
        out_tensor = _pils_to_tensor(success_pils)
        status = (f"成功生成{len(success_pils)}张图片 "
                 f"(使用Key #{(key_index-1)%len(api_keys)+1}, 尝试{attempt+1}/{attempts}, "
                 f"种子: {current_seed})")
        return (out_tensor, status)


# 注册到ComfyUI
NODE_CLASS_MAPPINGS = {
    "nanobanana apiAICG阿洋": OpenRouterImageGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "nanobanana apiAICG阿洋": "nanobanana apiAICG阿洋",
}