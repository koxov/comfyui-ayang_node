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
        # 返回一个占位图像而不是空张量
        placeholder = Image.new('RGB', (64, 64), color=(255, 255, 255))
        pils = [placeholder]

    # 确保所有图像尺寸一致
    first_size = pils[0].size
    for i, pil in enumerate(pils):
        if pil.size != first_size:
            # 调整尺寸以匹配第一张图像
            pils[i] = pil.resize(first_size, Image.LANCZOS)
            print(f"警告：图像{i}尺寸{pil.size}与第一张图像{first_size}不一致，已调整尺寸")

    np_imgs = []
    for pil in pils:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil, dtype=np.uint8)  # [H,W,3]
        np_imgs.append(arr)

    batch = np.stack(np_imgs, axis=0).astype(np.float32) / 255.0  # [B,H,W,3]
    return torch.from_numpy(batch)


class OpenRouterTextToImage:
    """
    使用OpenRouter进行文生图的节点
    """
    CATEGORY = "OpenRouter"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"multiline": False,
                                    "default": "google/gemini-2.5-flash-image-preview:free"}),
            },
            "optional": {
                # 备用API Key
                "api_key_1": ("STRING", {"multiline": False, "default": ""}),
                "api_key_2": ("STRING", {"multiline": False, "default": ""}),
                "api_key_3": ("STRING", {"multiline": False, "default": ""}),
                "api_key_4": ("STRING", {"multiline": False, "default": ""}),
                "api_key_5": ("STRING", {"multiline": False, "default": ""}),
                "image_format": (["jpeg", "png"], {"default": "jpeg"}),
                # 内置白底画布裁剪预设
                "canvas_preset": ([
                    "1:1 - 1024x1024",
                    "3:4 - 896x1152",
                    "5:8 - 832x1216",
                    "9:16 - 768x1344",
                    "9:21 - 640x1536",
                    "4:3 - 1152x896",
                    "3:2 - 1216x832",
                    "16:9 - 1344x768",
                ], {"default": "1:1 - 1024x1024"}),
                "attach_canvas": (["True", "False"], {"default": "True"}),
            },
        }

    def _call_openrouter(
        self,
        api_key: str,
        pil_refs: List[Image.Image],
        prompt_text: str,
        model: str,
        image_format: str = "jpeg"
    ) -> Tuple[List[Image.Image], str, bool]:
        """调用OpenRouter API生成图像"""
        if OpenAI is None:
            return [], "请先安装openai库: pip install openai", False

        if not api_key:
            return [], "API Key不能为空", False

        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

            # 在提示前加固定前缀，指明在空白画布上生成
            # 如果传入参考画布，优先从参考图获取目标像素并明确要求模型返回精确尺寸
            size_req = ""
            if pil_refs and len(pil_refs) > 0 and isinstance(pil_refs[0], Image.Image):
                try:
                    w, h = pil_refs[0].size
                    size_req = f"请以 {w}x{h} 像素精确输出最终图像，不要缩放、裁剪或添加边框；输出必须与该像素完全一致。\n"
                except Exception:
                    size_req = ""

            full_prompt = (
                f"{size_req}请在空白的画布上帮我生成：{prompt_text}\n"
                "仅返回生成的图像，不要返回任何文字描述或额外说明。"
            )

            # 如果有内嵌或裁剪后的参考画布，优先附加第一张参考图片（放在文本前），以便模型在该画布上修改
            content_items = []
            if pil_refs and len(pil_refs) > 0:
                try:
                    # 只发送第一张参考画布以降低请求体大小
                    data_url = _pil_to_base64_data_url(pil_refs[0], format=image_format)
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                    print("附加参考画布到请求（第一张参考图）")
                except Exception as e:
                    return [], f"参考图转换失败: {str(e)}", False

            # 将文本提示放在图片之后
            content_items.append({"type": "text", "text": full_prompt})

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
            err_msg = f"API调用错误: {str(e)}"
            status_code = getattr(e, 'status_code', 0)
            retryable_codes = {401, 403, 429, 502, 503, 504}
            retryable = status_code in retryable_codes
            return [], err_msg, retryable
        except Exception as e:
            tb = traceback.format_exc()
            return [], f"生成图片时出错: {tb}", False

    def generate(
        self,
        api_key: str,
        prompt: str = "",
        model: str = "google/gemini-2.5-flash-image-preview:free",
        api_key_1: str = "",
        api_key_2: str = "",
        api_key_3: str = "",
        api_key_4: str = "",
        api_key_5: str = "",
        image_format: str = "jpeg",
        canvas_preset: str = "1:1 - 1024x1024",
        attach_canvas: str = "True",
    ):
        """文生图主函数"""
        # 收集可用API Key
        api_keys = [k for k in [api_key, api_key_1, api_key_2, api_key_3, api_key_4, api_key_5] if k.strip()]
        if not api_keys:
            # 返回占位图像而不是空张量
            placeholder = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            out_tensor = _pils_to_tensor([placeholder])
            return out_tensor, "错误：未提供有效API Key，请至少填写一个（返回占位画布）"

        key_index = 0
        def next_key() -> str:
            nonlocal key_index
            key = api_keys[key_index % len(api_keys)]
            key_index += 1
            return key

        # 验证提示词
        prompt_text = prompt.strip()
        if not prompt_text:
            # 返回占位图像而不是空张量
            placeholder = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            out_tensor = _pils_to_tensor([placeholder])
            return out_tensor, "错误：请输入提示词（返回占位画布）"

        # 生成内置2048x2048白底画布并根据预设进行裁剪，作为参考图发送
        preset_map = {
            "1:1 - 1024x1024": (1024, 1024),
            "3:4 - 896x1152": (896, 1152),
            "5:8 - 832x1216": (832, 1216),
            "9:16 - 768x1344": (768, 1344),
            "9:21 - 640x1536": (640, 1536),
            "4:3 - 1152x896": (1152, 896),
            "3:2 - 1216x832": (1216, 832),
            "16:9 - 1344x768": (1344, 768),
        }

        canvas_size = preset_map.get(canvas_preset, (1024, 1024))
        try:
            # 内嵌白底2048x2048
            base_canvas = Image.new('RGB', (2048, 2048), color=(255, 255, 255))
            target_w, target_h = canvas_size
            # 中心裁剪
            left = (2048 - target_w) // 2
            top = (2048 - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            cropped = base_canvas.crop((left, top, right, bottom))
            # 将裁剪结果作为参考图
            pil_refs = [cropped]
        except Exception as e:
            pil_refs = []
            target_w, target_h = canvas_size
            print(f"内置画布生成/裁剪失败: {str(e)}")

        # 明确告知模型输出尺寸要求，增加被遵从的概率
        size_instruction = f" 请以 {target_w}x{target_h} 像素输出最终图像，并返回该尺寸的图片。"
        prompt_text = prompt_text + size_instruction

        # 是否将内置画布以 data URL 附加到请求中（有时会导致请求体过大或 API 返回 400）
        attach = True if str(attach_canvas).lower() in ("true", "1", "yes") else False

        # 生成逻辑
        attempts = len(api_keys)
        last_err = ""
        success_pils: List[Image.Image] = []

        for attempt in range(attempts):
            used_key = next_key()
            # 只有在 attach=True 时才把裁剪画布作为参考图发送
            refs_to_send = pil_refs if attach else []
            out_pils, err, retryable = self._call_openrouter(
                used_key, refs_to_send, prompt_text, model, image_format
            )

            if out_pils:
                success_pils = out_pils
                break
            if not retryable:
                # 返回占位裁剪白图，避免空输出导致 downstream 报错
                if not pil_refs:
                    placeholder = Image.new('RGB', (target_w, target_h), color=(255, 255, 255))
                    placeholder_list = [placeholder]
                else:
                    placeholder_list = pil_refs
                out_tensor = _pils_to_tensor(placeholder_list)
                return out_tensor, f"使用Key #{attempt+1}发生不可重试错误: {err}（返回占位画布）"
            last_err = err

        if not success_pils:
            # 所有Key尝试失败，返回裁剪后的白底占位图而不是空张量
            if not pil_refs:
                placeholder = Image.new('RGB', (target_w, target_h), color=(255, 255, 255))
                placeholder_list = [placeholder]
            else:
                placeholder_list = pil_refs
            out_tensor = _pils_to_tensor(placeholder_list)
            return out_tensor, f"所有API Key尝试失败: {last_err}（返回占位画布）"

        # 准备输出
        # 若模型返回图片但尺寸与目标不符，则在节点端进行中心裁剪并缩放到目标尺寸，确保输出像素满足选择
        def center_crop_and_resize(img: Image.Image, tw: int, th: int) -> Image.Image:
            img = img.convert('RGB')
            w, h = img.size
            # 先按目标长宽比裁剪中心区域
            src_ratio = w / h
            tgt_ratio = tw / th
            if src_ratio > tgt_ratio:
                # 源图更宽，裁剪左右
                new_w = int(h * tgt_ratio)
                left = (w - new_w) // 2
                img = img.crop((left, 0, left + new_w, h))
            elif src_ratio < tgt_ratio:
                # 源图更高，裁剪上下
                new_h = int(w / tgt_ratio)
                top = (h - new_h) // 2
                img = img.crop((0, top, w, top + new_h))
            # 最后缩放到目标像素
            if img.size != (tw, th):
                img = img.resize((tw, th), resample=Image.LANCZOS)
            return img

        # 处理并确保尺寸
        processed_pils: List[Image.Image] = []
        for p in success_pils:
            try:
                if p.size != (target_w, target_h):
                    proc = center_crop_and_resize(p, target_w, target_h)
                else:
                    proc = p.convert('RGB')
                processed_pils.append(proc)
            except Exception:
                # 如果处理失败，退回使用原图
                processed_pils.append(p.convert('RGB'))

        out_tensor = _pils_to_tensor(processed_pils)
        status = (f"成功生成{len(success_pils)}张图片 "
              f"(文生图模式, 使用Key #{(key_index-1)%len(api_keys)+1}, 尝试{attempt+1}/{attempts})")
        return (out_tensor, status)


class OpenRouterImageToImage:
    """
    使用OpenRouter进行图生图的节点
    """
    CATEGORY = "OpenRouter"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"multiline": False,
                                    "default": "google/gemini-2.5-flash-image-preview:free"}),
                "image": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                # 备用API Key
                "api_key_1": ("STRING", {"multiline": False, "default": ""}),
                "api_key_2": ("STRING", {"multiline": False, "default": ""}),
                "api_key_3": ("STRING", {"multiline": False, "default": ""}),
                "api_key_4": ("STRING", {"multiline": False, "default": ""}),
                "api_key_5": ("STRING", {"multiline": False, "default": ""}),
                "image_format": (["jpeg", "png"], {"default": "jpeg"}),
            },
        }

    def _call_openrouter(
        self,
        api_key: str,
        pil_refs: List[Image.Image],
        prompt_text: str,
        model: str,
        image_format: str = "jpeg"
    ) -> Tuple[List[Image.Image], str, bool]:
        """调用OpenRouter API生成图像"""
        if OpenAI is None:
            return [], "请先安装openai库: pip install openai", False

        if not api_key:
            return [], "API Key不能为空", False

        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            
            # 图生图提示词
            full_prompt = (f"请参考提供的图片内容和风格，根据以下提示词生成新图片：{prompt_text}\n"
                          "直接返回生成的图像，无需任何文字描述或额外说明。")
                
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
            err_msg = f"API调用错误: {str(e)}"
            status_code = getattr(e, 'status_code', 0)
            retryable_codes = {401, 403, 429, 502, 503, 504}
            retryable = status_code in retryable_codes
            return [], err_msg, retryable
        except Exception as e:
            tb = traceback.format_exc()
            return [], f"生成图片时出错: {tb}", False

    def generate(
        self,
        api_key: str,
        prompt: str = "",
        model: str = "google/gemini-2.5-flash-image-preview:free",
        image=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        api_key_1: str = "",
        api_key_2: str = "",
        api_key_3: str = "",
        api_key_4: str = "",
        api_key_5: str = "",
        image_format: str = "jpeg",
    ):
        """图生图主函数"""
        # 收集可用API Key
        api_keys = [k for k in [api_key, api_key_1, api_key_2, api_key_3, api_key_4, api_key_5] if k.strip()]
        if not api_keys:
            # 返回占位图像而不是空张量
            placeholder = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            out_tensor = _pils_to_tensor([placeholder])
            return out_tensor, "错误：未提供有效API Key，请至少填写一个（返回占位画布）"

        key_index = 0
        def next_key() -> str:
            nonlocal key_index
            key = api_keys[key_index % len(api_keys)]
            key_index += 1
            return key

        # 验证提示词
        prompt_text = prompt.strip()
        if not prompt_text:
            # 返回占位图像而不是空张量
            placeholder = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            out_tensor = _pils_to_tensor([placeholder])
            return out_tensor, "错误：请输入提示词（返回占位画布）"

        # 处理输入图像
        pil_refs = []
        try:
            # 转换主图像
            if image is not None:
                pil_refs.extend(_tensor_to_pils(image))
            # 转换额外图像
            for img in [image2, image3, image4, image5]:
                if img is not None:
                    pil_refs.extend(_tensor_to_pils(img))
        except Exception as e:
            # 返回占位图像而不是空张量
            placeholder = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            out_tensor = _pils_to_tensor([placeholder])
            return out_tensor, f"图像转换失败: {str(e)}（返回占位画布）"

        if not pil_refs:
            # 返回占位图像而不是空张量
            placeholder = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            out_tensor = _pils_to_tensor([placeholder])
            return out_tensor, "错误：未提供有效输入图像（返回占位画布）"

        # 生成逻辑
        attempts = len(api_keys)
        last_err = ""
        success_pils: List[Image.Image] = []

        for attempt in range(attempts):
            used_key = next_key()
            out_pils, err, retryable = self._call_openrouter(
                used_key, pil_refs, prompt_text, model, image_format
            )

            if out_pils:
                success_pils = out_pils
                break
            if not retryable:
                # 返回第一张参考图作为占位
                placeholder_list = [pil_refs[0]] if pil_refs else []
                out_tensor = _pils_to_tensor(placeholder_list)
                return out_tensor, f"使用Key #{attempt+1}发生不可重试错误: {err}（返回参考图）"
            last_err = err

        if not success_pils:
            # 所有Key尝试失败，返回第一张参考图作为占位
            placeholder_list = [pil_refs[0]] if pil_refs else []
            out_tensor = _pils_to_tensor(placeholder_list)
            return out_tensor, f"所有API Key尝试失败: {last_err}（返回参考图）"

        # 转换为张量输出
        out_tensor = _pils_to_tensor(success_pils)
        status = (f"成功生成{len(success_pils)}张图片 "
              f"(图生图模式, 使用Key #{(key_index-1)%len(api_keys)+1}, 尝试{attempt+1}/{attempts})")
        return (out_tensor, status)


class CanvasSizeCropForImg2Img:
    """生成内置2048x2048白底并裁剪为选定尺寸，用作图生图的输入（控制输出尺寸）。"""
    CATEGORY = "AICG阿洋"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "canvas_preset": ([
                    "1:1 - 1024x1024",
                    "3:4 - 896x1152",
                    "5:8 - 832x1216",
                    "9:16 - 768x1344",
                    "9:21 - 640x1536",
                    "4:3 - 1152x896",
                    "3:2 - 1216x832",
                    "16:9 - 1344x768",
                ], {"default": "1:1 - 1024x1024"}),
            },
        }

    def generate(self, **kwargs):
        # 兼容 ComfyUI 调用：优先从 kwargs 读取 canvas_preset
        canvas_preset = kwargs.get('canvas_preset', "1:1 - 1024x1024")

        preset_map = {
            "1:1 - 1024x1024": (1024, 1024),
            "3:4 - 896x1152": (896, 1152),
            "5:8 - 832x1216": (832, 1216),
            "9:16 - 768x1344": (768, 1344),
            "9:21 - 640x1536": (640, 1536),
            "4:3 - 1152x896": (1152, 896),
            "3:2 - 1216x832": (1216, 832),
            "16:9 - 1344x768": (1344, 768),
        }

        canvas_size = preset_map.get(canvas_preset, (1024, 1024))
        try:
            # 生成2048x2048白底画布
            base_canvas = Image.new('RGB', (2048, 2048), color=(255, 255, 255))
            target_w, target_h = canvas_size
            # 中心裁剪到目标尺寸
            left = (2048 - target_w) // 2
            top = (2048 - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            cropped = base_canvas.crop((left, top, right, bottom))
            # 转换为张量返回
            out_tensor = _pils_to_tensor([cropped])
            return (out_tensor,)
        except Exception as e:
            # 出错时返回默认尺寸的白底图
            default_img = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            out_tensor = _pils_to_tensor([default_img])
            print(f"画布生成失败: {str(e)}，返回默认尺寸")
            return (out_tensor,)


# 注册到ComfyUI
NODE_CLASS_MAPPINGS = {
    "nanobanana apiAICG阿洋（文生图）": OpenRouterTextToImage,
    "nanobanana apiAICG阿洋（图生图）": OpenRouterImageToImage,
    "nanobanana 图生图尺寸调整": CanvasSizeCropForImg2Img,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "nanobanana apiAICG阿洋（文生图）": "nanobanana apiAICG阿洋（文生图）",
    "nanobanana apiAICG阿洋（图生图）": "nanobanana apiAICG阿洋（图生图）",
    "nanobanana 图生图尺寸调整": "图生图尺寸调整",
}