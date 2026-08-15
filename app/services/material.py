import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, List
from urllib.parse import quote_plus, urlencode, urlsplit, urlunsplit

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.models.article import MediaAsset, safe_public_page
from app.models.schema import MaterialInfo, VideoAspect, VideoConcatMode
from app.services import material_cache, task_artifacts
from app.utils import utils

# Thread-safe counter for API key rotation
_api_key_counter = 0
_api_key_lock = threading.Lock()
_MIN_VIDEO_RENDITION_DIMENSION = 480
_DEFAULT_MATERIAL_DOWNLOAD_CONCURRENCY = 2
_MAX_MATERIAL_DOWNLOAD_CONCURRENCY = 4


def _safe_public_url(value: Any) -> str | None:
    """
    只保留可公开展示的 HTTP(S) 页面地址，并移除查询参数和凭据。

    素材下载地址可能携带 API Key、签名 JWT 或临时 token。任务清单只需要
    帮助用户回到供应商的公开素材页，不应保存鉴权参数；用户信息形式的 URL
    同样拒绝，避免 ``https://user:pass@example.com`` 一类内容落盘。
    """
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _creator_info(value: Any) -> dict[str, str] | None:
    """从不同供应商的作者结构中提取统一的公开字段。"""
    if isinstance(value, str) and value.strip():
        return {"name": value.strip()}
    if not isinstance(value, dict):
        return None

    creator: dict[str, str] = {}
    creator_id = value.get("id")
    creator_name = value.get("name") or value.get("username")
    creator_page = _safe_public_url(
        value.get("url") or value.get("profile_url") or value.get("profile_page")
    )
    if creator_id is not None:
        creator["id"] = str(creator_id)
    if creator_name:
        creator["name"] = str(creator_name)
    if creator_page:
        creator["profile_page"] = creator_page
    return creator or None


def _material_source_record(item: MaterialInfo, local_path: str) -> dict[str, Any]:
    """
    为成功下载的素材生成轻量来源记录。

    ``source_info`` 可能来自缓存，甚至来自外部构造的 ``MaterialInfo``，因此
    不能原样写入。这里按白名单重新构造，只保留公开页面、业务标识和尺寸，
    并只记录本地文件名，避免用户目录或 Docker 挂载路径进入任务文件。
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    record: dict[str, Any] = {
        "provider": str(item.provider or source.get("provider") or ""),
        "local_file": Path(local_path).name,
        "duration": int(item.duration),
    }

    search_term = source.get("search_term")
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if isinstance(search_term, str) and search_term.strip():
        record["search_term"] = search_term.strip()
    if asset_id not in (None, ""):
        record["asset_id"] = str(asset_id)
    if source_page:
        record["source_page"] = source_page

    creator = _creator_info(source.get("creator"))
    if creator:
        record["creator"] = creator

    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            record["rendition"] = rendition
    return record


def _persist_material_sources(
    task_id: str,
    material_sources: list[dict[str, Any]],
) -> None:
    """
    将当前实际下载成功的素材来源补充到任务清单。

    任务记录是辅助能力，不能改变视频下载函数的返回值，也不能因为写盘失败
    中断成片主流程。``patch_script_data`` 会负责原子替换和异常日志；这里仅在
    成功后记录数量，便于确认任务追溯信息是否已经落盘。
    """
    try:
        saved = task_artifacts.patch_script_data(
            task_id,
            material_sources=material_sources,
        )
        if saved:
            logger.info(
                f"saved material source records: "
                f"task_id={task_id}, count={len(material_sources)}"
            )
    except Exception as exc:
        # task_artifacts 自身已经按失败降级设计，这里仍保留最后一道隔离，
        # 防止未来实现调整或目录解析异常意外影响素材下载返回值。
        logger.warning(
            "failed to persist material source records: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )


def _get_tls_verify() -> bool:
    # 默认开启 TLS 证书校验，防止素材搜索和下载过程被中间人篡改。
    # 仅在企业代理、自签证书等明确需要的场景下，允许用户通过
    # `config.toml` 显式设置 `tls_verify = false` 临时关闭。
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")

    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )

    return bool(tls_verify)


def _video_rendition_dimensions(
    video: dict,
    target_width: int,
    target_height: int,
) -> tuple[int, int] | None:
    try:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width > 0 and height <= 0:
        height = int(round(width * (target_height / target_width)))
    elif height > 0 and width <= 0:
        width = int(round(height * (target_width / target_height)))
    if width < _MIN_VIDEO_RENDITION_DIMENSION or height < _MIN_VIDEO_RENDITION_DIMENSION:
        return None
    return width, height


def _video_rendition_score(
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> tuple:
    target_ratio = target_width / target_height
    rendition_ratio = width / height
    orientation_penalty = int((width >= height) != (target_width >= target_height))
    ratio_penalty = abs(rendition_ratio - target_ratio) / target_ratio
    meets_target = width >= target_width and height >= target_height
    target_area = target_width * target_height
    rendition_area = width * height

    if meets_target:
        # Prefer the smallest target-or-better rendition: same visual quality for
        # the final 1080p output, lower download time and less disk churn.
        return (orientation_penalty, ratio_penalty, 0, rendition_area - target_area)

    shortfall = max(0.0, (target_width - width) / target_width) + max(
        0.0, (target_height - height) / target_height
    )
    # If every rendition is below target, use the least-bad one.
    return (orientation_penalty, ratio_penalty, 1, shortfall, -rendition_area)


def _select_best_video_rendition(
    video_files,
    target_width: int,
    target_height: int,
    url_field: str,
) -> tuple[dict, str, int, int] | None:
    if isinstance(video_files, dict):
        iterable = video_files.items()
    else:
        iterable = enumerate(video_files or [])

    candidates = []
    for fallback_id, video in iterable:
        if not isinstance(video, dict):
            continue
        video_url = video.get(url_field)
        dimensions = _video_rendition_dimensions(video, target_width, target_height)
        if not video_url or dimensions is None:
            continue
        width, height = dimensions
        rendition_id = video.get("id")
        if rendition_id in (None, ""):
            rendition_id = fallback_id
        candidates.append(
            (
                _video_rendition_score(width, height, target_width, target_height),
                video,
                str(rendition_id),
                width,
                height,
            )
        )

    if not candidates:
        return None
    _, video, rendition_id, width, height = min(candidates, key=lambda candidate: candidate[0])
    return video, rendition_id, width, height


def _get_material_download_concurrency() -> int:
    try:
        configured = int(
            config.app.get(
                "material_download_concurrency",
                _DEFAULT_MATERIAL_DOWNLOAD_CONCURRENCY,
            )
        )
    except (TypeError, ValueError):
        configured = _DEFAULT_MATERIAL_DOWNLOAD_CONCURRENCY
    return max(1, min(_MAX_MATERIAL_DOWNLOAD_CONCURRENCY, configured))


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def _redact_secret(message: str, secret: str) -> str:
    """
    对即将写入日志的异常文本做最小范围脱敏。

    requests 的连接异常可能包含完整请求 URL，而 Pixabay API Key 通过查询
    参数传递。这里同时替换原始值和 URL 编码值，既保留网络错误信息用于排查，
    又避免密钥进入日志文件。
    """
    safe_message = str(message)
    if not secret:
        return safe_message

    safe_message = safe_message.replace(secret, "***")
    encoded_secret = quote_plus(secret)
    if encoded_secret != secret:
        safe_message = safe_message.replace(encoded_secret, "***")
    return safe_message


def _redact_request_error(error: Exception, *secrets: str) -> str:
    """
    保留网络异常的可排查信息，同时移除 API Key 和代理凭据。

    直接只记录异常类型会丢失 DNS、证书、超时等关键上下文；直接记录原始异常
    又可能回显完整请求 URL。统一入口可以让三个素材供应商使用相同脱敏规则。
    """
    safe_message = str(error)
    for secret in secrets:
        safe_message = _redact_secret(safe_message, str(secret or ""))
    for proxy_url in config.proxy.values():
        safe_message = _redact_secret(safe_message, str(proxy_url))
    return safe_message


def _is_cloudflare_challenge(response: requests.Response) -> bool:
    """
    识别 Cloudflare 返回的 HTML Challenge，而不是把它当成 Pixabay JSON。

    Cloudflare 通常会设置 `cf-mitigated: challenge`；部分部署只返回带有
    "Just a moment" 或 challenge-platform 的 HTML，因此保留内容特征兜底。
    响应正文仅在内存中判断，不写入日志，避免记录无价值的大段 HTML。
    """
    headers = getattr(response, "headers", {}) or {}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True

    content_type = str(headers.get("content-type", "")).lower()
    if "text/html" not in content_type:
        return False

    body = str(getattr(response, "text", "")).lower()
    return "just a moment" in body or "/cdn-cgi/challenge-platform/" in body


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_orientation = aspect.name
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 20, "orientation": video_orientation}
    query_url = f"https://api.pexels.com/videos/search?{urlencode(params)}"
    logger.info(f"searching videos on pexels: term={search_term!r}")

    try:
        r = requests.get(
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items = []
        if "videos" not in response:
            logger.error("pexels video search returned an unsupported response")
            return video_items
        videos = response["videos"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            selected = _select_best_video_rendition(
                v.get("video_files"),
                target_width=video_width,
                target_height=video_height,
                url_field="link",
            )
            if selected is None:
                continue
            video, rendition_id, w, h = selected
            item = MaterialInfo()
            item.provider = "pexels"
            item.url = video["link"]
            item.duration = duration
            item.source_info = {
                "provider": "pexels",
                "search_term": search_term,
                "asset_id": (
                    str(v.get("id")) if v.get("id") is not None else None
                ),
                "source_page": _safe_public_url(v.get("url")),
                "metadata_text": _coerce_metadata_text(
                    v.get("tags"),
                    _semantic_text_from_url(v.get("url")),
                ),
                "creator": _creator_info(v.get("user")),
                "rendition": {
                    "id": rendition_id,
                    "width": w,
                    "height": h,
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "pexels video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "video_type": "all",  # Accepted values: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(
        f"searching videos on pixabay: term={search_term!r}, "
        f"proxy_enabled={bool(config.proxy)}"
    )

    try:
        r = requests.get(
            query_url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(30, 60)
        )
        status_code = int(getattr(r, "status_code", 200))
        headers = getattr(r, "headers", {}) or {}
        content_type = str(headers.get("content-type", ""))
        retry_after = headers.get("retry-after")
        cf_ray = headers.get("cf-ray")

        if _is_cloudflare_challenge(r):
            logger.error(
                "pixabay search was blocked by a Cloudflare challenge: "
                f"status={status_code}, cf_ray={cf_ray or 'unknown'}. "
                "Check the server network or proxy, or use Pexels/Coverr instead."
            )
            return []

        if status_code == 429:
            logger.error(
                "pixabay API rate limit exceeded: "
                f"status=429, retry_after={retry_after or 'unknown'}"
            )
            return []

        if status_code >= 400:
            logger.error(
                "pixabay search request failed: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        try:
            response = r.json()
        except ValueError:
            logger.error(
                "pixabay returned an unexpected non-JSON response: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        video_items = []
        if "hits" not in response:
            logger.error("pixabay video search returned an unsupported response")
            return video_items
        videos = response["hits"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            selected = _select_best_video_rendition(
                v.get("videos"),
                target_width=video_width,
                target_height=video_height,
                url_field="url",
            )
            if selected is None:
                continue
            video, rendition_id, w, h = selected
            item = MaterialInfo()
            item.provider = "pixabay"
            item.url = video["url"]
            item.duration = duration
            item.source_info = {
                "provider": "pixabay",
                "search_term": search_term,
                "asset_id": (
                    str(v.get("id")) if v.get("id") is not None else None
                ),
                "source_page": _safe_public_url(v.get("pageURL")),
                "metadata_text": _coerce_metadata_text(
                    v.get("tags"),
                    _semantic_text_from_url(v.get("pageURL")),
                ),
                "creator": _creator_info(
                    {
                        "id": v.get("user_id"),
                        "name": v.get("user"),
                    }
                ),
                "rendition": {
                    "id": rendition_id,
                    "width": w,
                    "height": h,
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        error_message = _redact_request_error(e, api_key)
        logger.error(
            "pixabay search request failed: "
            f"error={type(e).__name__}, detail={error_message}"
        )

    return []


def search_videos_coverr(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """
    Coverr (https://coverr.co) - free HD/4K stock videos,
    subject to Coverr license terms (https://coverr.co/license).

    Coverr API notes (based on official docs at api.coverr.co/docs/):
      - 鉴权: Authorization: Bearer <api_key>
      - 搜索端点: GET /videos?query=...,响应结构 {"hits": [...], ...}
      - 加 ?urls=true 在搜索响应里直接返回 mp4 直链
      - URL 是 signed JWT(绑定 API key,无过期时间)
      - Coverr 库以 16:9 横屏为主,9:16 portrait 占比极低(约 1%)
        因此本函数不做 aspect_ratio 过滤,由下游 video.py 的
        resize + letterbox 逻辑统一处理
      - duration 字段同时存在 number 和 string 两种形态,本函数都接受

    本函数使用 urls.mp4_download 字段作为下载地址 —— 按 Coverr 官方文档
    (https://api.coverr.co/docs/videos/#download-a-video) 的说法,
    GET 这个 URL 本身就被 Coverr 当作一次合法的 download 事件计入统计,
    无需再调用 PATCH /videos/:id/stats/downloads。
    """
    api_key = get_api_key("coverr_api_keys")
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "query": search_term,
        "page_size": 20,
        "urls": "true",
        "sort": "popular",
    }
    query_url = f"https://api.coverr.co/videos?{urlencode(params)}"
    logger.info(f"searching videos on coverr: term={search_term!r}")

    try:
        r = requests.get(
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items: List[MaterialInfo] = []

        if not isinstance(response, dict) or "hits" not in response:
            logger.error("coverr video search returned an unsupported response")
            return video_items

        for v in response["hits"]:
            # duration 在不同响应里可能是 number(11.625) 或 string("10.500000")
            try:
                duration = int(float(v.get("duration") or 0))
            except (TypeError, ValueError):
                continue
            if duration < minimum_duration:
                continue

            video_id = v.get("id")
            mp4_download_url = (v.get("urls") or {}).get("mp4_download")
            if not video_id or not mp4_download_url:
                continue

            item = MaterialInfo()
            item.provider = "coverr"
            item.url = mp4_download_url
            item.duration = duration
            item.source_info = {
                "provider": "coverr",
                "search_term": search_term,
                "asset_id": str(video_id),
                "source_page": _safe_public_url(v.get("canonical_url") or v.get("url")),
                "metadata_text": _coerce_metadata_text(
                    v.get("title"),
                    v.get("name"),
                    v.get("description"),
                    v.get("tags"),
                    _semantic_text_from_url(v.get("canonical_url") or v.get("url")),
                ),
                "creator": _creator_info(v.get("creator") or v.get("author")),
                "rendition": {
                    "id": "mp4_download",
                    "width": v.get("max_width"),
                    "height": v.get("max_height"),
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "coverr video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


def _validate_video_file(video_path: str) -> bool:
    if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
        return False

    clip = None
    try:
        clip = VideoFileClip(video_path)
        return bool(clip.duration > 0 and clip.fps > 0)
    except Exception as e:
        logger.warning(f"invalid video file: {video_path} => {str(e)}")
        return False
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as close_error:
                logger.warning(
                    f"failed to close video clip: {video_path}, error: {str(close_error)}"
                )


def _remove_file_safely(file_path: str) -> None:
    try:
        os.remove(file_path)
    except FileNotFoundError:
        return
    except Exception as remove_error:
        logger.warning(
            f"failed to remove file: {file_path}, error: {str(remove_error)}"
        )


def save_video(video_url: str, save_dir: str = "") -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url_without_query = video_url.split("?")[0]
    url_hash = utils.md5(url_without_query)
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"

    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        if _validate_video_file(video_path):
            logger.info(f"video already exists: {video_path}")
            return video_path
        _remove_file_safely(video_path)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    partial_video_path = f"{video_path}.part"
    _remove_file_safely(partial_video_path)
    response = None
    try:
        response = requests.get(
            video_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(60, 240),
            stream=True,
        )
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()

        with open(partial_video_path, "wb") as f:
            if hasattr(response, "iter_content"):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            else:
                f.write(response.content)

        os.replace(partial_video_path, video_path)
    except Exception:
        _remove_file_safely(partial_video_path)
        _remove_file_safely(video_path)
        raise
    finally:
        if response is not None and hasattr(response, "close"):
            response.close()

    if _validate_video_file(video_path):
        return video_path
    _remove_file_safely(video_path)
    return ""


def _search_videos_with_cache(
    provider: str,
    search_videos: Callable[..., List[MaterialInfo]],
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    统一处理三个在线素材源的 24 小时搜索缓存。

    缓存只包裹搜索 API，不改变后续视频下载与去重逻辑。远端返回空列表时不写
    缓存，因为现有 provider 接口使用空列表同时表示“没有结果”和“请求失败”；
    在两者尚未拆分为明确结果类型前，宁可下次重试，也不能把临时故障缓存一天。
    """
    cache_args = {
        "provider": provider,
        "search_term": search_term,
        "minimum_duration": minimum_duration,
        "video_aspect": video_aspect,
    }

    def load_cache_safely() -> List[MaterialInfo] | None:
        try:
            return material_cache.load_material_search_cache(**cache_args)
        except Exception as exc:
            # 缓存是可选优化，任何缓存实现异常都必须按未命中处理，不能阻断
            # Pexels、Pixabay 或 Coverr 的正常远端搜索。
            logger.warning(
                "material search cache read failed, continue with remote search: "
                f"provider={provider}, error={type(exc).__name__}, detail={exc}"
            )
            return None

    cached_items = load_cache_safely()
    if cached_items is not None:
        return cached_items

    cache_lock = material_cache.get_material_search_cache_lock(**cache_args)
    with cache_lock:
        # 等待相同搜索条件的线程完成后再次读取，避免多个 API 任务在首次缓存
        # 未命中时同时请求远端，降低第三方接口限流和风控触发概率。
        cached_items = load_cache_safely()
        if cached_items is not None:
            return cached_items

        items = search_videos(
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        # Provider 正常会写入当前关键词，但测试替身、第三方扩展或旧实现可能
        # 遗漏或携带错误值。缓存读取会根据缓存键恢复该字段，因此远端结果也在
        # 同一入口校正，保证首次搜索与缓存命中的任务来源记录保持一致。
        for item in items:
            if isinstance(item.source_info, dict):
                item.source_info = dict(item.source_info)
                item.source_info["search_term"] = search_term
        if items:
            try:
                material_cache.save_material_search_cache(
                    **cache_args,
                    items=items,
                )
            except Exception as exc:
                logger.warning(
                    "material search cache write failed, use remote results: "
                    f"provider={provider}, error={type(exc).__name__}, detail={exc}"
                )
        return items


def _download_material_item(
    item: MaterialInfo,
    material_directory: str,
    max_clip_duration: int,
    *,
    ordered_search_term: str | None = None,
) -> tuple[str, dict[str, Any] | None, float] | None:
    source_info = item.source_info if isinstance(item.source_info, dict) else {}
    ordered_label = (
        f" for {ordered_search_term!r}" if ordered_search_term is not None else ""
    )
    logger.info(
        f"downloading{ordered_label} {item.provider} video: "
        f"asset_id={source_info.get('asset_id') or 'unknown'}"
    )
    saved_video_path = save_video(video_url=item.url, save_dir=material_directory)
    if not saved_video_path:
        return None

    logger.info(f"video saved: {saved_video_path}")
    material_source = None
    try:
        material_source = _material_source_record(item, saved_video_path)
    except Exception as source_error:
        # 来源记录异常不能把已经成功下载的素材视为下载失败，更不能
        # 阻断视频生成；保留供应商和异常类型用于后续定位。
        logger.warning(
            "failed to prepare material source record: "
            f"provider={item.provider}, "
            f"error={type(source_error).__name__}, detail={source_error}"
        )
    return saved_video_path, material_source, min(max_clip_duration, item.duration)


def _append_download_result(
    result: tuple[str, dict[str, Any] | None, float] | None,
    video_paths: list[str],
    material_sources: list[dict[str, Any]],
) -> float:
    if not result:
        return 0.0
    saved_video_path, material_source, seconds = result
    video_paths.append(saved_video_path)
    if material_source:
        material_sources.append(material_source)
    return float(seconds)


def _download_video_items_serial(
    items: List[MaterialInfo],
    material_directory: str,
    audio_duration: float,
    max_clip_duration: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    video_paths: list[str] = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    for item in items:
        try:
            result = _download_material_item(
                item=item,
                material_directory=material_directory,
                max_clip_duration=max_clip_duration,
            )
            total_duration += _append_download_result(
                result, video_paths, material_sources
            )
            if result and total_duration > audio_duration:
                logger.info(
                    f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                )
                break
        except Exception as e:
            logger.error(
                "failed to download material video: "
                f"provider={item.provider}, error={type(e).__name__}, "
                f"detail={_redact_request_error(e, item.url)}"
            )
    return video_paths, material_sources


def _download_video_items_parallel(
    items: List[MaterialInfo],
    material_directory: str,
    audio_duration: float,
    max_clip_duration: int,
    concurrency: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    video_paths: list[str] = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    next_index = 0
    futures = {}

    def submit_next(executor: ThreadPoolExecutor) -> None:
        nonlocal next_index
        if next_index >= len(items):
            return
        item = items[next_index]
        future = executor.submit(
            _download_material_item,
            item,
            material_directory,
            max_clip_duration,
        )
        futures[future] = item
        next_index += 1

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while next_index < len(items) and len(futures) < concurrency:
            submit_next(executor)

        while futures:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                item = futures.pop(future)
                try:
                    result = future.result()
                    total_duration += _append_download_result(
                        result, video_paths, material_sources
                    )
                except Exception as e:
                    logger.error(
                        "failed to download material video: "
                        f"provider={item.provider}, error={type(e).__name__}, "
                        f"detail={_redact_request_error(e, item.url)}"
                    )

            if total_duration > audio_duration:
                logger.info(
                    f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                )
                # Let already-started downloads finish and include them; avoid
                # launching more work after the target duration is covered.
                while futures:
                    done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = futures.pop(future)
                        try:
                            result = future.result()
                            _append_download_result(
                                result, video_paths, material_sources
                            )
                        except Exception as e:
                            logger.error(
                                "failed to download material video: "
                                f"provider={item.provider}, error={type(e).__name__}, "
                                f"detail={_redact_request_error(e, item.url)}"
                            )
                break

            while next_index < len(items) and len(futures) < concurrency:
                submit_next(executor)

    return video_paths, material_sources


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    match_script_order: bool = False,
) -> List[str]:
    provider = "pexels"
    remote_search_videos = search_videos_pexels
    if source == "pixabay":
        provider = "pixabay"
        remote_search_videos = search_videos_pixabay
    elif source == "coverr":
        provider = "coverr"
        remote_search_videos = search_videos_coverr

    def search_videos(
        search_term: str,
        minimum_duration: int,
        video_aspect: VideoAspect,
    ) -> List[MaterialInfo]:
        return _search_videos_with_cache(
            provider=provider,
            search_videos=remote_search_videos,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    if match_script_order:
        return _download_videos_by_script_order(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            provider=provider,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )

    search_started_at = time.perf_counter()
    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    for search_term in search_terms:
        video_items = search_videos(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            video_aspect=video_aspect,
        )
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    utils.log_runtime_benchmark(
        "material_search",
        search_started_at,
        source=provider,
        terms=len(search_terms),
        candidates=len(valid_video_items),
    )
    concat_mode_value = getattr(video_concat_mode, "value", video_concat_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    download_concurrency = _get_material_download_concurrency()
    download_started_at = time.perf_counter()
    effective_download_concurrency = 1
    if (
        concat_mode_value == VideoConcatMode.random.value
        and audio_duration > 0
        and download_concurrency > 1
    ):
        effective_download_concurrency = download_concurrency
        logger.info(
            f"downloading random materials with concurrency={download_concurrency}"
        )
        video_paths, material_sources = _download_video_items_parallel(
            items=valid_video_items,
            material_directory=material_directory,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            concurrency=download_concurrency,
        )
    else:
        video_paths, material_sources = _download_video_items_serial(
            items=valid_video_items,
            material_directory=material_directory,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
        )
    utils.log_runtime_benchmark(
        "material_download",
        download_started_at,
        source=provider,
        downloaded=len(video_paths),
        concurrency=effective_download_concurrency,
    )
    logger.success(f"downloaded {len(video_paths)} videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_by_script_order(
    task_id: str,
    search_terms: List[str],
    search_videos,
    provider: str,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """
    按脚本文案顺序下载素材。

    默认下载逻辑会把所有关键词的候选素材合并成一个大列表；如果第一个
    关键词返回很多结果，最终下载时可能一直消耗这个关键词的素材，后续
    脚本主题就排不上时间线。这里按关键词分组后轮询下载：
    第 1 轮取每个关键词的第 1 个候选，第 2 轮取每个关键词的第 2 个候选。
    这样在不重写视频合成引擎的前提下，尽量保证素材顺序贴近文案顺序。
    """
    logger.info("downloading videos with script-order material matching")
    search_started_at = time.perf_counter()
    candidate_groups = []
    valid_video_urls = set()
    found_duration = 0.0

    for search_term in search_terms:
        video_items = search_videos(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            video_aspect=video_aspect,
        )
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        term_items = []
        for item in video_items:
            if item.url in valid_video_urls:
                continue
            term_items.append(item)
            valid_video_urls.add(item.url)
            found_duration += item.duration

        if term_items:
            candidate_groups.append((search_term, term_items))

    logger.info(
        f"found total ordered video candidates: {sum(len(items) for _, items in candidate_groups)}, "
        f"required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    utils.log_runtime_benchmark(
        "material_search",
        search_started_at,
        source=provider,
        terms=len(search_terms),
        candidates=sum(len(items) for _, items in candidate_groups),
        ordered=True,
    )

    download_started_at = time.perf_counter()
    video_paths = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    candidate_index = 0
    while candidate_groups and total_duration <= audio_duration:
        has_candidate = False
        for search_term, term_items in candidate_groups:
            if candidate_index >= len(term_items):
                continue

            has_candidate = True
            item = term_items[candidate_index]
            try:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                logger.info(
                    f"downloading ordered {item.provider} video for {search_term!r}: "
                    f"asset_id={source_info.get('asset_id') or 'unknown'}"
                )
                saved_video_path = save_video(
                    video_url=item.url, save_dir=material_directory
                )
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        logger.warning(
                            "failed to prepare ordered material source record: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, "
                            f"detail={source_error}"
                        )
                    total_duration += min(max_clip_duration, item.duration)
                    if total_duration > audio_duration:
                        logger.info(
                            f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                        )
                        break
            except Exception as e:
                logger.error(
                    "failed to download ordered material video: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )

        if not has_candidate:
            break
        candidate_index += 1

    logger.success(f"downloaded {len(video_paths)} ordered videos")
    utils.log_runtime_benchmark(
        "material_download",
        download_started_at,
        source=provider,
        downloaded=len(video_paths),
        concurrency=1,
        ordered=True,
    )
    _persist_material_sources(task_id, material_sources)
    return video_paths


# =============================================================================
# Article Mode: image providers and licensed-asset acquisition
#
# These adapters search stock-photo APIs (Pexels, Pixabay) and return typed
# ``MediaAsset`` records carrying license/attribution/provenance so that an
# images-only or mixed-media article video can be rendered with traceable
# rights. Search-engine image scraping is intentionally NOT supported. An
# article's own lead image is never auto-reused unless its license is known.
# =============================================================================

_PEXELS_LICENSE = ("Pexels License", "https://www.pexels.com/license/")
_PIXABAY_LICENSE = ("Pixabay Content License", "https://pixabay.com/service/license-summary/")
_COVERR_LICENSE = ("Coverr License", "https://coverr.co/license")
_IMAGE_MAGIC_BYTES = (
    b"\xff\xd8\xff",  # JPEG
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"GIF87a",
    b"GIF89a",
    b"BM",  # BMP
)
_MIN_IMAGE_DIMENSION = 400
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MIN_SCENE_ASSET_RELEVANCE = 0.4
_RELEVANCE_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "image",
    "images",
    "photo",
    "photos",
    "pexels",
    "pixabay",
    "the",
    "video",
    "videos",
    "with",
}


def _orientation_for(aspect: VideoAspect) -> str:
    if aspect == VideoAspect.portrait:
        return "portrait"
    if aspect == VideoAspect.landscape:
        return "landscape"
    return "square"


def _semantic_text_from_url(value: Any) -> str:
    try:
        parsed = urlsplit(str(value or ""))
    except ValueError:
        return ""
    path_text = re.sub(r"[^A-Za-z0-9]+", " ", parsed.path)
    return path_text.strip()


def _coerce_metadata_text(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            values.extend(_coerce_metadata_text(item) for item in part)
        elif isinstance(part, dict):
            values.extend(_coerce_metadata_text(value) for value in part.values())
        else:
            text = str(part).strip()
            if text:
                values.append(text)
    return " ".join(value for value in values if value).strip()[:1000]


def search_images_pexels(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[MediaAsset]:
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("pexels_api_keys")
    headers = {"Authorization": api_key, "User-Agent": "Influencer-Automation-2.0/ArticleMode"}
    params = {
        "query": search_term,
        "per_page": per_page,
        "orientation": _orientation_for(aspect),
    }
    query_url = f"https://api.pexels.com/v1/search?{urlencode(params)}"
    logger.info(f"searching images on pexels: term={search_term!r}")
    try:
        r = requests.get(
            query_url, headers=headers, proxies=config.proxy,
            verify=_get_tls_verify(), timeout=(30, 60),
        )
        response = r.json()
        assets: List[MediaAsset] = []
        for photo in response.get("photos", []) or []:
            src = photo.get("src") or {}
            download_url = src.get("large2x") or src.get("large") or src.get("original")
            if not download_url:
                continue
            source_page = safe_public_page(photo.get("url"))
            assets.append(
                MediaAsset(
                    media_type="image",
                    provider="pexels",
                    url=download_url,
                    width=int(photo.get("width") or 0),
                    height=int(photo.get("height") or 0),
                    asset_id=str(photo.get("id") or ""),
                    creator=str(photo.get("photographer") or ""),
                    metadata_text=_coerce_metadata_text(
                        photo.get("alt"),
                        _semantic_text_from_url(source_page),
                    ),
                    license_name=_PEXELS_LICENSE[0],
                    license_url=_PEXELS_LICENSE[1],
                    attribution_text=(
                        f"Photo by {photo.get('photographer', 'Pexels')} on Pexels"
                    ),
                    source_page_url=source_page,
                    search_query=search_term,
                )
            )
        return assets
    except Exception as e:
        logger.error(
            "pexels image search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )
    return []


def search_images_pixabay(
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[MediaAsset]:
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("pixabay_api_keys")
    orientation = "vertical" if aspect == VideoAspect.portrait else (
        "horizontal" if aspect == VideoAspect.landscape else "all"
    )
    params = {
        "key": api_key,
        "q": search_term,
        "image_type": "photo",
        "orientation": orientation,
        "per_page": max(3, per_page),
        "safesearch": "true",
    }
    query_url = f"https://pixabay.com/api/?{urlencode(params)}"
    logger.info(f"searching images on pixabay: term={search_term!r}")
    try:
        r = requests.get(
            query_url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(30, 60)
        )
        if _is_cloudflare_challenge(r):
            logger.error("pixabay image search was blocked by a Cloudflare challenge")
            return []
        response = r.json()
        assets: List[MediaAsset] = []
        for hit in response.get("hits", []) or []:
            download_url = hit.get("largeImageURL") or hit.get("webformatURL")
            if not download_url:
                continue
            source_page = safe_public_page(hit.get("pageURL"))
            assets.append(
                MediaAsset(
                    media_type="image",
                    provider="pixabay",
                    url=download_url,
                    width=int(hit.get("imageWidth") or 0),
                    height=int(hit.get("imageHeight") or 0),
                    asset_id=str(hit.get("id") or ""),
                    creator=str(hit.get("user") or ""),
                    metadata_text=_coerce_metadata_text(
                        hit.get("tags"),
                        _semantic_text_from_url(source_page),
                    ),
                    license_name=_PIXABAY_LICENSE[0],
                    license_url=_PIXABAY_LICENSE[1],
                    attribution_text=(
                        f"Image by {hit.get('user', 'Pixabay')} on Pixabay"
                    ),
                    source_page_url=source_page,
                    search_query=search_term,
                )
            )
        return assets
    except Exception as e:
        logger.error(
            "pixabay image search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )
    return []


def search_images(
    provider: str,
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    per_page: int = 20,
) -> List[MediaAsset]:
    """Dispatch to a configured image provider. Unknown providers return []."""
    provider = (provider or "pexels").strip().lower()
    if provider == "pixabay":
        return search_images_pixabay(search_term, video_aspect, per_page)
    if provider == "pexels":
        return search_images_pexels(search_term, video_aspect, per_page)
    if provider == "comfyui":
        # Generates on-brand frames locally instead of searching stock. See
        # brand_footage.py for why stock cannot be steered safely here.
        from app.services import brand_footage

        return brand_footage.search_images_comfyui(search_term, video_aspect, per_page)
    logger.warning(f"unsupported image provider: {provider}")
    return []


def _material_video_to_asset(item: MaterialInfo, search_term: str) -> MediaAsset:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    provider = str(item.provider or source.get("provider") or "").lower()
    rendition = source.get("rendition") if isinstance(source.get("rendition"), dict) else {}
    creator_info = _creator_info(source.get("creator")) or {}
    license_name, license_url = {
        "pixabay": _PIXABAY_LICENSE,
        "coverr": _COVERR_LICENSE,
    }.get(provider, _PEXELS_LICENSE)
    creator = creator_info.get("name", "")
    provider_label = provider.capitalize() if provider else "stock provider"
    return MediaAsset(
        media_type="video",
        provider=provider,
        url=item.url,
        width=int(rendition.get("width") or 0),
        height=int(rendition.get("height") or 0),
        duration=float(item.duration or 0),
        asset_id=str(source.get("asset_id") or ""),
        creator=creator,
        metadata_text=_coerce_metadata_text(
            source.get("metadata_text"),
            _semantic_text_from_url(source.get("source_page")),
        ),
        license_name=license_name,
        license_url=license_url,
        attribution_text=(
            f"Video by {creator} on {provider_label}" if creator else f"Video from {provider_label}"
        ),
        source_page_url=safe_public_page(source.get("source_page")),
        search_query=search_term,
    )


def search_video_assets(
    provider: str,
    search_term: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    minimum_duration: int = 1,
) -> List[MediaAsset]:
    provider = (provider or "pexels").strip().lower()
    searchers = {
        "pexels": search_videos_pexels,
        "pixabay": search_videos_pixabay,
        "coverr": search_videos_coverr,
    }
    searcher = searchers.get(provider)
    if searcher is None:
        logger.warning(f"unsupported video provider for article mode: {provider}")
        return []
    return [
        _material_video_to_asset(item, search_term)
        for item in searcher(search_term, minimum_duration, video_aspect)
    ]


def validate_image_file(image_path: str) -> tuple[int, int]:
    """Validate a downloaded image by magic bytes, size and dimensions.

    Returns ``(width, height)`` or raises ``ValueError``. This is a technical
    safety check (is this really a usable image?), not an editorial one."""
    if not os.path.isfile(image_path):
        raise ValueError("image file does not exist")
    size = os.path.getsize(image_path)
    if size == 0 or size > _MAX_IMAGE_BYTES:
        raise ValueError(f"image file size out of range: {size} bytes")
    with open(image_path, "rb") as fp:
        header = fp.read(16)
    is_webp = header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    if not is_webp and not any(header.startswith(magic) for magic in _IMAGE_MAGIC_BYTES):
        raise ValueError("file is not a supported image type")
    from PIL import Image as _Image

    with _Image.open(image_path) as im:
        im.verify()
    with _Image.open(image_path) as im:
        width, height = im.size
    if width < _MIN_IMAGE_DIMENSION or height < _MIN_IMAGE_DIMENSION:
        raise ValueError(f"image dimensions too small: {width}x{height}")
    return width, height


def save_image(image_url: str, save_dir: str = "") -> str:
    """Download and validate a remote image. Returns the local path or "".

    Locally generated footage (see brand_footage.py) arrives as a path that
    already exists, so it is validated and passed through rather than fetched.
    """
    if image_url and not image_url.startswith(("http://", "https://")) \
            and os.path.exists(image_url):
        try:
            validate_image_file(image_url)
            return image_url
        except ValueError as exc:
            logger.warning(f"generated frame failed validation: {exc}")
            return ""
    if not save_dir:
        save_dir = utils.storage_dir("cache_images", create=True)
    os.makedirs(save_dir, exist_ok=True)
    url_without_query = image_url.split("?")[0]
    ext = utils.parse_extension(url_without_query) or "jpg"
    if ext not in ("jpg", "jpeg", "png", "webp", "bmp", "gif"):
        ext = "jpg"
    image_path = os.path.join(save_dir, f"img-{utils.md5(url_without_query)}.{ext}")
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        try:
            validate_image_file(image_path)
            return image_path
        except ValueError:
            try:
                os.remove(image_path)
            except OSError:
                pass
    headers = {"User-Agent": "Mozilla/5.0 Influencer-Automation-2.0/ArticleMode"}
    try:
        with requests.get(
            image_url, headers=headers, proxies=config.proxy,
            verify=_get_tls_verify(), timeout=(30, 120), stream=True,
        ) as r:
            r.raise_for_status()
            downloaded = 0
            with open(image_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > _MAX_IMAGE_BYTES:
                        raise ValueError("image exceeds maximum size")
                    f.write(chunk)
        validate_image_file(image_path)
        return image_path
    except Exception as e:
        logger.warning(
            "failed to download image: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, image_url)}"
        )
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except OSError:
            pass
    return ""


def _relevance_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in _RELEVANCE_STOPWORDS
    }


def score_asset_relevance(
    asset: MediaAsset,
    query: str,
    entities: List[str] | None = None,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> float:
    """Deterministic relevance signal for a candidate asset (0..1).

    Combines query/metadata token overlap, orientation fit and resolution. This
    is a fast pre-filter; an optional multimodal scorer can be layered on top via
    the pipeline without changing this interface."""
    query_tokens = _relevance_tokens(query)
    meta = _coerce_metadata_text(
        getattr(asset, "metadata_text", ""),
        asset.source_page_url,
        asset.creator,
        asset.attribution_text,
    ).lower()
    meta_tokens = _relevance_tokens(meta)
    if query_tokens and meta_tokens:
        overlap = len(query_tokens & meta_tokens) / max(1, len(query_tokens))
    elif query_tokens:
        # Some providers return little or no semantic metadata for a valid
        # result. Keep those candidates possible, but do not treat them as
        # verified matches.
        overlap = 0.55
    else:
        overlap = 0.5
    entity_bonus = 0.0
    if entities:
        entity_text = meta
        entity_bonus = 0.1 * sum(1 for e in entities if e.lower() in entity_text)
    aspect = VideoAspect(video_aspect)
    orient_score = 0.5
    if asset.width and asset.height:
        ratio = asset.width / asset.height
        target = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0}[aspect.value]
        orient_score = max(0.0, 1.0 - min(1.0, abs(ratio - target) / target))
    resolution_score = min(1.0, (min(asset.width, asset.height) or 0) / 1080.0)
    score = 0.5 * overlap + 0.2 * orient_score + 0.2 * resolution_score + entity_bonus
    if query_tokens and meta_tokens and overlap <= 0 and entity_bonus <= 0:
        score = min(score, _MIN_SCENE_ASSET_RELEVANCE - 0.05)
    return round(max(0.0, min(1.0, score)), 4)


def _query_contains_entity(query: str, entities: List[str]) -> bool:
    query_text = f" {query.lower()} "
    for entity in entities:
        entity_text = str(entity or "").strip().lower()
        if entity_text and entity_text in query_text:
            return True
    return False


def _scene_search_queries(scene, entities: List[str] | None = None) -> list[str]:
    raw_queries = list(getattr(scene, "visual_queries", []) or [])
    if not raw_queries:
        raw_queries = [getattr(scene, "narration", "")[:60]]

    cleaned: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        value = str(query or "").strip()
        key = value.lower()
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)

    if not entities:
        return cleaned

    contextual: list[str] = []
    for query in cleaned:
        if not _query_contains_entity(query, entities):
            for entity in entities[:2]:
                entity_text = str(entity or "").strip()
                if not entity_text:
                    continue
                value = f"{entity_text} {query}".strip()
                key = value.lower()
                if key not in seen:
                    contextual.append(value)
                    seen.add(key)
        contextual.append(query)
    return contextual[:8]


def select_scene_assets(
    provider: str,
    scenes,
    video_aspect: VideoAspect = VideoAspect.portrait,
    entities: List[str] | None = None,
    media_mode: str = "images_only",
    searcher=None,
    minimum_relevance: float = _MIN_SCENE_ASSET_RELEVANCE,
) -> List[MediaAsset]:
    """Pick one best asset per scene, in scene order.

    For each scene we try its ``visual_queries`` in order, score candidates,
    and keep the highest-scoring non-duplicate. Continues with the best
    available visual rather than blocking when results are thin. ``searcher`` is
    injectable for tests."""
    def _default_search(query: str) -> List[MediaAsset]:
        mode = str(media_mode or "images_only")
        assets: List[MediaAsset] = []
        if mode in {"images_only", "mixed"}:
            assets.extend(search_images(provider, query, video_aspect))
        if mode in {"videos_only", "mixed"}:
            assets.extend(search_video_assets(provider, query, video_aspect))
        return assets

    search = searcher or _default_search
    chosen: List[MediaAsset] = []
    used_asset_keys: set = set()
    for beat_index, scene in enumerate(scenes):
        queries = _scene_search_queries(scene, entities)
        best: MediaAsset | None = None
        best_score = -1.0
        for query in queries:
            for asset in search(query) or []:
                key = (
                    f"{asset.provider}:{asset.asset_id}"
                    if asset.provider and asset.asset_id
                    else asset.url
                )
                if key in used_asset_keys:
                    continue
                asset.search_query = query
                asset.relevance_score = score_asset_relevance(
                    asset, query, entities, video_aspect
                )
                if asset.relevance_score > best_score:
                    best, best_score = asset, asset.relevance_score
            if best is not None and best_score >= 0.6:
                break  # good enough; don't burn extra queries
        if best is not None and best_score >= minimum_relevance:
            best.beat_index = beat_index
            best.illustrative = bool(getattr(scene, "is_contextual_visual", True))
            best.selection_reason = (
                f"best of query {best.search_query!r} "
                f"(relevance {best.relevance_score:.2f})"
            )
            used_key = (
                f"{best.provider}:{best.asset_id}"
                if best.provider and best.asset_id
                else best.url
            )
            if used_key:
                used_asset_keys.add(used_key)
            chosen.append(best)
    return chosen


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )
