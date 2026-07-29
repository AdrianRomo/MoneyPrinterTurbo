# 文章模式（新闻 → 视频）

文章模式可将新闻主题、RSS/Atom 订阅源或单篇文章 URL 自动生成短视频，自动化
程度可自由配置。它是**自动化优先**的：由 LLM 对每条新闻打分、生成有来源依据的
脚本、自动审校并在需要时重写、挑选有授权的图片/视频素材并渲染。人工审核是
**可选的**。

> ⚠️ **重要说明。** 文章模式**不保证内容真实**，也**不是专业事实核查**。生成的视频会
> 标注 *"Generated from available sources."*（根据现有来源生成）。请始终把输出视为
> “基于生成时可获得的报道的 AI 新闻摘要”。

原有的“主题 → 视频”流程完全不变。只有当 `content_mode` 为 `article_url` 或
`article_feed` 时才启用文章模式；默认 `topic` 行为与之前完全一致。

## 工作流程

```
拉取订阅 → 抽取并聚类文章 → AI 打分 → 阈值判定
→ 生成有来源依据的脚本 → AI 自动审校（+自动重写）
→ 视觉素材检索与挑选 → 渲染 →（可选）发布
```

确定性逻辑只用于保护软件与基础设施（URL/SSRF 校验、文件校验、数据库完整性、
密钥脱敏），**不作为事实核查的硬性门槛**。是否可信、最可能的说法、确定性高低，
这些由 LLM 以**置信度分数**表达，而非逐句证明。

### “有来源依据”指什么

脚本生成会把抽取到的正文、来源元数据和已知的不确定点交给模型，并要求它以此为
事实基础，**不得编造**人名/日期/数字/引语，来源冲突时自然表达不确定性。随后由 AI
审校器对照同样的来源检查脚本。因此输出是**可追溯到所列来源**的，而非“已被证明为真”。

### 为什么无法保证绝对真实

来源可能出错、不完整或相互矛盾；进行中的事件会变化；模型也可能误读。文章模式通过
打分、佐证信号与审校降低风险，但没有任何自动系统能认证真相。因此默认发布策略保守，
且每个视频都附带来源清单。

## 安装与启动

建议使用仓库锁文件安装完整运行与测试依赖：

```bash
uv sync --all-extras --dev --frozen
uv lock --check
```

文章模式必需依赖包括 `feedparser` 与 `trafilatura`；完整运行还需要 API/WebUI、
LLM、TTS、FFmpeg、Pexels/Pixabay 等已配置提供方的依赖与凭据。启动命令：

```bash
python main.py
uvicorn app.asgi:app --host 0.0.0.0 --port 8080
streamlit run webui/Main.py
python -m app.services.article_worker --once
```

建议在预发环境使用独立配置与存储目录，避免影响生产数据库：

```bash
IA2_CONFIG_FILE=/tmp/ia2-article-config.toml \
IA2_STORAGE_DIR=/tmp/ia2-article-storage \
python -m app.services.article_worker --once --automated
```

## 自动化模式

在 `config.toml` 中设置 `article_automation_mode`（也可按订阅覆盖）：

| 模式 | 拉取 | 打分 | 生成 | 渲染 | 发布 |
|------|:--:|:--:|:--:|:--:|:--:|
| **assisted 协助** | ✅ | ✅ | ✅ | ⏸ 需审批 | ⏸ |
| **automated 自动** | ✅ | ✅ | ✅ | ✅ | ⏸ 需审批 |
| **autonomous 全自主** | ✅ | ✅ | ✅ | ✅ | ✅（启用后） |

全自主模式仅在反复技术失败或 AI 评估风险极高时暂停。**自动发布默认关闭**，必须显式
设置 `article_auto_publish_enabled = true` 才会启用。

## 配置

所有键位于 `config.toml` 的 `[app]` 下，完整带注释的示例见 `config.example.toml`：
数据库路径、抓取超时/大小上限、默认新鲜度与轮询间隔、可信/屏蔽域名、图片提供方、
`article_voice_name`、媒体模式、各项自动化阈值、每日生成/发布上限、敏感类别等。

- **可信来源** `article_trusted_domains`：提升来源质量信号并识别权威一手来源（如
  官方 `.gov` 公告），是“偏好”而非硬门槛。
- **佐证**：多个独立域名会提升置信度；开启 `article_allow_single_source_stories`
  后，可信的单一来源（官方公告、新闻稿、政府发布、已核实声明、权威专业媒体）也可
  生成。转载**同一通讯社稿件**的多个 URL 只算**一个**来源。
- **图片授权**：仅使用有授权的图库（Pexels、Pixabay），不抓取搜索引擎图片；未知
  授权的文章配图不会自动复用。素材的授权名称、授权链接、署名与公开页面地址都会写入
  `media_manifest.json`；签名下载链接与 API Key 绝不写入产物或日志。
- **Worker 语音**：无人值守渲染使用 `article_voice_name`。生产环境应设置真实 TTS
  语音；若为空且 WebUI 未保存默认语音，则回退到 `no-voice`，只生成本地静音计时音频，
  适合离线验证，不适合正式发布。
- **敏感话题**：`article_sensitive_categories` 中的类别会更保守处理，默认要求人工
  审批后才可发布。

## WebUI

打开 Streamlit 后进入 **Article Mode** 标签页。界面通过 Article Mode 服务/API 层
操作，不会在 Streamlit 中启动无限 worker 循环。支持订阅列表/新增/编辑/启停/确认删除、
RSS 源、可信/屏蔽域名、新鲜度、轮询间隔、阈值、立即轮询；候选文章和聚类过滤、展开来源、
查看分数/置信度/视觉分/风险/不确定点、重新评估、生成与重试；脚本查看与编辑、自动审校、
场景视觉查询、素材预览和替换、图片/视频/混合模式、画幅选择、渲染、预览、下载与发布。
界面会说明置信度是 AI 评估，不是真实性保证；开启自动发布时会显示醒目警告。

## RSS 订阅示例

```bash
curl -X POST http://127.0.0.1:8080/api/v1/article-subscriptions \
  -H 'Content-Type: application/json' \
  -d '{"name":"AI 头条","language":"zh","rss_urls":["https://www.theverge.com/rss/index.xml"],"poll_interval_minutes":60}'
```

## 轮询 Worker（独立于 Streamlit）

```bash
python -m app.services.article_worker            # 按配置间隔循环
python -m app.services.article_worker --once     # 单次拉取+处理后退出
python -m app.services.article_worker --autonomous
python -m app.services.article_worker --subscription sub-xxxx
python -m app.services.article_worker --interval 900
```

Docker Compose 可增加一个共享同一镜像、配置与 `storage` 卷的 worker 服务；SQLite
WAL 模式保证 API 读取与 worker 写入在共享卷上安全并发。

使用 Ctrl-C 或服务管理器正常停止 worker。循环会按任务记录失败状态，并记录
`article worker interrupted; shutting down` 后退出。

## API 与 CLI

API 端点见英文文档（订阅增删改查、`/poll`、`/articles`、`/assess`、`/generate`）。
CLI 新增（原命令不变）：

```bash
python cli.py --article-url "https://example.com/news/story" --media-mode images_only --video-aspect 9:16
python cli.py --approve-article article-123 --media-mode mixed
python cli.py --poll-subscription sub-xxxx
python cli.py --list-articles
```

## 可复现验证

运行本地文章模式发布候选 smoke：

```bash
python scripts/validate_article_mode.py
```

该命令会在 `/tmp` 下创建临时配置、存储和 SQLite 数据库，禁用发布，并通过真实生产
管线的可注入边界使用本地 fixture。它会验证：单篇 URL 抽取、RSS worker 轮询、第二次
轮询去重、AI 评估与审校/重写、`no-voice` TTS、字幕、图片模式渲染、混合媒体渲染、
MP4 可播放、视频时长覆盖旁白、素材顺序、溯源产物、密钥/签名 URL 脱敏。失败时返回
非零退出码，并打印 task、产物目录、MP4 路径与时长。

这是**本地集成验证**，不是 live provider 验证。要验证真实提供方，需要配置 LLM、TTS、
Pexels/Pixabay 等凭据，并在关闭发布的前提下跑公开 URL、订阅轮询与渲染流程。

## 审核与发布

- **assisted**：准备候选与脚本，人工审核后再渲染/发布。
- **automated**：自动渲染，人工审核后发布。
- **autonomous**：自动渲染并发布（需开启 `article_auto_publish_enabled`）。

当新闻属于敏感类别（除非显式允许）或 AI 评估风险超过
`article_maximum_risk_for_auto_publish` 时，视频**绝不会**自动发布。每个任务都会保存
可事后查看的溯源文件：`article.json`、`sources.json`、`script_plan.json`、
`media_manifest.json`、`assessment.json`、`review.json`、`provenance.json`。

## 故障排查

- 轮询 0 篇文章：检查 feed 可达性、屏蔽域名、去重，以及 poll run 的 `errors`。
- URL 被拒绝：非 `http(s)`、本机/内网/link-local 地址或重定向到内网都会被 SSRF 校验拒绝。
- 故事总被跳过：检查 LLM 配置和 `article_minimum_*` 阈值。
- 没有素材或黑底：图库凭据缺失或搜索无结果；管线会降级到背景视频以保证渲染完成。
- TTS 失败：设置 `article_voice_name` 并检查对应提供方凭据与网络；离线验证可用 `no-voice`。
- FFmpeg 失败：确保 FFmpeg 在 `PATH` 中，或设置 `IMAGEIO_FFMPEG_EXE`。
- 无法发布：默认行为；必须启用 `article_auto_publish_enabled` 并使用 `autonomous`，敏感/高风险仍需人工审批。

数据库默认位于 `storage/article/articles.db`，除非设置了 `article_database_path`。备份或重置前先停止 worker：

```bash
cp storage/article/articles.db storage/article/articles.db.bak
rm storage/article/articles.db storage/article/articles.db-wal storage/article/articles.db-shm
```

生产数据库重置前必须先确认备份可用。更完整的说明请见
[`article-mode-en.md`](./article-mode-en.md)。
