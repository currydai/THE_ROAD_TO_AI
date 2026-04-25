# Production Checklist

- API Key 通过 Secret 管理，不写入代码和日志。
- 文件上传限制类型、大小和存储目录。
- 高风险工具加入 human-in-the-loop。
- 数据按用户、租户或项目隔离。
- 向量库支持重建、备份和版本管理。
- Prompt 有版本号和变更记录。
- LangSmith tracing 在开发和灰度环境开启。
- Evaluation 数据集覆盖正常、边界和失败样本。
- API 有鉴权、限流和请求超时。
- 长任务进入队列，不阻塞 Web 请求。
- 生产 checkpoint 使用数据库。
- Docker 镜像固定 Python 和依赖版本。
