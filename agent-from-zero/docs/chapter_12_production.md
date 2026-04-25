# Chapter 12：生产化与最佳实践

## 本章目标

理解 Demo 到产品之间的差距。

## 本章最终效果

可以通过 Docker Compose 启动 API：

```bash
docker compose up --build
```

## 前置知识

了解 Docker、环境变量和 Web 服务部署。

## 核心概念

生产 Agent 需要安全、稳定、可观测、可评估。不要让模型直接执行高风险动作；不要在日志中记录敏感信息；不要把用户数据混在一个全局向量库里。

## 项目文件

- `Dockerfile`
- `docker-compose.yml`
- `docs/production_checklist.md`

## 代码实现

Docker 镜像安装项目依赖并启动 FastAPI。真实生产环境应拆分 API、任务队列、向量库和数据库。

## 运行方式

```bash
docker compose up --build
```

访问：

```text
http://127.0.0.1:8000/docs
```

## 常见错误

- 容器中没有 API Key：用 `.env` 或平台 Secret 注入。
- 数据丢失：把 `data/` 挂载成 volume。
- 成本失控：设置并发限制、缓存和评估抽样率。

## 扩展练习

把内存 checkpoint 替换成 PostgreSQL checkpoint。

## 本章小结

生产化不是换个部署方式，而是把安全边界、数据隔离、观测和评估都纳入工程流程。
