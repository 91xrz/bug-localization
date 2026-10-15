# Elasticsearch 安装与启停

本项目固定使用 Windows x86_64 Elasticsearch 9.5.5。BM25 和向量检索均由同一 ES 服务执行，无需单独安装 BM25。本机已有安装时直接进入“启停”。

## 首次准备

请自行下载 [官方 Windows ZIP](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.5.5-windows-x86_64.zip)，然后在项目根目录运行（将路径替换为实际下载位置）：

```powershell
.\.venv\Scripts\python.exe -B scripts\elasticsearch\setup_elasticsearch.py --archive "C:\Users\2428\Downloads\elasticsearch-9.5.5-windows-x86_64.zip"
```

脚本不会下载软件；它根据 `configs/elasticsearch/distribution.json` 中的 SHA512 校验 ZIP，解压至 `.runtime/elasticsearch/elasticsearch-9.5.5/`，生成独立配置并记录安装状态。发行包自带 JDK，无需另装 Java。安装记录已存在时，不带 `--archive` 可重新生成配置。

## 启停

```powershell
.\scripts\elasticsearch\start_elasticsearch.ps1
```

脚本在前台启动服务，保持终端打开。另开终端检查响应或运行检索：

```powershell
Invoke-RestMethod http://127.0.0.1:9200
```

预期集群名称为 `bug-localization-local`，版本为 `9.5.5`。停止服务可使用启动终端的 Ctrl+C，或运行：

```powershell
.\scripts\elasticsearch\stop_elasticsearch.ps1
```

停止脚本会核对本项目安装路径和 Java 进程。若 9200 端口被其他服务占用，请先检查占用，勿直接结束不明进程。

## 本地文件

| 路径 | 用途 |
| --- | --- |
| `configs/elasticsearch/` | 可提交的发行版本、校验和、服务配置模板 |
| `.runtime/elasticsearch/elasticsearch-9.5.5/` | 已解压程序和自带 JDK |
| `.runtime/elasticsearch/config/` | 生成的实际配置，堆内存固定 1 GiB |
| `.runtime/elasticsearch/data/` | ES 索引与节点数据 |
| `results/elasticsearch/installation.json` | 安装验证记录 |
| `results/elasticsearch/server/` | 服务日志 |

本地配置采用单节点、关闭认证，仅监听 `127.0.0.1:9200`，用于本机实验。程序、索引和日志均被 Git 忽略。清理旧实验输出不会清除 ES 索引；删除 `data/` 会失去索引，不属于常规文件清理。
