# 库龄入库代次与日历周混淆下的折价来源识别

本地分析工作台保留已有业务资料与预览能力，并提供独立分析资料的校验、描述预览和
`POST /cohort-effects` 联合分析。`data/study` 保存本地合成资料，`app/study.py`
解释资料结构，`app/cohort.py` 实现库龄、入库代次与日历周的联合分析。

数据均为本地合成，不含真实姓名、电话号码或生产系统凭据。`standard` 是常规资料，
`edge` 用于检查信息不足或约束冲突。它们只描述观测，不预设分析结论。

## 分析口径

- 分析对象是每个时间单元的 `discounted / offered`，即折价箱数占待售箱数的比例；
  不使用各批总销售额。同一箱可在不同时点再次待售，因此箱次不独立，标准误偏乐观。
- `cohort + age = period` 是确定关系。三列线性趋势因此秩亏，软件不会因为某个
  正则化器给出唯一系数，就把库龄、代次、日历三个趋势说成分别被数据识别。
- 真正可估的是三条两两组合对比：
  - `within_cohort`：A + P，同一代次内库龄增加一周；
  - `cross_cohort_same_age`：C + P，同一库龄下较晚代次对较早代次；
  - `within_period_age`：A − C，同一日历周内较大库龄对较小库龄。
- 三方趋势拆分只在明确标注的规范化（C=0、P=0、A=0）或外部锚点下给出，并单列
  会随假设翻转符号的结论。外部锚点只收窄与它同口径的原始代次对差异，不因此
  识别方向对比或三方分摊。
- 不同等级按原经营范围分等级拟合；合并口径只加等级截距并检验等级交互。等级嵌套在
  入库代次中，等级可分离性依赖代次趋势为线性的假设。
- `known_at > as_of_period` 的折价按原有可见口径排除；`heldout` 代次不进入拟合，
  验证只使用完整且可见的留出代次，并另做逐代次轮换。
- 曲率先在样本内检验，再在完整代次留出中复核；只有两者都改善才称为数据支持。

## 本地使用

Python 3.11，依赖版本固定在 `requirements.txt`。NumPy/SciPy 是分析工作台的数值工具；
预览逻辑主要使用标准库。准备阶段安装依赖，仓库不在启动或测试时下载任何资源。

支持 Python 3.11 系列的补丁版本；Docker 基础镜像通过 SHA256 固定为 Python 3.11.14，
依赖清单保持精确版本。相同依赖还需在非 root 的 Python 3.11.2 Linux 环境中验证，
应用不依赖特定补丁版本。

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m app.demo
python -m app.server
```

服务监听 `127.0.0.1:8080`，也可通过 `PORT` 调整。预览与分析请求由 `app/client.py`
发起；`app.demo` 在临时端口启动真实服务，通过实际 HTTP 消费调用打印旧业务预览、
资料描述预览和 `/cohort-effects` 分析摘要，不是录制日志的回放。服务接口和数据文件的
可见性约定不能互相替代。

## 可复现环境

```sh
docker build -t local/dispatch-congestion:baseline .
docker run --rm --network none local/dispatch-congestion:baseline python -m unittest discover -s tests -v
docker run --rm --network none local/dispatch-congestion:baseline
```

准备阶段允许构建镜像和预装依赖；正式演示不进行构建或安装。演示通过真实 HTTP 调用
展示输入口径、分析结果与限制，小样本在 90 秒内结束。功能测试启用方式：
`RUN_ACCEPTANCE=1 python -m unittest tests.test_acceptance -v`。
