# 运行与验证

`python -m unittest discover -s tests -v` 覆盖：

- 原有业务预览、资料校验与真实 HTTP 传输；
- `tests/test_cohort.py` 对联合分析的可核对计算，包括：
  - 三列线性趋势秩亏，零空间方向为 age + cohort − period；
  - 可估对比在库龄/代次整体平移后不变；
  - 每种规范化都保持 A + P 与 C + P 等于对应的可估对比；
  - 自造单调库龄小例的 log-odds 对比与解析值一致；
  - 晚补录与留出单元的可见口径、完整代次留出与部分留出排除；
  - 锚点覆盖与校验、锚点只收窄同口径原始差异；
  - 等级分范围、等级嵌套与交互检验；
  - 输出不含 NaN/Infinity，无法判断的量用 null。

`RUN_ACCEPTANCE=1 python -m unittest tests.test_acceptance -v` 验证新分析消费接口的
基本字段、快照来源和信息不足契约。该测试不是算法正确性的证明，算法正确性由
`tests/test_cohort.py` 的可核对计算、敏感性与信息截止回归补充。

`python -m app.demo` 临时启动服务，通过消费端真实请求旧业务预览、资料描述预览和
`/cohort-effects`，并打印输入口径、可估对比、约束、敏感性、验证与诊断。演示不按
资料名返回静态结论；正式展示不安装依赖，资料规模允许在 90 秒内结束。
