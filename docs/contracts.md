# 产品数据与调用约定

cells 的 cohort 是自 origin 起的入库周，age 是库龄周，period 是经营日历周，满足 cohort+age=period；offered 为该等级/时间单元待售箱数，discounted 为其中折价箱数，同一箱在不同时点可再次待售，不能据箱次假设独立果实。known_at 决定 as_of_period 时是否可见。external_anchors 是独立资料给出的代次 log-odds 差异区间，不得当精确真值；heldout 为完整代次留出。新接口 POST /cohort-effects 接收 {case, anchors?}，返回 {snapshot, estimable, constraints, sensitivity, validation, diagnostics}。estimable 中注明对比、估计和不确定性含义；constraints 区分规范化与实质假设；不要求给出唯一三方趋势分摊。

现有 GET /health、GET /cases 和 POST /preview 保留原有资料预览。新增 GET /study-cases 和 POST /study-preview {case} 已可用，只校验与描述 data/study 的资料；推断消费路由尚未实现。app.client.analysis 指向上文约定的新消费接口。所有 snapshot 都是相应原始资料字节的 SHA-256，输出仅使用有限 JSON 数值，无法判断用明确状态与 null 表达，不得返回 NaN。资料只供本地分析，不连接外部服务。

estimable 是可估对比记录列表，constraints 是记录列表且每条含 kind，值为 normalization、external 或 substantive；external 只能引用实际提供的锚点。
