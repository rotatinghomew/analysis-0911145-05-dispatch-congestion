# 产品数据与调用约定

cells 的 cohort 是自 origin 起的入库周，age 是库龄周，period 是经营日历周，满足
cohort+age=period；offered 为该等级/时间单元待售箱数，discounted 为其中折价箱数，
同一箱在不同时点可再次待售，不能据箱次假设独立果实。known_at 决定 as_of_period 时
是否可见。external_anchors 是独立资料给出的代次 log-odds 差异区间，不得当精确真值；
heldout 为完整代次留出。

新接口 POST /cohort-effects 接收 {case, anchors?}；`anchors` 省略时使用资料自带锚点，
显式传入空列表时表示不使用外部锚点。返回
{snapshot, estimable, constraints, sensitivity, validation, diagnostics, conclusions,
scopes}。estimable 中注明对比、估计和不确定性含义；constraints 区分规范化与实质假设；
不要求给出唯一三方趋势分摊。

`estimable` 只包含可估的两两方向对比（within_cohort、cross_cohort_same_age、
within_period_age），每项给出估计、标准误、区间、拟二项离散度和聚类标准误。三方线性
趋势的单独分摊不出现在 estimable 中，只在 sensitivity 的规范化/外部情景里以
`data_supported: false` 出现。

`constraints` 是记录列表且每条含 kind，值为 normalization、external 或 substantive；
external 只能引用实际提供且可用的锚点。锚点不可用（代次对不相邻、相同或区间无效）
时接口返回 400，不静默忽略。

`sensitivity` 给出：
- `scenarios`：C=0、P=0、A=0 三种规范化与锚点取值下的 (A, C, P) 分摊；
- `flips`：同一趋势仅因规范化或锚点不同而改变符号的记录；
- `anchor_checks`：锚点与同口径原始代次对差异的比较，说明收窄了什么；
- `anchor_narrowing`：加入锚点前后真正收窄的可比结论，以及没有收窄的估计量。

`validation` 使用完整且可见的留出代次比较线性与曲率设定，并做逐代次轮换；部分标记
留出的代次不用于验证，以免同代次信息泄漏。

`diagnostics` 至少说明：三趋势秩亏与零空间方向、等级是否嵌套于代次、曲率在哪些范围
不可识别、晚补录单元被排除的数量、外部锚点缺失或不可用的原因。无法判断的量用明确
状态与 null 表达，不返回 NaN。

现有 GET /health、GET /cases 和 POST /preview 保留原有资料预览。GET /study-cases、
POST /study-preview {case} 只校验与描述 data/study 的资料；POST /cohort-effects 使用
同一份 study 资料做推断。所有 snapshot 都是相应原始资料字节的 SHA-256，输出仅使用
有限 JSON 数值。资料只供本地分析，不连接外部服务。
