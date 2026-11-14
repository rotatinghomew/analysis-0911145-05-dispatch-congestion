# 产品数据与调用约定

时间单位为班次起点后的分钟。jobs 中 ready 为可作业时间，scans 可缺 start 或 end，绝不能据此假设服务为零。resource_calendar 的 capacity 是半开时间区间内可用人数，loans 从同一资源池调拨，不能两头增加容量。departures 的 released_jobs 是整车共同发运关系，target_units 是放行条件而非实测满载量。POST /capacity {case, scenario?} 返回 {snapshot, fit, validation, scenarios, diagnostics}。fit 说明参数及其辨识范围，validation 使用 holdout_shifts；scenarios 报告积压和时间尾部定义及不确定性，不要求特定随机过程模型。

当前 GET /cases、POST /preview 和 GET /health 已可用；client.analysis 是待接入的业务消费调用，新分析路由尚不存在。原始数据通过 app.store 读取，snapshot 为输入字节的 SHA-256。新增接口须保留现有预览能力和 JSON 有限数值约定。

fit.training_shifts 记录参与校准的班次，validation 是含 shift 字段的记录列表，必须覆盖每个 holdout_shifts。拟合失败也应保留留出记录并标记原因。借调期间可暂停已开始作业，扫描起止是墙钟跨度，不直接等同连续占工时。
