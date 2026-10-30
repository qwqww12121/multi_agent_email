# Llama 3.1 Base 四角色提示与生成约束：v4r4

本版遵守 2026-10-29 本聊天确认：**保留 Llama‑3.1‑8B Base，只修改提示与输出约束。**
本版已于北京时间 2026-10-29 15:34 完成新服务器上的真实 Base 小测试：四库各取冻结顺序第一封，共 4 封、16 份角色报告。R3 的源文件与原始结果保持独立。

## 2026-10-29 真实 Base 小测试结果

本轮用户授权连接新任务 a21678764524040192134980。使用同一 Base 文件哈希、同一冻结输入、1536 新 token 上限和生成时结构约束，未做校准、PPO 或 RAG。16 份报告全部保存，测试正常结束，进程已退出；未扩大到 2000 封。

- 可解析的完整 JSON：6/16；达到长度上限：10/16。
- 完整 Schema 校验通过：0/16；整套字段与原文引用校验通过：0/16。
- 四角色全部有效邮件：0/4。没有有效最终分类，不能把它写成分类准确率为零。
- 主测试循环耗时约 478.7 秒（8 分钟，不含安装、模型核对加载及短验收）。

具体失败：3 份行动报告输出空动作列表却没有符合协议的解释；3 份情境报告的 unknown/not_present 状态与非 null 值冲突。另有真实内容问题：ePhishLLM 原文请求查看附件并反馈，行动报告却为空；情境报告把字段定义照抄成值，并生成输入未提供的 `mike@example.com` 和 From 邮件头。PhishingPot 请求协助资金转移，行动角色也误称没有请求。已闭合的三份情境报告都存在来源字段中找不到的引用。

这证明报告起点和部分结构约束生效，但仍不能正常完成本任务。短样本只用于诊断，不能推断整体分类能力。完整原始记录在 `qa/server_probe_20261029/records.jsonl`，四封输入在 `inputs.jsonl`，最终快照在 `snapshot.json`，独立结构与引文复核在 `audit.json`。所有输出未经补齐、修复或重试。

运行目录：`/phishing_email/test_1/benchmark_native_20261014/four_text_v4r4_probe_20261029/runs/probe4_v4r4_20261029`。参考运行器的 progress 仍以全池 8000 个角色为分母，因而显示 partial；本次授权的 16 角色 probe 已全部完成，setup_status 为 probe_finished。

## 当前问题与证据

R3 本地快照时间为北京时间 2026-10-29 14:28，不表示服务器实时状态。快照有 14 条角色记录，全部达到 1536 token 上限，3 封邮件完成四角色调用，四角色全部有效数为 0。原始输出开头包括 `JSON Schema for the four-role model`，随后是 Schema 解释和重复文字。来源是相邻目录 `four_text_v4r3/qa/inspection_records.jsonl`、`inspection_snapshot.json` 和 `runtime.json`。

R3 的实际模型路径与权重清单对应 Base。报告提示是“职责、Schema、邮件”组成的完整 JSON 文档；字符串在文档闭合后结束，没有单独的回答起点。生成器直接以 `add_special_tokens=False` 编码，提示没有 Base 起始标记；生成时只在出现闭合对象后停止，没有对字段、类型和枚举做 token 过滤。

因此可以确认输入与输出控制存在具体问题。**将 Schema 文档续写视为“模型不懂垃圾邮件”的证据还不充分。** Base 续写训练、提示形态和缺少生成约束共同影响输出；各项的独立作用仍需对照验证。

## 官方说明和社区讨论怎样使用

- [Meta 官方 Llama 3.1 Prompt Formats](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md) 单独列出 Base 文本续写与 Instruct 对话格式。Base 使用 `<|begin_of_text|>` 开始文本；Instruct 使用消息角色头。把同一权重包装成 system/user 并不会新增指令微调能力。本版采用 Base 续写格式。
- [Meta 在 Hugging Face 的 Base 模型页](https://huggingface.co/meta-llama/Llama-3.1-8B) 与 [Instruct 模型页](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 分别展示 completion 与 chat 使用方式。它们是不同检查点，本版核对 R3 保存的模型及 tokenizer 文件哈希，拒绝换用另一检查点。
- [Hugging Face Transformers 4.57.1 模板说明](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating) 解释所有因果语言模型都在续写 token，模板需匹配训练格式，并说明回答前缀与重复特殊 token 的问题。本文据此设计明确的报告续写起点；这是一项本地设计，官方并未保证我们的零示例 Base 任务必然成功。
- [HF Llama‑3.1‑8B‑Instruct 讨论 #140](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/140) 涉及微调后的 Yes/No 输出和训练、推理格式一致性。它属于社区经验，模型及训练条件与本次不同。
- [Meta GitHub llama-models #179](https://github.com/meta-llama/llama-models/issues/179) 报告 Instruct 在 RAG 中额外续写问答；该讨论没有给出可验证的修复结论，不能据此认定本地根因。
- [Meta GitHub llama3 #154](https://github.com/meta-llama/llama3/issues/154) 讨论多余文本和 JSON；其中 `format="JSON"` 建议之后仍有无效反馈。这是 Llama 3 的旧讨论，不是本次 3.1 Base 的修复证明。模板参数与实际 token 约束须区分。
- [LM Format Enforcer 官方仓库](https://github.com/noamgat/lm-format-enforcer/tree/v0.11.3) 及其 [Transformers 接口](https://github.com/noamgat/lm-format-enforcer/blob/v0.11.3/lmformatenforcer/integrations/transformers.py) 提供逐 token 过滤。项目已采用 Transformers，因此本版直接接该接口，不迁移推理后端。

## 新提示怎样组织

输入顺序为：Base 起始标记 → 数据集任务含义 → 公共字段与证据要求 → 单个角色职责和字段定义 → 当前邮件 → 明确的报告起点。

报告起点仅包含程序已知的 `schema_version` 与 `role`，例如行动角色的固定结构前缀。它没有填写邮件事实、判断、证据或标签，不是 few-shot 示例。`raw_text` 单独保存实际生成的后缀，`prefill_text` 保存前缀，`text` 为二者拼接；不会把固定前缀计作模型生成内容。

四份可读提示模板位于 `prompts/action.txt`、`context.txt`、`persuasion.txt`、`purpose.txt`。模板中的占位符由实际渲染器替换成数据集定义、完整输入范围和当前邮件；不插入示范邮件。完整 JSON Schema 单独保存在旁边，供解码器及事后校验使用，不再作为待续写的主体文档。

四个角色分别回答：

1. **行动**：谁要求谁，对什么对象执行什么操作，结果给谁，有什么条件？区分当前、历史、引用、条件与否定行为。
2. **情境**：对方声称是谁，和收件人是什么关系，当前背景怎样支持请求？区分身份自称、可见佐证、具体矛盾及未决信息。
3. **说服**：哪段表达通过什么理由或后果推动什么行为？将表达与行为连起来，避免把普通期限、礼貌或利益直接等同于欺骗。
4. **目的**：邮件谈什么，声称为何联系，希望发生什么，照做会直接产生什么结果，涉及什么对象，由谁接收？区分明示目的、有依据的推断和未见后续事件。

保留 R3 的具名字段协议：`value/status/scope/evidence`。字段定义、原生标签映射和两种标签定义顺序下的候选对数似然计算保留。标签评分阶段使用独立提示，不要求再生成报告；本版为评分提示也加入 Base 起始标记，因此评分输入和运行签名发生变化，不能复用旧分数或训练产物。

## 哪些约束已实现，哪些仍需检查

生成时调用 `prefix_allowed_tokens_fn`，为每一批中的每个角色建立独立状态，约束必填键、额外键、对象/数组/标量类型及枚举。按固定字段顺序先生成主要关系，避免先花大量输出在附属元数据上。LMFE 固定为 0.11.3；缺少依赖或约束出错时失败，不自动退回自由生成。

解码 Schema 是显式展开的结构子集。原 Schema 的 `if/then` 跨字段关系仍由完整 JSON Schema 在生成后检查，包括 observed/inferred 的证据要求，以及空动作列表的解释要求。原文引文匹配继续由现有校验器处理。**不能声称硬解码已经保证全部 Schema 条件，更不能声称 JSON 合法就代表内容正确。**

输出可能仍然被截断、全部填 unknown、引文真实但不支持判断，或遗漏关键关系。`content_status=empty_analysis` 单独标记没有任何 supported claim 的报告；它不被报告成“有内容的分析”。分类准入和分母没有暗中放宽。自动引文检查只证明原文出现过该文字，语义支持需人工核对。

## 实验边界与验证状态

本版仍采用同一 Base、贪心生成、1536 个新 token 上限、18432 上下文预算、最多 8 请求批量及 30000 批次 token 预算。完整具名报告可能仍超过输出预算，达到上限会保留为失败，不补齐或悄悄提高预算。输出提示中的短引文与简洁表述是内容要求，不等于强制长度保障。

这是四角色诊断分支，沿用另一个聊天明确指定的“不做校准、不做 PPO、不接 RAG”的本轮设置；历史六列实验不被本版覆盖。`reference_runner.py` 可复用 R3 的四库各 500 封冻结输入和标签，逐文件检查哈希；不会重新按结果挑样本。

已进行离线结构与传输测试：使用真实 LMFE 解析器，以及真实 Transformers `generate()` 配合人工构造的测试 logits 检查 token 过滤、四角色批处理、逐行停止与原始后缀保存。测试用的小模型和人工输出只是程序测试夹具，**不是 Llama 实测结果，也不进入正式提示**。其他测试覆盖原生候选标签衔接、非法结构拒绝、引文校验与截断失败。

真实 Llama 的首次小测试结果已记录在本文开头：16 份报告没有一份通过全部校验。更大范围分类效果尚未测量。依赖加入了 CPU 侧逐 token 工作，不能沿用 R3 的吞吐估计。

建议首次真实验证仅使用冻结顺序下每库前 1 封（4 封、16 角色）检查格式、引文、空报告与角色内容，再按预先固定的范围继续。此批是已经查看过的诊断数据，不是新的盲测集；不按准确率门槛选择提示版本。通过格式检查后，仍要核对“字段是否抓住这封邮件的对应关系”。

准备新运行目录：`reference_runner.py reuse-sample --previous <R3_RUN> --out <NEW_R4_RUN>`；首批运行：`reference_runner.py run --out <NEW_R4_RUN> --limit-mails 1`。先在独立环境安装本目录的 `[model]` 依赖。本次已在新实例完成上述小测试，未执行后续全量运行。

父版本文件清单在 `qa/parent_manifest.json`；本版源码、依赖版本、模型哈希和协议均进入运行签名。旧结果不能冒充 R4 结果，今后如改变预算、字段协议或样本范围，需建立新的签名与说明。
