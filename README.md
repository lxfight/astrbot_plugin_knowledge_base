---
<h1 align="center">🚀 AstrBot 知识库插件 🚀</h1>

<p align="center">
<strong>为你的 AstrBot 注入专属知识，打造更智能的对话体验！</strong>
</p>


<p align="center">
<code>astrbot_plugin_knowledge_base</code> 是一个为 <a href="https://github.com/Soulter/AstrBot">AstrBot</a> 聊天机器人框架量身定制的强大知识库插件。
它允许您的机器人连接到自定义的知识源，通过先进的检索增强生成 (RAG) 技术，
让机器人能够基于准确、具体的私有信息进行对话，而不仅仅依赖预训练模型的通用知识。
</p>
---

本项目目前处于公开测试期，期间可能会经历版本的快速迭代。我们欢迎您的试用、反馈与贡献！

感谢每一位关注者和贡献者，您的 Star 和 Fork 是我们持续开发的最大动力！如果您觉得这个项目对您有所帮助，请不吝点亮那颗小星星 ✨。

---

### 🚨 警示柱：数据无价，操作需慎！ 🚨

**请务必注意：** 数据库中的数据是您的宝贵资产，任何删除操作都具有**不可逆转的破坏性**。在执行任何删除指令前，务必**三思、确认、再三确认！**

**特别提醒：跨插件数据风险！**
本插件具备直接操作**数据库表**的权限。这意味着它**可能意外删除甚至其他插件的数据**，例如：

- [`astrbot_plugin_mnemosyne`](https://github.com/lxfight/astrbot_plugin_mnemosyne) 的记忆数据表。
  （**使用 Milvus Server 作为数据库**）

### 💔 血的教训与沉痛提醒 💔

> **我们沉痛地提醒您，由于一次不慎的操作，社区成员 [@wuyan](https://github.com/wuyan1003) 曾经为此付出了巨大的代价：**
>
> **痛失了数千条宝贵的记忆数据！**
>
> **这是一次无法挽回的损失，也是我们永远的警钟。**
>
> **愿后人引以为戒，切勿重蹈覆辙！**

---

## 🚀 核心功能一览

<div align="center">

| 功能点                | 描述                                                       | 图标 |
| :-------------------- | :--------------------------------------------------------- | :--: |
| 🧠 **智能 RAG 集成**  | 自动检索知识增强 LLM 回答，更精准、更相关                  |  🎯  |
| 💾 **多向量数据库**   | 支持 Faiss, Milvus Lite, Milvus Server，满足不同规模需求   |  🗂️  |
| ✍️ **灵活内容导入**   | 通过指令添加文本、本地文件 (`.txt`, `.md`)                 |  📥  |
| 🗣️ **便捷指令交互**   | `/kb` 系列指令，覆盖创建、搜索、切换、管理等全流程         |  💬  |
| ⚙️ **高度可配置**     | 自定义 Embedding、文本处理、RAG 策略，深度定制您的知识引擎 |  🛠️  |
| 📦 **持久化存储**     | 知识数据与用户偏好安全存储，插件更新无忧                   |  🛡️  |
| 👥 **会话级知识库**   | 为不同用户或群组设定专属的默认知识库，实现个性化知识服务   |  🌐  |
| 📄 **文件下载与解析** | 直接从 URL 下载并解析文本文件，快速扩充知识                |  🔗  |

</div>

---

## 🤔 为何选择本插件？

- 🌟 **赋予机器人“记忆”与“专长”**：让您的机器人掌握特定领域知识，如产品手册、项目文档、常见问答库等。
- 🎯 **提升回答质量**：基于事实依据进行回答，有效减少大型语言模型的“幻觉”和不准确性。
- 💡 **个性化互动体验**：根据对话场景或用户群体，动态调用相应知识库，提供更贴心的服务。
- 🔐 **数据自主可控**：知识数据存储在您选择的环境（本地或私有服务器），保障数据安全与隐私。
- 🧩 **与 AstrBot 无缝集成**：专为 AstrBot 设计，安装配置简单，开箱即用。

---

## 🛠️ 安装与配置指南

### 1. 安装插件

将 `astrbot_plugin_knowledge_base` 文件夹放置到 AstrBot 的 `plugins` 目录下。

### 2. 安装依赖

        [新版本插件已经实现依赖的自动判断及安装]

- **Faiss**
- **Milvus Lite**:
- **Milvus Server**: 确保您有一个正在运行的 Milvus 服务实例。

💡 **提示**: 如果希望使用 `astrbot_plugin_embedding_adapter` 自动配置 Embedding，请确保该插件也已安装。

---

## 🎮 快速开始：指令

| 在新的会话中一定要使用：`/kb use [知识库名称]` 来激活知识库的使用，激活后`/kb`相关指令在不填写知识库名称的情况下会默认使用指定的知识库

通过向机器人发送以 `/kb` (或别名 `知识库`) 开头的指令来与知识库互动：

- `✨ /kb help` - 显示所有可用指令，您的入门向导。
- `➕ /kb add text "AstrBot真棒！" my_astr_facts` - 添加文本到 `my_astr_facts` 知识库。
- `📄 /kb add file https://example.com/faq.txt common_issues` - 从 URL 添加文件内容。
- `🔍 /kb search "如何安装插件？" my_astr_facts` - 在知识库中搜索答案。
- `🚀 /kb use project_alpha` - 将当前会话的默认知识库切换到 `project_alpha`。（在会话中一定要使用这个，否则将不会在提示词中嵌入知识库中的内容）
- `📋 /kb list` - 查看所有已创建的知识库。
- `🗑️ /kb delete old_stuff` - (管理员) 删除知识库 `old_stuff` (会有二次确认)。

---

## 💡 RAG 工作流程示意

当您向启用了知识库增强的 AstrBot 提问时：

```
   用户提问 🗣️
       │
       └───┐
           ▼
     🤖 AstrBot 接收请求
           │
   (知识库插件介入)
           │
   1. 🔍 在当前知识库中搜索与提问最相关的信息
           │
   2. 📚 获取相关知识片段
           │
   3. ✍️ 将知识片段 + 用户原始提问 整合进 Prompt
           │
           ▼
     🧠 发送给大语言模型 (LLM) 进行处理
           │
           ▼
   ✨ 生成包含知识库信息的、更优质的回答
```

---

<div align="center">
  <h2>🚀 更新日志 (Changelog)</h2>
  <p>我们致力于让 AstrBot 知识库插件越来越好用！以下是近期的主要更新：</p>
</div>

<details open>
  <summary>
    <h3><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="25" height="25" /> v0.4.0 - 体验优化与底层加固 <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="25" height="25" /></h3>
  </summary>
  <blockquote>
    <p>本次更新聚焦于提升 Milvus Server 用户的使用体验，并对插件内部结构进行了优化，为未来的功能迭代打下坚实基础。</p>
  </blockquote>
  <ul>
    <li>
      <p>✨ <strong>[核心修复] Milvus Server 索引创建优化</strong></p>
      <ul>
        <li><strong>问题</strong>：此前版本中，当使用 Milvus Server 作为向量数据库时，<code>embedding</code> 字段可能未被正确、及时地创建索引，导致后续检索效率低下或功能异常。</li>
        <li><strong>修复</strong>：我们重写并验证了 Milvus Server 的集合创建与索引管理逻辑，确保 <code>embedding</code> 字段在集合创建后能被<strong>自动且正确地创建向量索引</strong>。同时优化了索引存在性检查和加载流程，确保知识库在首次使用前已准备就绪。</li>
        <li><strong>用户价值</strong>：使用 Milvus Server 的用户将体验到更稳定、高效的知识检索性能，无需手动干预索引创建。</li>
      </ul>
    </li>
    <li>
      <p>🏗️ <strong>[架构升级] 代码结构重构</strong></p>
      <ul>
        <li><strong>改进</strong>：对插件内部代码进行了模块化重构，使得各功能模块职责更清晰，代码更易于理解和维护。</li>
        <li><strong>未来展望</strong>：此次重构为后续引入更多高级功能（如更细致的权限管理、更丰富的导入导出选项、插件间交互API等）铺平了道路。</li>
      </ul>
    </li>
    <li>
      <p>📄 <strong>[增强] 文件编码支持扩展</strong></p>
      <ul>
        <li>针对 <code>.txt</code> 和 <code>.md</code> 文件增加了更多的文件编码格式支持。</li>
      </ul>
    </li>
  </ul>
</details>

<details>
  <summary>
    <h3><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Airplane.png" alt="Airplane" width="25" height="25" /> v0.3.0 - 数据处理能力飞跃 <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Airplane.png" alt="Airplane" width="25" height="25" /></h3>
  </summary>
  <blockquote>
    <p>本次更新是里程碑式的，带来了对多种文件格式的强大支持，并显著提升了数据处理的效率和兼容性。</p>
  </blockquote>
  <ul>
    <li>
      <p>🌟 <strong>[重磅功能] 全能文本解析引擎集成！</strong></p>
      <ul>
        <li><strong>新能力</strong>：深度集成了业界领先的 <code>markitdown</code> 解析库，现在您可以<strong>直接导入并解析</strong>以下多种主流文件格式的内容作为知识源：
          <ul>
            <li>文档：<code>.pdf</code>, <code>.docx</code>, <code>.doc</code></li>
            <li>演示文稿：<code>.pptx</code>, <code>.ppt</code></li>
            <li>表格数据：<code>.xlsx</code>, <code>.xls</code>, <code>.csv</code></li>
            <li>网页与结构化数据：<code>.html</code>, <code>.htm</code>, <code>.json</code>, <code>.xml</code></li>
            <li>电子书：<code>.epub</code></li>
            <li>图片内容提取 (OCR)：<code>.jpg</code>, <code>.jpeg</code>, <code>.png</code> (依赖相应 OCR 配置)</li>
          </ul>
        </li>
        <li><strong>用户价值</strong>：告别手动复制粘贴！现在可以轻松地将您已有的各种格式文档、报告、网页内容直接转化为机器人知识库的一部分，极大拓宽了知识获取的边界，提升了知识库构建效率。</li>
        <li>💖 <strong>特别鸣谢</strong>：此项功能的实现离不开社区成员 <a href="https://github.com/Yxiguan">@Yxiguan</a> 的核心代码贡献！</li>
      </ul>
    </li>
    <li>
      <p>🛠️ <strong>[优化与修复] 依赖与稳定性</strong></p>
      <ul>
        <li>优化了插件依赖管理，减少了潜在的冲突。</li>
        <li>修复了若干在特定环境下可能出现的兼容性问题。</li>
        <li>提升了处理大型文件时的稳定性和内存使用效率。</li>
      </ul>
    </li>
  </ul>
</details>

<hr>

## 🛣️ 未来发展路线图 (Roadmap)

我们致力于持续改进 `astrbot_plugin_knowledge_base`，使其成为 AstrBot 生态中最强大、最灵活的知识管理工具。以下是我们近期和远期的主要开发计划：

### 1. 🧩 开放底层知识库数据管理接口

目前，插件的核心功能主要通过 `/kb` 指令对外提供。为了促进 AstrBot 插件生态的繁荣，我们将对核心知识库管理逻辑进行解耦，并提供更底层的 API 接口：

- **模块化与可重用性：** 将知识数据的增删改查、向量化、检索等核心功能抽象为独立的、可调用的方法。
- **为其他插件赋能：** 其他 AstrBot 插件开发者将能够直接调用这些底层方法，来构建自己的、更复杂的知识库应用场景，而无需重复实现基础的数据管理逻辑。
- **构建知识服务基础：** 旨在将本插件的核心逻辑演变为 AstrBot 平台的一个基础知识服务层，为整个框架提供统一、高效的知识存储与检索能力，提升 AstrBot 的整体智能化水平。

我们相信，这些改进将极大地提升插件的实用性和扩展性，为 AstrBot 用户和开发者带来更丰富的可能性。

---

## 🤝 社区与支持

欢迎通过 Pull Requests 或 Issues 为本项目贡献代码、提出建议或报告问题！您的参与是我们前进的动力。

- **QQ 群：** 遇到问题或希望进行更快的交流，可以添加 QQ 群:`953245617`。问题验证填写关键词 `lxfight` 即可。

⭐️ **如果您觉得这个插件对您有所帮助，请给个 Star 吧！** 您的支持对我们至关重要。

## 📜 许可证

本项目遵循 AGPLv3 许可证。请查看 [LICENSE](LICENSE) 文件以获取更多信息。

---
