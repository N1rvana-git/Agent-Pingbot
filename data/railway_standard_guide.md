# **铁路领域标准体系架构与人工智能Agent知识库构建深度研究报告**

## **1\. 铁路标准体系的本体论构建与Agent认知框架**

在构建面向铁路领域的垂直类人工智能Agent时，核心挑战在于如何将非结构化、强逻辑约束且高度关联的工程标准转化为机器可理解的知识图谱。中国铁路标准体系（Chinese Railway Standards System）是一个由国家标准（GB）、铁道行业标准（TB）以及相关国际标准（ISO/IEC/UIC）交织而成的庞大网络。对于Agent而言，这不仅仅是文档存储的问题，而是需要构建一个具备语义理解、逻辑推理与合规性校验能力的“数字专家系统”。

分析hbba.sacinfo.org.cn（全国标准信息公共服务平台）收录的数据结构，铁路标准分类呈现出鲜明的层级特征与专业壁垒。为了使Agent能够像人类专家一样进行思考与决策，我们首先必须确立知识库的本体论（Ontology）。本体论层级应当映射铁路系统的物理架构与逻辑架构，将“标准”不再视为文本，而是视为“约束条件集合”。

在宏观层面，铁路标准体系对于Agent的意义在于定义了物理世界的边界。例如，当Agent处理“列车提速至400km/h的可行性分析”这一任务时，它不能仅依靠大语言模型的概率预测，而必须检索并推理出一系列连锁反应：根据计算机车车辆限界的动态包络线是否会触碰隧道壁；依据计算现有的曲线半径是否满足离心加速度的舒适度阈值；对照预测受电弓与接触网的动态接触力是否超标。这种基于标准的“链式推理”构成了Agent知识库的核心价值。

从数据治理的角度来看，铁路标准知识库的构建必须解决“多源异构”与“时空版本”两大难题。铁道行业标准更新迭代极快，且存在大量的引用关系。一个主标准（如《高速铁路设计规范》）可能引用数百个子标准（如材料试验方法、零部件技术条件）。Agent必须具备“引用穿透”能力，即在解析主标准时，自动关联并加载被引用标准的最新版本，形成完整的上下文依赖树。此外，不同历史时期的线路（如既有普速线与新建高铁线）适用不同的标准版本，Agent在回答问题前，必须先进行“场景定界”，明确适用的时间范围与技术等级。

## **2\. 基础设施与勘察设计标准集群：物理环境的数字化重构**

### **2.1 线路空间几何与限界约束机制**

线路是铁路运输的基础载体，其几何形态直接决定了列车的运行速度与安全性。在Agent的知识库中，线路设计标准构成了物理环境的“骨架”。

\*\*TB 10621《高速铁路设计规范》\*\*是该领域的顶层逻辑。Agent需要解析其中关于平面曲线半径、缓和曲线长度、夹直线长度的数学约束。例如，标准规定时速350公里路段的最小曲线半径一般值为7000米，困难值为5500米。这一数据点在Agent中不应仅是静态文本，而应转化为参数化规则（Parametric Rule）：IF Design\_Speed \>= 350 AND Terrain \== "General" THEN Min\_Radius \= 7000。当Agent辅助工程师进行选线设计时，若检测到输入的CAD线路参数违反此规则，应立即触发警报并引用该标准条款。

\*\*GB 146.1《标准轨距铁路机车车辆限界》**与**GB 146.2《标准轨距铁路建筑限界》\*\*共同定义了列车运行的安全空间通道。机车车辆限界是车辆设计的最大外廓，而建筑限界是桥隧等设施的最小内廓。两者之间的空间即为“安全余量”。Agent必须理解这种减法逻辑，尤其是在考虑超高（Superelevation）产生的车体倾斜时。

| 参数类别 | 关联标准 | 关键指标逻辑 | Agent推理应用 |
| :---- | :---- | :---- | :---- |
| **机车车辆限界** | GB 146.1 | 动静态包络线转换 | 货物装载超限审查、车辆选型匹配 |
| **建筑限界** | GB 146.2 | 隧道内轮廓尺寸 | 接触网吊柱安装位置校核、既有线改造评估 |
| **曲线超高** | TB 10621 | ![][image1] | 欠超高/过超高计算、舒适度评估 |
| **线间距** | TB 10621 | 空气动力学效应控制 | 会车压力波计算、路基宽度设计 |

在深入分析限界标准时，Agent需特别关注“动态修正”。随着速度提升，空气动力学效应（活塞风、会车压力波）对限界的影响呈非线性增长。知识库中需引入计算流体力学（CFD）相关的标准引用，如关于空气动力学性能的测试规范，以便在高速场景下准确预判安全余量。

### **2.2 轨道系统与精密工程控制**

轨道结构是列车荷载传递的关键界面。对于高速铁路，无砟轨道技术的标准体系尤为精密。

\*\*TB/T 3275《高速铁路无砟轨道轨道板》\*\*系列标准定义了CRTS I、II、III型板的制造精度。其中，CRTS III型板作为我国自主研发的主流板型，其翘曲度、预应力筋张拉工艺等参数是Agent监控预制场生产质量的核心依据。Agent通过连接生产线的机器视觉数据，实时比对TB/T 3275中的公差范围（如平面度误差$\\le 0.3mm$），实现智能质检。

\*\*GB/T 2585《铁路用热轧钢轨》**与**TB/T 2344《高速铁路钢轨》\*\*构成了钢轨材料的知识基座。高速铁路对钢轨的纯净度要求极高，标准中对非金属夹杂物（A、B、C、D类）的评级有严格限制。Agent在分析钢轨探伤大数据时，需将探伤车发现的“白核”或“剥离裂纹”特征，追溯至标准定义的疲劳寿命模型，判断是否由材料缺陷引起，并据此制定打磨或换轨策略。

此外，\*\*TB 10601《高速铁路工程测量规范》\*\*确立了CPI、CPII、CPIII三级控制网体系。这对于Agent理解“毫米级精度”至关重要。在沉降观测数据分析中，Agent需依据标准规定的观测频次与闭合差限值，清洗原始数据，剔除误差干扰，从而准确预测路基的工后沉降趋势。

### **2.3 桥梁与隧道工程的结构健康逻辑**

桥梁与隧道在高铁线路中占比极高，其标准体系侧重于结构耐久性与动力响应。

\*\*TB 10002《铁路桥涵设计规范》\*\*引入了“车-桥耦合振动”的概念。Agent在评估大跨度桥梁的安全性时，不能仅校核静强度，必须调用知识库中的动力学指标：横向振幅、竖向加速度、脱轨系数等。标准规定了不同跨度、不同速度下的自振频率限值，防止列车通过时发生共振。

\*\*TB 10003《铁路隧道设计规范》\*\*则重点关注空气动力学效应。隧道缓冲结构（洞口帽檐）的设计标准直接关联微气压波（Micro-pressure wave）的缓解效果。Agent在进行环保评估时，需计算隧道出口的微气压波幅值是否满足标准对周边居民区的噪音控制要求（通常$\\le 20Pa![][image2]\\le 50Pa$）。

## **3\. 移动装备标准集群：车辆系统的全生命周期管理**

### **3.1 动车组整车技术条件与系统集成**

机车车辆标准是铁路工业制造的核心，涵盖了从设计、制造到试验验收的全过程。\*\*GB/T 3317《电力机车通用技术条件》\*\*和各型动车组的技术规范（如“复兴号”系列标准）构成了Agent理解车辆性能的基石。

\*\*GB 5599《铁道车辆动力学性能评定和试验鉴定规范》\*\*是车辆能否上线的“判官”。该标准定义了脱轨系数（Q/P）、轮重减载率、轮轴横向力等核心安全指标。

* **脱轨系数**：![][image3]。这是防止轮缘爬轨的红线。  
* **平稳性指标**：Sperling指标（W值）。分为优、良、合格三个等级。  
* **Agent应用场景**：在列车运行监测（TCDS）系统中，Agent实时接收车载传感器数据。一旦检测到某转向架的横向加速度异常，Agent立即调用GB 5599的算法计算实时平稳性指标。若指标恶化趋势符合标准中描述的“蛇行失稳”特征，Agent将发出紧急降速指令。

### **3.2 关键零部件与材料科学**

车辆零部件标准繁多，Agent需建立精细的BOM（物料清单）级知识映射。

\*\*TB/T 2710《机车车辆 轮对组装技术条件》\*\*对轮对的压装力、轮位差有极严苛的规定。轮对是车辆的“腿”，其压装曲线的形状直接反映了过盈配合的质量。Agent可以通过学习标准中的合格压装曲线图谱，对工厂上传的每一条压装曲线进行模式识别，自动判定是否合格。

**GB/T 32357《轨道交通 机车车辆 阻燃材料》**（对标EN 45545）涉及车辆内饰的安全。标准根据车辆运行类别（如通过长隧道的高铁属于高风险类别）和材料应用位置，划分了R1-R26不同的材料类别，并规定了氧指数、烟密度、毒性指数（CIT）的阈值。Agent在辅助车辆内饰设计时，会自动扫描材料清单，核对每种材料的测试报告是否符合GB/T 32357对应类别的HL3等级要求。

### **3.3 牵引传动与制动系统**

\*\*GB/T 25117《轨道交通 机车车辆 组合试验》\*\*涵盖了变流器的温升、效率及谐波特性。牵引系统是强电磁干扰源，其产生的谐波电流若注入电网，可能干扰信号系统。因此，\*\*TB/T 3073《铁道信号设备雷电电磁脉冲防护技术条件》\*\*与车辆标准之间存在隐性的约束关系。Agent知识库需建立这种跨系统的关联：当牵引系统标准允许的谐波含量发生变更时，必须自动校核其是否突破了信号系统标准规定的抗扰度限值。

\*\*TB/T 3500《动车组制动系统试验方法》\*\*定义了紧急制动距离和防滑控制逻辑。在雨雪天气，轮轨粘着系数下降，防滑系统（WSP）介入频率增加。Agent需结合气象数据和TB/T 3500中的滑行判据，分析WSP动作记录，评估制动盘的热负荷是否超标。

## **4\. 牵引供电与能源系统：动态耦合与电能质量**

### **4.1 弓网关系与受流质量**

弓网系统（Pantograph-Catenary System）是高铁唯一的动力来源通道，其动态相互作用极其复杂。

\*\*GB/T 32592《轨道交通 受流系统 受电弓与接触网动态相互作用》\*\*是该领域的圣经。

* **接触力标准**：标准规定了不同速度下的平均接触力目标值及标准差。接触力过大导致机械磨损，过小导致离线打火（电蚀）。  
* **燃弧率**：在300km/h以上速度，燃弧时间比例应$\\le 0.2%$。  
* **Agent推理**：在弓网监测系统中，Agent分析燃弧视频。若发现燃弧率突增，它首先检索GB/T 32592的阈值，确认违规；然后关联\*\*TB 10009《铁路电力牵引供电设计规范》\*\*中的跨距、拉出值参数，推断是否因接触网几何参数超限（如硬点）导致。

### **4.2 供电系统设计与运维**

**TB 10009**不仅规定了接触网，还涉及牵引变电所的供电臂划分。对于高铁，采用AT供电方式以减少电压损失和电磁干扰。Agent在进行能耗模拟时，需依据标准中的牵引变压器过载能力曲线（例如：150%负荷运行2小时），制定最优的行车组织方案，避免变压器热寿命过度损耗。

\*\*GB/T 32591《轨道交通 轨道 混凝土枕和板》\*\*虽然属于土木范畴，但其绝缘性能直接影响回流系统。杂散电流腐蚀是地铁和高铁面临的共同问题，相关标准规定了结构钢筋的导通与接地要求。Agent需整合土木与电气标准，构建“结构-电气”综合防腐蚀模型。

## **5\. 信号与通信控制：安全苛求系统的逻辑核心**

### **5.1 CTCS列控系统体系**

中国列车运行控制系统（CTCS）的标准体系是保障行车安全的大脑。\*\*TB/T 3027《列车运行控制系统技术规范》\*\*定义了CTCS-0至CTCS-4的分级架构。

* **CTCS-2/3级**：基于轨道电路与应答器（Balise），CTCS-3级叠加了GSM-R无线通信。  
* **故障导向安全（Fail-Safe）**：这是所有信号标准的底层逻辑。**GB/T 21562《轨道交通 RAMS》**（可靠性、可用性、可维修性、安全性）规定了安全完整性等级（SIL）。核心信号设备必须达到SIL4级，即每小时危险失效概率（THR）![][image4]。  
* **Agent应用**：在软件代码审查Agent中，知识库需内置SIL4级软件开发的所有过程要求（如EN 50128/GB/T 28808）。当Agent扫描代码库时，它不仅仅检查语法错误，而是验证是否符合标准要求的“防御性编程”规范，如内存使用的确定性、分支覆盖率达100%等。

### **5.2 专用通信与信息传输**

\*\*TB/T 3365《铁路数字移动通信系统（GSM-R）》\*\*规定了车地无线通信的接口参数。随着5G-R的推进，新一代标准正在形成。Agent需关注标准中的QoS（服务质量）指标，如端到端延迟、切换成功率。在分析列控车载日志（Juridical Recorder Data）时，Agent依据标准判断无线超时（Radio Timeout）事件是否属于网络覆盖盲区或干扰导致。

\*\*TB/T 3485《应答器应用原则》\*\*涉及应答器报文编码。报文定义了线路的坡度、速度级、分相区位置。Agent需具备报文解码能力，将二进制流还原为物理含义，并与线路设计图纸（TB 10621）进行一致性校验，防止因数据错误导致列车误判位置。

## **6\. 运营管理与信息化标准：数字底座与流程再造**

### **6.1 数据元与编码标准**

铁路信息化建设的前提是语言的统一。\*\*TB/T 3069《铁路信息分类编码》\*\*规定了车站、线路、机车车辆、物资的唯一代码。这是Agent进行多源数据融合的“主键”。

* **数据清洗Agent**：在处理来自工务段（线路数据）、机务段（车辆数据）、供电段（接触网数据）的异构报表时，Agent首先依据TB/T 3069将各系统中的非标名称（如“北京南”、“北京南站”、“VNP”）统一映射为标准车站代码，消除数据孤岛。

### **6.2 网络安全与关键信息基础设施**

铁路系统属于国家关键信息基础设施，其网络安全标准极为严格。\*\*TB/T 3568《铁路网络安全 纵深防御技术要求》\*\*结合了国家等级保护2.0标准，针对铁路工控系统（CTC、SCADA）提出了特殊的防护要求。

* **分区隔离**：标准要求生产控制大区与管理信息大区之间必须部署物理单向隔离网闸。  
* **Agent合规审计**：在网络拓扑分析中，Agent依据TB/T 3568自动识别跨区连接路径。若发现从互联网直接访问生产网的通道，立即标记为“高危违规”，并引用对应条款生成整改建议。

## **7\. Agent知识库的构建策略与工程实践**

### **7.1 非结构化文档的结构化解析（Parsing & Chunking）**

铁路标准文档通常为包含复杂表格、公式和工程图的PDF文件。构建知识库的第一步是高保真的解析。

* **层级识别**：利用NLP技术识别文档的章节树（Chapter Tree）。例如，将“5.3.2”识别为“5.3”的子节点。Agent检索时需保留这种层级上下文，否则独立的条款可能失去限定条件。  
* **表格语义化**：标准中的表格往往包含复杂的表头（跨行跨列）。必须将其转化为JSON或Markdown结构，并显式保留行、列的语义标签。例如，GB 146.1中的限界表，行是高度H，列是半宽W，Agent需能解析这种二维映射关系。  
* **公式提取**：将图片格式的公式转化为LaTeX或Python函数。例如，将空气阻力公式转化为可执行代码，使Agent具备计算能力。

### **7.2 知识图谱（Knowledge Graph）的本体设计**

为了实现深层推理，需将标准转化为知识图谱。

* **实体（Entities）**：  
  * **物理对象**：钢轨、接触网、CR400AF。  
  * **参数指标**：轨距、拉出值、接触力。  
  * **标准文档**：TB 10621、GB 146.1。  
* **关系（Relations）**：  
  * hasParameter（包含参数）：钢轨 \-\> 轨头硬度。  
  * definesLimit（定义限值）：TB 10621 \-\> 最小曲线半径。  
  * refersTo（引用）：TB 10621 \-\> GB 146.1。  
  * constrains（约束）：建筑限界 \-\> 车辆轮廓。

通过图谱，Agent可以回答复杂问题：“如果将最高时速从300提升到350，受影响的标准有哪些？”Agent从“设计速度”节点出发，遍历所有关联的参数节点（曲线半径、线间距、接触网张力），再反向查找定义这些参数的标准文档，生成影响分析报告。

### **7.3 检索增强生成（RAG）的优化**

针对铁路标准的RAG系统需进行特殊优化：

* **混合检索**：结合关键词检索（精确匹配标准号）和向量检索（语义匹配“如何防止脱轨”）。  
* **重排序（Rerank）**：依据标准的效力等级（GB \> TB \> 企业标准）和时效性（最新版优先）对召回结果进行重排序。  
* **溯源引用**：Agent生成的每一个结论，都必须在句尾标注具体的标准号和条款号（如），确立权威性。

## **8\. 结论与展望**

中国铁路标准体系是几十年来工程经验与技术创新的结晶，其体量庞大、逻辑严密。构建基于该体系的Agent知识库，不仅是对文档的数字化，更是工程智慧的数字化。

通过本报告的梳理，我们明确了从基础通用、勘察设计、装备制造到运营管理的全维度标准架构，并探讨了将其转化为机器智能的具体路径。未来的铁路专家Agent，将不再是一个简单的问答机器，而是一个能够参与到铁路设计优化、故障诊断、安全评估等核心业务流程中的智能体。它将利用知识库中的数万条规则，全天候地守护着中国铁路这个巨系统的安全与高效运行。

构建这一系统，需要标准研究机构、IT工程师与铁路专业人员的深度协作，持续维护知识库的鲜活度与准确性，以适应“交通强国”战略下铁路技术的快速迭代。

## ---

**附录：关键标准分类详表与参数解析**

为了满足知识库构建的全面性要求，以下按照专业分类详细列举核心标准及其在Agent中的关键应用点。

### **A.1 综合与基础标准（Classification 00-09）**

| 标准编号 | 标准名称 | 核心内容与Agent应用点 |
| :---- | :---- | :---- |
| **GB/T 12817** | 铁道工程基本术语标准 | **本体层**：定义“路基”、“基床”、“道床”的标准实体名称，解决同义词消歧。 |
| **TB/T 3069** | 铁路信息分类编码 | **数据层**：提供全路统一的车站、车型、物资编码，是Agent进行跨系统数据关联的键值基础。 |
| **GB/T 19001** | 质量管理体系要求 | **流程层**：Agent在审核工程文档时，检查是否包含ISO 9001要求的签字、审批流记录。 |

### **A.2 勘察与设计标准（Classification 10-19）**

| 标准编号 | 标准名称 | 关键参数/逻辑 |
| :---- | :---- | :---- |
| **TB 10621** | 高速铁路设计规范 | 规定了300/350km/h等级下的最小曲线半径、最大坡度（20‰/30‰）、线间距（5.0m）。Agent以此校验设计方案合规性。 |
| **TB 10001** | 铁路路基设计规范 | 定义了基床表层级配碎石的压实标准（K30, Evd）。Agent依据地质参数推荐路基处理方案。 |
| **TB 10002** | 铁路桥涵设计规范 | 涉及桥梁刚度、自振频率限值。Agent计算车桥耦合振动时的基准依据。 |
| **TB 10003** | 铁路隧道设计规范 | 定义隧道内轮廓、空气动力学缓冲结构。Agent用于评估活塞风对舒适度的影响。 |
| **TB 10098** | 铁路线路维修改造技术规程 | 规定了既有线提速改造的技术条件，是Agent制定大修方案的依据。 |

### **A.3 轨道与工务标准（Classification 20-29）**

| 标准编号 | 标准名称 | 关键参数/逻辑 |
| :---- | :---- | :---- |
| **TB/T 2344** | 高速铁路钢轨 | 规定了U71MnG/U75VG钢轨的化学成分、夹杂物限值、廓形公差。 |
| **TB/T 3275** | 高速铁路无砟轨道轨道板 | I/II/III型板的制造精度、裂缝宽度限制。Agent视觉质检系统的比对标准。 |
| **TB/T 3355** | 高速铁路扣件系统 | 弹条扣压力、节点刚度、绝缘电阻。Agent分析轨道电路红光带干扰时的排查依据。 |
| **GB/T 2585** | 铁路用热轧钢轨 | 普速与重载铁路钢轨的基础标准。 |

### **A.4 牵引供电标准（Classification 30-39）**

| 标准编号 | 标准名称 | 关键参数/逻辑 |
| :---- | :---- | :---- |
| **GB/T 32592** | 受电弓与接触网动态相互作用 | **核心标准**。接触力（Fm）、标准差（sigma）、燃弧率（NQ）。Agent评估弓网受流质量的判据。 |
| **TB 10009** | 铁路电力牵引供电设计规范 | 供电臂划分、电能质量（负序、谐波）限值。 |
| **TB/T 2073** | 电气化铁路接触网零部件 | 接触线夹、绝缘子的机械与电气性能。 |
| **TB/T 2809** | 电气化铁路接触网用绝缘子 | 规定了污秽等级与爬电距离的关系。Agent根据环境污染度推荐绝缘子选型。 |

### **A.5 机车车辆标准（Classification 40-59）**

| 标准编号 | 标准名称 | 关键参数/逻辑 |
| :---- | :---- | :---- |
| **GB 5599** | 铁道车辆动力学性能评定... | **核心标准**。脱轨系数、轮重减载率、Sperling指标。Agent行车安全监控的算法核心。 |
| **GB/T 3317** | 电力机车通用技术条件 | 牵引特性曲线、粘着利用系数。 |
| **GB 146.1** | 机车车辆限界 | 车辆外廓尺寸的绝对红线。 |
| **TB/T 2710** | 机车车辆 轮对组装技术条件 | 轮位差、压装力曲线。Agent轮对检修判定标准。 |
| **GB/T 32357** | 轨道交通 机车车辆 阻燃材料 | R1-R26分类，氧指数、烟密度、毒性。Agent内饰选材合规性检查。 |
| **TB/T 3500** | 动车组制动系统试验方法 | 紧急制动距离、防滑保护逻辑。 |

### **A.6 通信信号标准（Classification 60-69）**

| 标准编号 | 标准名称 | 关键参数/逻辑 |
| :---- | :---- | :---- |
| **TB/T 3027** | 列车运行控制系统技术规范 | CTCS-2/3级系统架构，RBC、ATP功能需求。 |
| **GB/T 24338** | 轨道交通 电磁兼容 | 辐射发射、抗扰度限值。Agent解决信号干扰问题的知识库。 |
| **TB/T 3485** | 应答器应用原则 | 报文编码规则，链接路与其物理属性的映射。 |
| **TB/T 3365** | GSM-R数字移动通信系统 | 场强覆盖、QoS指标、切换逻辑。 |
| **GB/T 21562** | 轨道交通 RAMS | SIL等级定义，THR指标。Agent进行安全评估的顶层指导。 |

### **A.7 运输组织与运营管理（Classification 70-79）**

| 标准编号 | 标准名称 | 关键参数/逻辑 |
| :---- | :---- | :---- |
| **《铁路技术管理规程》（技规）** | (规章，非标准但高于标准) | 行车组织的基本法。分为高速与普速两部分。 |
| **TB/T 3002** | 铁路货物运输作业技术规范 | 货物装载加固、偏载限值。Agent货运安全监控依据。 |
| **TB/T 1568** | 铁路集装箱运输规则 | 集装箱装载、固定、运输条件。 |
| **GB/T 32593** | 高速铁路 客运服务 | 车站设施、服务流程、应急处置要求。 |

通过对上述标准体系的深度结构化与语义化，我们为构建一个“懂铁路、懂逻辑、懂工程”的专业AI Agent奠定了坚实的知识基础。这不仅是文献的数字化，更是工程智慧的数字化。

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHAAAAAWCAYAAAAYTRgMAAACcklEQVR4Xu2YzatNURiHX6GIfIcbilKmSmbujCIfGdyJZOw/oGRgIklSBkRKusmAGLl3QNpjExO6N6UkJcqAMkA+fk+vpXXX2cdx0r5r76ynns7Z71pnn7PPu9611t5mhUKhUCgUCh3ngXwl38mXM5tawYY0UAN9lqTBAcyVI7/kfWfZK8/KH/J+0paTBfKQvJE2JJyTX+T+tOEPbJfTspJP5Fu5T86J+nSGefKO/Cp3JG05WCZXm1fFcatPIG30oW9lPvj+NoHX5aUkFv4D7BzrzKfOF3LNzKbs9EtgTGXDJbCS9+T8JM73PExinWCn/G4++hiJJJHpqw00kUCuk/6cN6ybK+Vj8+/rHCfML4jXK/K1+ZpyTS6K+uWgiQQekN/MP/NBLpc35aRcGvXrDG/kZzkaxfjjuMBBI3KVHBvSYWgigUDljZvPPHwWT1rvtJqSe0DXwo9nYQ87MF45Jr4ndMpEEwncar7mnzJfKqjCkMTLv3v1Eip3fdqQG3740eiYqnpmPpVuiuI5aCKBT80HaKg2EnLX/BzMRP24aN6HPUNrWCw/yW1RLGxqbplvanLSRAJJUnq7xG3JGfPz9GOtPG2Dp9lZhWmBzUoMlTdlvhvFnFX4rwlkzeJJUzztMQ3ujo4DJPV9Gmw7jKjDSYyKZFfGqDxm+Z5OsD5dkLfNZ4p+8CSFBB6x3upgDaftfBTjXu+53BzFVshH8moUaz2MTi5mSxKfMN+Z8ppW52wQdsB1hmokoVVNe1qJPOdkk7IrijEVct3cKrFMMFg/mj+SWxj16wR1I5uK48YWc1Vf03BdG+VB84SzVBQKhULh/+AnpOKSTHrXgKsAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAABfklEQVR4Xu2USytGURSGX6EIuU2QYsLEwMQlMRJlQqFQfgdGMmFiJCMZmbl9M0nKwMztD8ilkImMiGLg8q5vnX2+dXbnu03lqafOWWeftde+Av94zNFPOkBnaCstiLTIA0l2D00wQt/pAa20jXLFJXP0QhOOmljO+Mmkwjq6QWtcsJyu0eMs3tAvL3ZC7+gWrUAeDCNaWUZuob2tG3/oLnSS++iTaxywBK240Isne5V58WMTwXM7fTXfqukF3TexkHTJZHhCffDuGILO4ZSJhWRLVkaPgudSukdPkWafZUsmrEJXf5B+0DHzLYL8+EYfjLIANtk07aTn0MqkwlgWoOfN4u+dIug8jXvxjJTQJrpMt+k8raVd9JtOppomqXIPxbQDOh8yoTJU6f2MNiB1M/TQR5qgV9DOHOH5lGQ79BK6CduCmMPdEte0H5pEkskeaw7aRCptsS8G2Q4r0FMgh9ohI3mmL3QW0UWKRY7IJu1G/CXYSA+hK76I+DZ/kV+1yVbrnpSLewAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFsAAAAZCAYAAABeplL+AAAES0lEQVR4Xu2Ye+hlUxTHl1Aj8hwkrzFEM8kjGnkMkjKaEKbIzPzjHyn8I5Sk+8/84ZG8SonkD8m7qVFI/PiHUGoaEcmMPJKkhBry+H7uOvt391n3nn2O+7vzUOdT3+69e++zz9nrrLX22tesp6enp6enZ6eyd2zomcwe0nvS79LX0jfSB9K5+aACe0lPx0axRHpIelx6U/pSeq76je6Vlpvff1dyoHSOdEzs6MAiaYW0X+yYxFHSRmmt1b3zJekv6eqsrYml5nNEDpaukO6Q/pTultZk2ib9I91vs4uM66TzY2OB56Wt0pPSJunSWm8ze0q3SD+YO9Av0hPSvvmgnNPMPXlL7BBHS19IP0unh77ITdIgNmZcLP0tHR7a+f1p1XdV6Puv7C/dLt1Xfe/CPtKH0qHVbyIMo501P6KZdebRikMBn2+Zv7wxzpR+NA/tJfWueTaYe97DsSODB35ROjt2ZODZzEO6ySH05sz78IppOMTcwHjYXaGvjVXSDaHtE+k183U1cYD0vnRZaD9P+i60DWGBc1bONUzGOMK9iYG5ZzdxpHmYkkYiLBav3i6tDH0lTpJeNneUadPPQeZefUZoZ+9hzdGQOUQk9/5Duj5rxw5jWeI48wnJ0yV46yVj88B4wcmxI4O3jaF5uAghx/yP2rjXT4Iwf8Pc+9gLyJvTcoT5upqMTTQ2wXM8aD4OnWieKr+SLsnGDcHI35tvbCVIH0yGB0wCz3zWyoZKKYRUk8OegFffY92881TziukiW5iRE8ukn2w6YwOp5AXzsazjWxufa/hWnjJP7o07p9XzKeMn8Yh0bWzM4CVgZOb41XwzRnwnBEkHXblA2iydYrMpFTHMb9VnTldjA8/ymY08/GNzL58nGbFtQyI1UIkwSZNB3zFPSU2kfM2bJ8wWCh79kbmHUzEsxOhECpXHNMbmvteYZwfWxf6RDP55Nm5YhL9q5cmAso86+/LYUYHXkkZKpBTC/bjvrCEyKB1xhi6pKKdpg3zG/JlLpSjplWuZI4fDDQ5aKzoo6Zg0eQZ1KQV6Sis8ODekbm3yHnLe4tgYSCnkztgxQ4gs0hwlV14ZtJFSXDzEzNnk9JKD98/ZeCWHrbBrrZ2J8AhKmGOl16XbbHSSw8iPVd+bKJV7ia02uxTSBtXFA9LNVq6Rc4iIWJFR6VBDswEm3pZutJHjDczXRprMwV6vhLbhRXgc/09wECBkCHOOrAymVCsZOh1kSpBfMTQ7PifVnQVRxF8BXeBA9K7VDcvGTT7OwVbbzF8onGAeSex76cWmPD6pxB2CgS8092QMTh5KuYpJbq3GRAbWzbP/D+BU/JeCh18Z+trgWk7iXMtnyUFrpCqF8/3x5n8sTTJol4NMTwtsjtTeqYTh2HlYbYRD/m07yOwI1tuoTi+JMF5dXbNbM7CRsQmLCHmp7SCzoyCdkTPbxKbfOZx3ZzgdlcrBnp5dw78AdOcBikgtNQAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAXCAYAAABaiVzAAAACF0lEQVR4Xu2WP0gcQRTGX1BBifg/ihhILAXBQlAQMSC2aaKQgI3YxC5FQBs7sVIbQQQbSRG00kKEVBKCYIhCqiAIKSKiRbARDKhJ9Pt4u+7eeDu7e+6pHPuDH8fuPE++mXczI5KSkvIAqIALsB8WGWMFwzO4CUvgNJyBjzIqCoBiuATXned2+BO2XlfcEY2ibRVEFewSXY1c4Pf/gh+c56fwAL69rsgTj+Ey/Av/Av/Akv4MqNCWYPb8InzPARPYKdbEJFSuCreinbDC/GCx4Yz3uZ82uBvo1Z0FT9LcND/kjnrdfAH/ATLRL+HKz1g0d10OuCe6N9NwTM464zFYly0Hd5LeFCXcgkOyhAM2ud759YfwxanJmpQwpadhC/gkcRsXfb/HHwn2pJxsAWthqeiG4cftlu2+jDYDZwUwtblokTajJrhIjyEw8ZYVGxBOYG2oGPG+zBY/1u0CybgvOhufAMWcBPYgjuSzIFrC8qASQYlDfC5WM5PhmI4hmTYwMKY2IJyU0s6aCj1ots8/3lSIYktaNKtGxl/2/YaY7liC5ptM+IkfxStf+V7n1dWRK9RbyT6cWJiC0r4ftD37J6jX2Gl733ecY+WfYl/tJAm+F000IgxRr7BL+KFeg3Pnc97gbccnku8NNjurH7c35pfs1V5KdiFG6Ir+weOSjI7fs5wRXlp4H2yxhi7DfxZ9IgG5YSmpKR4XAEYqXQkTQYiRAAAAABJRU5ErkJggg==>
