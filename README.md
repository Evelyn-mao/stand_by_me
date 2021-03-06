# Mixlab 百度飞桨创意赛
Mixlab AI创造营 参赛作品名
个性图腾-灵魂使者
Stand by me

# 作品简介
Jojo立的具象版，从此路人也能看到灵魂使者了！不用再担心你熊熊燃烧的中二之魂，随心所欲随时随地的来模仿jojo立吧！

# 实现原理
基于paddlepaddle的人体关键点检查预训练模型和opencv进行一系列的图像处理实现。
先通过标准的jojo立动作来确定标准动作库
打开摄像头，当检测你的pose大致符合jojo立的时候(关键点夹角80%符合在标准动作的15°内算通过)，触发特效，在原有视频流的基础上叠加替身的动画特效。
特效为灵魂使者的出现，并播放相关灵魂使者的处刑曲，灵魂使者会出现在用户的身边或身后。身后的特效是利用paddlepaddle的人像抠图功能，随后再把用户叠加在替身上，类似b站防挡脸的效果。真正做到动画里灵魂使者在身后实现。

# 目前进度
介于时间和能力的有效，目前能触发3个动作和一个白金之星还有黄金镇魂曲的特效。未来会持续更新，添加更多的触发效果和替身。

# 后续预期升级功能的思考🤔
1. 多人场景的触发，比如dio和乔太郎互相对立的名场景复现。
2. 触发条件的升级，比如摆好姿势后，还要喊出替身的名字才能触发。
3. 引入战力系统(动作打分)，战力由动作的标准与否还有相关配饰决定。动作和配饰越接近原作，使者的外形(高矮胖瘦)，出拳速度等越还原。然后可以真的进行格斗互动，而不是放动画。实现原理类似名字大战，外形通过放射变换对替身的外形进行处理...
4. 导入替身的3D模型，匹配人体关键点进行联动...
5. ....欢迎大家一起脑洞...

# 商业前景
1. 主题乐园，游乐场等特效合影功能
2. 与海报合影的特效联动
3. 博物馆等教育主题下，与历史人物的联动等
4. 漫展等还原名场景的情况

# 其他说明与jojo的介绍
* 项目最初其实是想通特定的表情或肢体动作来触发特殊的图腾(守护者)的概念，再讨论的过程中发现用jojo来说明这个应用的场景特别合适。

* Jojo的奇妙冒险是一部享誉全球的动漫名作，赞美了人类勇气的赞歌。作者通过非凡的想象力和奇妙的画风影响了一代又一代的读者。由于原作中的角色会摆出很多看起来像雕塑一样的奇特pose，引起全世界各地的模仿，甚至影响了时尚界。

  原作中替身使者的概念是灵魂具象化的表现，只有同为替身使者的角色才能互相看见彼此的替身使者，路人视角是看不到的。所以才会造成路人视角看起来是两个人在原地站着，看不到使者们的战斗。

  所以当有人模仿Jojo立的时候，在不了解jojo的普通人的视野里，会觉得很奇怪，但其实在了解该作的漫友们眼里，是可以联想到他们的替身使者的。

* 而这款程序就是用来打破次元壁，让大家更加直观地了解模仿的特效。
