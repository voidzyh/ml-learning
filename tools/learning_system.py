#!/usr/bin/env python3
"""
ML/DL 学习系统 - 统一入口
整合 Obsidian 集成和学习追踪功能
"""

import sys
import os
from pathlib import Path

# 添加项目根路径和tools路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from obsidian_integration import ObsidianIntegration


def print_banner():
    """打印欢迎横幅"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║           🧠 ML/DL 50周学习系统                            ║
    ║                                                            ║
    ║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ║
    ║                                                            ║
    ║   软工科班生的机器学习/深度学习系统化学习之路              ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)


def print_help():
    """打印帮助信息"""
    print("""
📚 可用命令:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 每日学习:
    today           查看今日学习计划
    daily           创建今日 Obsidian 笔记
    done            标记今日完成
    skip <原因>     跳过今天

📊 进度查看:
    status          查看总进度仪表盘
    week            查看本周概览
    dashboard       更新 Obsidian 仪表盘

📝 知识管理:
    concept <名称>  创建概念笔记
    explain <概念>  讲解概念（创建详细笔记）
    quiz [主题]     生成测验（创建测验笔记）
    review [周数]   创建周回顾

🚀 项目管理:
    project <ID>    创建项目笔记
    projects        列出所有项目

📖 间隔复习 (SM-2):
    review-today              查看今日复习卡片
    review-done <概念> <0-5>  评分复习卡片
    review-stats              查看复习统计

⚙️  初始化:
    init            初始化 Obsidian Vault
    init-mocs       初始化知识领域索引

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 示例:
    python learning_system.py today
    python learning_system.py daily
    python learning_system.py quiz linear-algebra
    python learning_system.py concept 梯度下降
    python learning_system.py review 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


def cmd_today(obsidian):
    """查看今日学习计划"""
    from ml_tutor import format_today_plan
    plan = obsidian.tutor.get_today_plan()
    print(format_today_plan(plan))


def cmd_daily(obsidian):
    """创建今日 Obsidian 笔记"""
    filepath = obsidian.create_daily_note()
    print(f"""
✅ 今日学习笔记已创建！

📁 文件位置: {filepath}

💡 接下来的步骤:
   1. 在 Obsidian 中打开此笔记
   2. 开始学习，记录笔记
   3. 完成后运行: python learning_system.py done
    """)


def cmd_done(obsidian):
    """标记今日完成"""
    from ml_tutor import format_status
    result = obsidian.tutor.mark_done()
    status = obsidian.tutor.get_status()

    print(f"""
✅ 第{result['week']}周第{result['day']}天已完成！

📊 总进度: {result['progress']:.1f}%
🔥 连续学习: {result['streak']}天
""")

    if result['is_saturday']:
        print("🎉 一周结束！建议创建周回顾:")
        print(f"   python learning_system.py review {result['week']}")
        print()

    # 更新仪表盘
    obsidian.update_progress_dashboard()
    print("📊 进度仪表盘已更新")


def cmd_status(obsidian):
    """查看总进度"""
    from ml_tutor import format_status
    status = obsidian.tutor.get_status()
    print(format_status(status))
    print()
    print(f"💡 在 Obsidian 中查看详细进度: {obsidian.vault_path}/📊 Progress.md")


def cmd_week(obsidian):
    """查看本周概览"""
    from ml_tutor import format_week_overview
    overview = obsidian.tutor.get_week_overview()
    print(format_week_overview(overview))


def cmd_concept(obsidian, concept_name):
    """创建概念笔记"""
    filepath = obsidian.create_concept_note(concept_name)
    print(f"""
✅ 概念笔记已创建: {concept_name}

📁 文件位置: {filepath}

💡 接下来:
   1. 在 Obsidian 中打开此笔记
   2. 填写各个部分的内容
   3. 建立与其他概念的双向链接
    """)


def cmd_explain(obsidian, concept_name):
    """讲解概念并创建详细笔记"""
    # 这里可以调用 Claude 的 explain 功能
    # 目前先创建基础笔记
    print(f"📖 正在讲解: {concept_name}")
    print()
    print("💡 这个功能需要 Claude Code 集成")
    print("   目前已创建基础笔记，请在 Obsidian 中补充内容")
    print()

    filepath = obsidian.create_concept_note(concept_name)
    print(f"📁 笔记位置: {filepath}")


def cmd_quiz(obsidian, topic=None):
    """创建测验笔记"""
    quiz_data = obsidian.tutor.generate_quiz(topic, 5)

    # 打印题目
    from ml_tutor import format_quiz
    print(format_quiz(quiz_data))
    print()

    # 创建笔记
    filepath = obsidian.create_quiz_note(
        quiz_data['topic'],
        quiz_data['questions']
    )
    print(f"📁 测验笔记已创建: {filepath}")


def cmd_review(obsidian, week=None):
    """创建周回顾"""
    if week is None:
        week = obsidian.tutor.tracker['current_week']

    from ml_tutor import format_review
    review_data = obsidian.tutor.generate_weekly_review(week)

    print(format_review(review_data))
    print()

    filepath = obsidian.create_weekly_review(week)
    print(f"📁 周回顾已保存: {filepath}")


def cmd_dashboard(obsidian):
    """更新进度仪表盘"""
    filepath = obsidian.update_progress_dashboard()
    print(f"✅ 进度仪表盘已更新!")
    print(f"📁 文件位置: {filepath}")
    print(f"🌐 在 Obsidian 中打开: {filepath}")


def cmd_skip(obsidian, reason=""):
    """跳过今天"""
    result = obsidian.tutor.mark_skip(reason)
    print(f"⏭️  已跳过第{result['week']}周第{result['day']}天")
    if reason:
        print(f"   原因: {reason}")


def cmd_project(obsidian, project_id):
    """创建项目笔记"""
    filepath = obsidian.create_project_note(project_id)
    print(f"""
✅ 项目笔记已创建: {project_id}

📁 文件位置: {filepath}

💡 可用的项目ID:
   titanic-eda, numpy-lr, spam-classifier, customer-churn
   numpy-neural-net, mnist-cnn-99, minigpt, bert-classification
   recommendation-web, rag-qa-system, mlops-pipeline
    """)


def cmd_projects(obsidian):
    """列出所有项目"""
    projects = obsidian.tutor.tracker.get('projects', {})

    print("📁 学习项目清单:")
    print("─" * 50)

    for project_id, info in projects.items():
        status_symbol = {
            'not_started': '⬜',
            'in_progress': '🔄',
            'done': '✅'
        }.get(info.get('status', 'not_started'), '⬜')

        print(f"{status_symbol} {project_id:25s} (W{info.get('week', 1):2d})")

    print()
    print("💡 使用 'project <ID>' 创建项目笔记")


def cmd_review_today(obsidian):
    """显示今日复习卡片"""
    sr = obsidian.tutor.sr_manager
    if sr is None:
        print("⚠️  间隔重复模块未安装")
        return
    due = sr.get_due_cards()
    from ml_tutor import format_due_reviews
    print(format_due_reviews(due))


def cmd_review_card(obsidian, concept, quality):
    """评分一张复习卡片"""
    sr = obsidian.tutor.sr_manager
    if sr is None:
        print("⚠️  间隔重复模块未安装")
        return
    result = sr.review_card(concept, int(quality))
    from ml_tutor import format_review_result
    print(format_review_result(result))


def cmd_review_stats(obsidian):
    """显示复习统计"""
    sr = obsidian.tutor.sr_manager
    if sr is None:
        print("⚠️  间隔重复模块未安装")
        return
    stats = sr.get_review_stats()
    from ml_tutor import format_review_stats
    print(format_review_stats(stats))


def cmd_init(obsidian):
    """初始化 Vault"""
    print("🚀 正在初始化 Obsidian Vault...")
    print(f"📁 位置: {obsidian.vault_path}")
    print()

    obsidian.init_concept_mocs()
    obsidian.update_progress_dashboard()

    print()
    print("✅ 初始化完成!")
    print()
    print("📌 接下来的步骤:")
    print("   1. 打开 Obsidian")
    print("   2. 选择 '打开文件夹作为仓库'")
    print(f"   3. 选择: {obsidian.vault_path}")
    print()
    print("💡 常用命令:")
    print("   python learning_system.py daily   # 创建今日笔记")
    print("   python learning_system.py today   # 查看今日计划")


def main():
    """主入口"""
    # 初始化
    obsidian = ObsidianIntegration()

    if len(sys.argv) < 2:
        print_banner()
        print_help()
        return

    cmd = sys.argv[1].lower()

    # 执行命令
    if cmd == "today":
        cmd_today(obsidian)

    elif cmd == "daily":
        cmd_daily(obsidian)

    elif cmd == "done":
        cmd_done(obsidian)

    elif cmd == "status":
        cmd_status(obsidian)

    elif cmd == "week":
        cmd_week(obsidian)

    elif cmd == "dashboard":
        cmd_dashboard(obsidian)

    elif cmd == "concept":
        if len(sys.argv) > 2:
            cmd_concept(obsidian, sys.argv[2])
        else:
            print("❌ 请提供概念名称")
            print("   用法: python learning_system.py concept <概念名>")

    elif cmd == "explain":
        if len(sys.argv) > 2:
            cmd_explain(obsidian, sys.argv[2])
        else:
            print("❌ 请提供概念名称")
            print("   用法: python learning_system.py explain <概念名>")

    elif cmd == "quiz":
        topic = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_quiz(obsidian, topic)

    elif cmd == "review":
        week = int(sys.argv[2]) if len(sys.argv) > 2 else None
        cmd_review(obsidian, week)

    elif cmd == "skip":
        reason = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        cmd_skip(obsidian, reason)

    elif cmd == "project":
        if len(sys.argv) > 2:
            cmd_project(obsidian, sys.argv[2])
        else:
            print("❌ 请提供项目ID")
            cmd_projects(obsidian)

    elif cmd == "projects":
        cmd_projects(obsidian)

    elif cmd == "review-today":
        cmd_review_today(obsidian)

    elif cmd == "review-done":
        if len(sys.argv) >= 4:
            cmd_review_card(obsidian, sys.argv[2], sys.argv[3])
        else:
            print("❌ 用法: review-done <概念> <评分0-5>")

    elif cmd == "review-stats":
        cmd_review_stats(obsidian)

    elif cmd == "init":
        cmd_init(obsidian)

    elif cmd in ["help", "-h", "--help"]:
        print_help()

    else:
        print(f"❌ 未知命令: {cmd}")
        print()
        print_help()


if __name__ == "__main__":
    main()
