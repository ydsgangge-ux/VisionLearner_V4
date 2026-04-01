# heartbeat.py — VisionLearner v4.0 心跳调度器（方向B）
"""
后台定时运行，不需要用户触发：
  - 每小时检查到期复习节点，自动推送提醒
  - 每天早8点发送今日学习建议
  - 每天晚9点检查未完成任务
  - 主动发现知识盲点，后台静默填充

运行：
    python heartbeat.py              # 前台运行
    python heartbeat.py --daemon     # 后台守护进程（Linux/Mac）

与 main.py 共享同一 learning_data/ 目录，数据完全互通。
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HEARTBEAT] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("heartbeat")

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

# ─────────────────────────────────────────────────────────────
# 通知适配器（终端 / Telegram / 未来可扩展）
# ─────────────────────────────────────────────────────────────

class Notifier:
    """通知推送适配器"""

    def __init__(self, telegram_token: str = None, chat_id: str = None):
        self._token   = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    def send(self, text: str) -> None:
        # 终端始终输出
        print(f"\n🔔 [{datetime.now().strftime('%H:%M')}] {text}\n")

        # Telegram（如果配置了）
        if self._token and self._chat_id:
            try:
                import requests
                requests.post(
                    f"https://api.telegram.org/bot{self._token}/sendMessage",
                    json={"chat_id": self._chat_id, "text": text,
                          "parse_mode": "HTML"},
                    timeout=10
                )
            except Exception as e:
                log.warning(f"Telegram推送失败: {e}")


# ─────────────────────────────────────────────────────────────
# 核心调度器
# ─────────────────────────────────────────────────────────────

class Heartbeat:
    """
    心跳调度器：与 LearningSystem 共享数据目录，独立进程运行
    """

    def __init__(self,
                 data_dir:   str = "./learning_data",
                 skills_dir: str = "./skills",
                 provider:   str = None):
        log.info("心跳调度器启动...")
        self.data_dir = Path(data_dir)
        self.notifier = Notifier()
        self._system: Optional[object] = None
        self._provider = provider
        self._skills_dir = skills_dir
        self._boot_time = datetime.now()

    def _get_system(self):
        """懒加载 LearningSystem（避免启动时就加载LLM）"""
        if self._system is None:
            try:
                from main_v4 import LearningSystem
                self._system = LearningSystem(
                    str(self.data_dir), self._skills_dir, self._provider)
            except Exception as e:
                log.error(f"加载LearningSystem失败: {e}")
        return self._system

    # ── 定时任务 ────────────────────────────────────────────

    def check_due_reviews(self):
        """每小时：检查到期复习节点"""
        log.info("🔍 检查到期复习...")
        sys_ = self._get_system()
        if not sys_: return

        try:
            due = sys_.db.get_due_reviews()
            if not due:
                log.info("  暂无到期复习"); return

            # 按目标分组
            grouped: dict = {}
            for r in due[:20]:
                gid = r.get("goal_id", "unknown")
                grouped.setdefault(gid, []).append(r)

            msg_parts = ["📚 <b>复习提醒</b>"]
            for gid, reviews in grouped.items():
                units = [r.get("unit","?") for r in reviews[:5]]
                msg_parts.append(f"• {', '.join(units)} 等 {len(reviews)} 个单元需要复习")

            self.notifier.send("\n".join(msg_parts))
        except Exception as e:
            log.warning(f"复习检查失败: {e}")

    def detect_and_fill_gaps(self):
        """每2小时：主动发现知识盲点，后台静默填充"""
        log.info("🔍 知识盲点探测...")
        sys_ = self._get_system()
        if not sys_ or not sys_.current_goal: return

        try:
            gid       = sys_.current_goal.id
            goal_type = sys_._goal_type.get(gid, "general")
            units     = sys_.col.load_goal_units(gid)
            report    = sys_.col.get_completion_report(gid, units[:100])

            # 找完成度 < 30% 的单元
            gaps = []
            for unit in units[:50]:
                tree = sys_._get_tree(gid, unit, goal_type)
                if tree:
                    nodes  = tree._all_nodes() if hasattr(tree,"_all_nodes") else []
                    done   = sum(1 for n in nodes if n.collected)
                    total  = len(nodes)
                    pct    = done / max(total, 1)
                    if pct < 0.30 and total > 0:
                        gaps.append((unit, pct, total-done))

            if not gaps:
                log.info("  无明显盲点"); return

            log.info(f"  发现 {len(gaps)} 个盲点，开始后台填充...")
            filled = 0
            for unit, pct, missing in gaps[:5]:  # 每次最多填5个
                tree   = sys_._get_tree(gid, unit, goal_type)
                before = sum(1 for n in (tree._all_nodes() if hasattr(tree,"_all_nodes") else [])
                             if n.collected)
                sys_.col.collect_tree(tree, unit, goal_type)
                sys_.col.save_tree(gid, unit, tree)
                sys_._trees.setdefault(gid, {})[unit] = tree
                after = sum(1 for n in (tree._all_nodes() if hasattr(tree,"_all_nodes") else [])
                            if n.collected)
                filled += (after - before)
                log.info(f"    ✅ {unit}: +{after-before} 节点")

            if filled > 0:
                self.notifier.send(
                    f"🆕 <b>后台学习</b>\n"
                    f"自动填充了 {filled} 个知识点\n"
                    f"总完成度：{report['overall_completion']:.0%}")
        except Exception as e:
            log.warning(f"盲点探测失败: {e}")

    def daily_morning(self):
        """每天早8点：今日学习建议"""
        sys_ = self._get_system()
        if not sys_ or not sys_.current_goal: return
        try:
            gid   = sys_.current_goal.id
            units = sys_.col.load_goal_units(gid)
            rep   = sys_.col.get_completion_report(gid, units[:100])
            pct   = rep.get("overall_completion", 0)

            msg = (f"[SUN] <b>Good Morning! Daily Learning Reminder</b>\n\n"
                   f"[BOOK] Current goal: {sys_.current_goal.description[:30]}\n"
                   f"[%] Total progress: {pct:.0%} ({rep.get('learned_units',0)}/{rep.get('total_units',0)})\n\n"
                   f"[IDEA] Suggestion: Complete 5-10 knowledge units today\n"
                   f"   Open VisionLearner to start learning!")
            self.notifier.send(msg)
        except Exception as e:
            log.warning(f"早安推送失败: {e}")

    def daily_evening(self):
        """每天晚9点：今日总结"""
        sys_ = self._get_system()
        if not sys_ or not sys_.current_goal: return
        try:
            gid   = sys_.current_goal.id
            units = sys_.col.load_goal_units(gid)
            rep   = sys_.col.get_completion_report(gid, units[:100])
            pct   = rep.get("overall_completion", 0)

            msg = (f"[MOON] <b>Today's Learning Summary</b>\n\n"
                   f"[%] Current progress: {pct:.0%}\n"
                   f"[OK] Mastered: {rep.get('learned_units',0)} units\n\n"
                   f"Keep going! See you tomorrow [~]")
            self.notifier.send(msg)
        except Exception as e:
            log.warning(f"Goodnight push failed: {e}")

    def daily_backup(self):
        """每天凌晨2点：自动备份学习数据"""
        sys_ = self._get_system()
        if not sys_: return
        try:
            path = sys_.db.backup_data()
            log.info(f"✅ 每日自动备份完成: {path}")
            # 同时清理30天前的旧备份
            removed = sys_.db.cleanup_old_data(days_old=30)
            if removed > 0:
                log.info(f"   清理旧备份 {removed} 个")
        except Exception as e:
            log.warning(f"自动备份失败: {e}")

    def heartbeat_tick(self):
        """每5分钟：心跳检测，写状态文件供其他进程读取"""
        status = {
            "alive": True,
            "boot_time": self._boot_time.isoformat(),
            "last_tick": datetime.now().isoformat(),
            "has_goal": self._system is not None and (
                getattr(self._system, "current_goal", None) is not None)
        }
        try:
            status_file = self.data_dir / ".heartbeat_status.json"
            status_file.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ── 调度注册 ────────────────────────────────────────────

    def _run_with_schedule(self):
        if not HAS_SCHEDULE:
            print("❌ schedule 未安装: pip install schedule")
            sys.exit(1)

        # 注册任务
        schedule.every(1).hours.do(self.check_due_reviews)
        schedule.every(2).hours.do(self.detect_and_fill_gaps)
        schedule.every().day.at("08:00").do(self.daily_morning)
        schedule.every().day.at("21:00").do(self.daily_evening)
        schedule.every().day.at("02:00").do(self.daily_backup)
        schedule.every(5).minutes.do(self.heartbeat_tick)

        log.info("📅 调度已注册：")
        log.info("   每1小时   → 复习检查")
        log.info("   每2小时   → 知识盲点探测+填充")
        log.info("   每天08:00 → 早安提醒")
        log.info("   每天21:00 → 晚安总结")
        log.info("   每天02:00 → 自动备份（保留最近30天）")
        log.info("   每5分钟   → 心跳 tick")

        # 立即跑一次心跳
        self.heartbeat_tick()
        # 立即检查一次盲点
        self.detect_and_fill_gaps()

        log.info("✅ 心跳调度器运行中（Ctrl+C 退出）\n")
        while True:
            schedule.run_pending()
            time.sleep(30)

    def _run_simple(self):
        """无 schedule 库的简单版本"""
        log.info("使用简单模式（1小时间隔）")
        self.detect_and_fill_gaps()
        tick = 0
        while True:
            time.sleep(60)
            tick += 1
            if tick % 5 == 0:    self.heartbeat_tick()
            if tick % 60 == 0:   self.check_due_reviews()
            if tick % 120 == 0:  self.detect_and_fill_gaps()
            now = datetime.now()
            if now.hour == 8 and now.minute < 2:   self.daily_morning()
            if now.hour == 21 and now.minute < 2:  self.daily_evening()

    def run(self):
        try:
            if HAS_SCHEDULE:
                self._run_with_schedule()
            else:
                self._run_simple()
        except KeyboardInterrupt:
            log.info("心跳调度器已停止")


def main():
    parser = argparse.ArgumentParser(description="VisionLearner 心跳调度器")
    parser.add_argument("--data-dir",   default="./learning_data")
    parser.add_argument("--skills-dir", default="./skills")
    parser.add_argument("--provider",   default=None)
    parser.add_argument("--daemon",     action="store_true", help="后台守护进程")
    args = parser.parse_args()

    if args.daemon:
        # 简单 fork 守护进程（Linux/Mac）
        try:
            pid = os.fork()
            if pid > 0:
                print(f"✅ 心跳调度器已在后台启动（PID: {pid}）")
                sys.exit(0)
            os.setsid()
            log.info(f"守护进程 PID: {os.getpid()}")
        except AttributeError:
            log.warning("Windows 不支持 fork，使用前台模式")

    hb = Heartbeat(args.data_dir, args.skills_dir, args.provider)
    hb.run()


if __name__ == "__main__":
    main()
