# web_server.py — VisionLearner v4.0 Web 服务器
"""
启动：
    python main.py --web --port 5000
    浏览器访问 http://localhost:5000
    手机（同一局域网）访问 http://192.168.x.x:5000

API端点：
    GET  /api/status          系统状态
    GET  /api/goals           目标列表
    POST /api/goal/new        新建目标 {description}
    POST /api/goal/select     切换目标 {goal_id}
    POST /api/ask             提问     {message, stream?}
    POST /api/skill/add       新增技能 {description}
    GET  /api/skills          技能列表
    GET  /api/mindmap         思维导图JSON {unit?}
    GET  /api/quiz/question   获取一道测验题（随机）
    POST /api/quiz/check      提交答案并评分 {user, answer}
    POST /api/autopilot       自动驾驶 {description?}
    GET  /api/progress        进度报告
    POST /api/backup          备份数据
    GET  /stream/ask          流式问答 SSE
"""

import json
import socket
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import LearningSystem

try:
    from flask import Flask, request, jsonify, Response, send_from_directory
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("[WARN] Flask not installed, run: pip install flask flask-cors")


UI_HTML = Path(__file__).parent / "visionlearner_ui.html"


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def create_app(system: "LearningSystem") -> "Flask":
    app = Flask(__name__, static_folder=None)

    # 启用 CORS 允许所有来源
    CORS(app)

    # ── 前端 ──────────────────────────────────────────────────
    @app.route("/")
    def index():
        if UI_HTML.exists():
            return UI_HTML.read_text(encoding="utf-8")
        return "<h1>VisionLearner</h1><p>UI file not found. Place visionlearner_ui.html here.</p>"

    @app.route("/test")
    def test_ui():
        test_html = Path(__file__).parent / "test_ui.html"
        if test_html.exists():
            return test_html.read_text(encoding="utf-8")
        return "<h1>Test UI not found</h1>"

    @app.route("/simple")
    def simple_test():
        simple_html = Path(__file__).parent / "simple_test.html"
        if simple_html.exists():
            return simple_html.read_text(encoding="utf-8")
        return "<h1>Simple Test not found</h1>"

    @app.route("/code_analyzer")
    def code_analyzer_ui():
        analyzer_html = Path(__file__).parent / "code_analyzer_ui.html"
        if analyzer_html.exists():
            return analyzer_html.read_text(encoding="utf-8")
        return "<h1>Code Analyzer UI not found</h1>"

    # ── 系统状态 ──────────────────────────────────────────────
    @app.route("/api/status")
    def api_status():
        try:
            return jsonify(system.api_status())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── 目标管理 ──────────────────────────────────────────────
    @app.route("/api/goals")
    def api_goals():
        try:
            return jsonify(system.list_goals())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/goal/new", methods=["POST"])
    def api_new_goal():
        data = request.get_json(force=True)
        desc = (data or {}).get("description", "").strip()
        depth = int((data or {}).get("depth", 3))
        if not desc:
            return jsonify({"error": "description 不能为空"}), 400
        try:
            goal = system.create_goal(desc, depth=depth)
            vision_score = None
            if system.vision:
                try:
                    assessment = system.vision.assess_goal(desc)
                    vision_score = assessment.get("score") if assessment else None
                except Exception:
                    pass
            return jsonify({
                "id": goal.id,
                "description": goal.description,
                "status": "created",
                "vision_score": vision_score,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/goal/select", methods=["POST"])
    def api_select_goal():
        data    = request.get_json(force=True)
        goal_id = (data or {}).get("goal_id", "").strip()
        ok      = system.select_goal(goal_id)
        return jsonify({"ok": ok})

    @app.route("/api/goal/delete", methods=["POST"])
    def api_delete_goal():
        data    = request.get_json(force=True)
        goal_id = (data or {}).get("goal_id", "").strip()
        if not goal_id:
            return jsonify({"error": "goal_id 不能为空"}), 400
        ok      = system.delete_goal(goal_id)
        return jsonify({"ok": ok})

    # ── 问答 ──────────────────────────────────────────────────
    @app.route("/api/ask", methods=["POST"])
    def api_ask():
        data    = request.get_json(force=True)
        message = (data or {}).get("message", "").strip()
        mode    = (data or {}).get("mode", "auto")  # auto/local/llm
        if not message:
            return jsonify({"error": "message 不能为空"}), 400
        try:
            answer = system.answer(message, stream=False, mode=mode)
            return jsonify({"answer": answer})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/stream/ask")
    def stream_ask():
        """Server-Sent Events 流式问答"""
        message = request.args.get("message", "").strip()
        mode    = request.args.get("mode", "auto")  # auto/local/llm
        if not message:
            return jsonify({"error": "message 参数缺失"}), 400

        def generate():
            try:
                for chunk in system.answer(message, stream=True, mode=mode):
                    data = json.dumps({"chunk": chunk}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                yield "data: {\"done\": true}\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    # ── 技能 ──────────────────────────────────────────────────
    @app.route("/api/skills")
    def api_skills():
        return jsonify(system.list_skills())

    @app.route("/api/skill/add", methods=["POST"])
    def api_add_skill():
        data = request.get_json(force=True)
        desc = (data or {}).get("description", "").strip()
        if not desc:
            return jsonify({"error": "description 不能为空"}), 400
        result = system.add_skill(desc)
        return jsonify(result)

    # ── 思维导图 ──────────────────────────────────────────────
    @app.route("/api/mindmap")
    def api_mindmap():
        unit = request.args.get("unit", "")
        return jsonify(system.api_mindmap(unit or None))

    @app.route("/api/mindmap/all")
    def api_mindmap_all():
        """返回当前目标所有单元的简要列表"""
        gid = system.current_goal.id if system.current_goal else None
        if not gid:
            return jsonify([])
        units   = system.col.load_goal_units(gid)
        gt      = system._goal_type.get(gid, "general")
        result  = []
        for u in units[:50]:
            tree = system._get_tree(gid, u, gt)
            if tree:
                nodes = tree._all_nodes() if hasattr(tree, "_all_nodes") else []
                # 只统计知识节点（depth > 0），排除根节点
                knowledge_nodes = [n for n in nodes if n.depth > 0]
                total = len(knowledge_nodes)
                done  = sum(1 for n in knowledge_nodes if n.collected)
            else:
                total = done = 0
            result.append({"unit": u, "total": total, "collected": done,
                            "pct": round(done/max(total,1), 2)})
        return jsonify(result)

    # ── 进度 ──────────────────────────────────────────────────
    @app.route("/api/progress")
    def api_progress():
        try:
            return jsonify(system.monitor_progress())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── 测验 ──────────────────────────────────────────────────
    @app.route("/api/quiz/question")
    def api_quiz_question():
        """返回一道题目（JSON），不执行CLI交互"""
        import random
        gid       = system.current_goal.id if system.current_goal else None
        goal_type = system._goal_type.get(gid, "general") if gid else "general"
        units     = system.col.load_goal_units(gid) if gid else []
        if not units:
            return jsonify({"error": "无目标单元"}), 400
        random.shuffle(units)
        for unit in units[:20]:
            tree = system._get_tree(gid, unit, goal_type)
            if not tree: continue
            nodes = [n for n in (tree._all_nodes() if hasattr(tree,"_all_nodes") else [])
                     if n.collected and n.depth > 0]
            if not nodes: continue
            node  = random.choice(nodes)
            from conversation import format_content
            ans   = format_content(node.content)
            if not ans or ans == "（暂无）": continue
            q_map = {"读音": f"「{unit}」怎么读？", "含义": f"「{unit}」是什么意思？",
                     "组词": f"用「{unit}」组一个词", "笔画": f"「{unit}」共几画？"}
            question = q_map.get(node.title, f"「{unit}」的{node.title}？")
            return jsonify({"unit": unit, "node": node.title,
                            "question": question, "answer": ans})
        return jsonify({"error": "暂无可测验内容"}), 400

    @app.route("/api/quiz/check", methods=["POST"])
    def api_quiz_check():
        data = request.get_json(force=True)
        user = (data or {}).get("user", "")
        ref  = (data or {}).get("answer", "")
        score = system._score_answer(user, ref)
        return jsonify({"score": score, "correct": score >= 0.8,
                        "partial": 0.4 <= score < 0.8})

    # ── 自动驾驶 ──────────────────────────────────────────────
    @app.route("/api/autopilot", methods=["POST"])
    def api_autopilot():
        data = request.get_json(force=True)
        desc = (data or {}).get("description", "")
        result = system.run_auto_pilot(desc or None)
        return jsonify(result)

    # ── 备份 ──────────────────────────────────────────────────
    @app.route("/api/backup", methods=["POST"])
    def api_backup():
        path = system.db.backup_data()
        return jsonify({"path": path, "ok": True})

    # ── 添加笔记 ───────────────────────────────────────────────
    @app.route("/api/note/add", methods=["POST"])
    def api_note_add():
        data = request.get_json(force=True)
        goal_id = (data or {}).get("goal_id", "").strip()
        unit = (data or {}).get("unit", "").strip()
        content = (data or {}).get("content", "").strip()
        node_title = (data or {}).get("node_title", "笔记").strip()

        # 如果未指定 goal_id，使用当前目标
        if not goal_id and system.current_goal:
            goal_id = system.current_goal.id

        if not goal_id:
            return jsonify({"error": "未指定学习目标"}), 400
        if not unit:
            return jsonify({"error": "unit 不能为空"}), 400
        if not content:
            return jsonify({"error": "content 不能为空"}), 400

        try:
            system.vector.add_unit_knowledge(
                goal_id=goal_id,
                unit=unit,
                content_text=content,
                node_title=node_title,
            )
            return jsonify({"ok": True, "message": f"已保存笔记到「{unit}」/{node_title}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── 填充 ──────────────────────────────────────────────────
    @app.route("/api/populate", methods=["POST"])
    def api_populate():
        data  = request.get_json(force=True)
        limit = (data or {}).get("limit", None)
        # None 或 0 表示全部单元
        unit_limit = None if limit is None or limit == 0 else limit
        result = system.populate(unit_limit=unit_limit)
        return jsonify(result)

    # ── 模型管理 ──────────────────────────────────────────────
    @app.route("/api/models/available")
    def api_available_models():
        """获取所有可用的LLM模型"""
        providers = system.llm.list_available_providers()
        return jsonify({"providers": providers})

    @app.route("/api/models/current")
    def api_current_model():
        """获取当前使用的模型"""
        return jsonify({
            "provider": system.llm.provider_name,
            "model": system.llm.model,
            "display_name": system.llm.config.name
        })

    @app.route("/api/models/switch", methods=["POST"])
    def api_switch_model():
        """切换到指定的模型"""
        data = request.get_json(force=True)
        provider = (data or {}).get("provider", "").strip() if data else ""
        model = (data or {}).get("model") if data else None

        if not provider:
            return jsonify({"error": "provider 不能为空"}), 400

        try:
            success = system.llm.switch_provider(provider, model)
            if success:
                # 同步更新所有子系统的llm客户端
                system.col.llm = system.llm
                system.skills.llm = system.llm
                # 同步更新perception模块的llm客户端
                if hasattr(system, 'perception') and system.perception:
                    system.perception.llm = system.llm
                    system.perception.extractor.llm = system.llm
                    system.perception.mindmap_gen.llm = system.llm
                    system.perception.trigger.llm = system.llm
                # 同步更新planner的llm客户端
                if hasattr(system, 'planner') and system.planner:
                    system.planner.llm = system.llm

                return jsonify({
                    "ok": True,
                    "provider": system.llm.provider_name,
                    "model": system.llm.model,
                    "display_name": system.llm.config.name
                })
            else:
                return jsonify({"error": f"切换到 provider '{provider}' 失败"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── 多模态识别 ─────────────────────────────────────────────
    @app.route("/api/multimodal/upload", methods=["POST"])
    def api_multimodal_upload():
        """上传图片或音频文件进行识别"""
        print(f"\n{'='*60}")
        print(f"[API] 收到 /api/multimodal/upload 请求")
        print(f"{'='*60}\n")

        try:
            # 检查文件上传
            if 'file' not in request.files:
                print("[ERROR] 未上传文件")
                return jsonify({"error": "未上传文件"}), 400

            file = request.files['file']
            print(f"[INFO] 收到文件: {file.filename}")

            if file.filename == '':
                print("[ERROR] 未选择文件")
                return jsonify({"error": "未选择文件"}), 400

            # 保存文件到临时目录
            import tempfile
            import os

            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            print(f"[INFO] 文件已保存到: {file_path}")
            print(f"[INFO] 文件大小: {os.path.getsize(file_path)} 字节")

            # 判断文件类型
            file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            is_image = file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
            is_audio = file_ext in ['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm']

            print(f"[INFO] 文件类型: {file_ext}, 是图片: {is_image}, 是音频: {is_audio}")

            result = {"type": "unknown", "content": "", "error": None}

            if is_image:
                # 图片识别
                print(f"\n[INFO] 开始图片识别流程...")
                try:
                    if hasattr(system, 'perception') and system.perception:
                        print("[DEBUG] 调用 perception.describe_image() 获取描述...")

                        # 获取图片描述
                        description = system.perception.describe_image(file_path)

                        if not description:
                            print("[ERROR] 图片描述为空")
                            result["type"] = "image"
                            result["content"] = "图片识别失败，未能获取描述"
                            result["error"] = "Failed to get image description"
                        else:
                            # 从描述中提取知识点
                            print(f"[DEBUG] 开始从描述中提取知识点...")
                            nodes = system.perception.extractor.extract_from_text(description, source=file_path)

                            print(f"[SUCCESS] 图片识别完成，提取到 {len(nodes)} 个知识点")
                            result["type"] = "image"
                            result["success"] = True
                            result["description"] = description  # 完整的识别描述
                            result["content"] = "图片识别成功，已提取知识点"
                            result["nodes"] = [{"title": n.title, "summary": n.summary, "content": n.content} for n in nodes]
                            result["node_count"] = len(nodes)
                            print(f"[INFO] 返回结果: type={result['type']}, node_count={result['node_count']}, description_length={len(description)}")
                    else:
                        print("[ERROR] 感知模块未加载")
                        result["type"] = "image"
                        result["content"] = "感知模块未加载"
                        result["error"] = "Perception module not available"
                except Exception as e:
                    print(f"[ERROR] 图片识别异常: {e}")
                    import traceback
                    traceback.print_exc()
                    result["type"] = "image"
                    result["error"] = str(e)

            elif is_audio:
                # 音频识别
                try:
                    if hasattr(system, 'perception') and system.perception:
                        # 尝试从音频提取文本
                        transcript = system.perception.extractor.extract_from_audio(file_path)

                        result["type"] = "audio"
                        result["transcript"] = transcript

                        # 如果成功转录了文本，可以进一步提取知识
                        if transcript and "语音识别API" not in transcript:
                            nodes = system.perception.extractor.extract_from_text(transcript, source="audio")
                            result["content"] = f"音频转录成功：{transcript}"
                            result["nodes"] = [{"title": n.title, "summary": n.summary} for n in nodes]
                            result["node_count"] = len(nodes)
                        else:
                            result["content"] = transcript
                            result["info"] = "当前使用的是基础音频识别，建议接入专门的语音识别API以获得更好的效果"
                    else:
                        result["type"] = "audio"
                        result["content"] = "感知模块未加载"
                        result["error"] = "Perception module not available"
                except Exception as e:
                    result["type"] = "audio"
                    result["error"] = str(e)

            else:
                result["type"] = "unsupported"
                result["error"] = f"不支持的文件类型: {file_ext}"

            # 清理临时文件
            try:
                os.unlink(file_path)
            except:
                pass

            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── 健康检查 ──────────────────────────────────────────────
    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "version": "4.0"})

    # ── 代码分析器 ──────────────────────────────────────────────
    try:
        from code_analyzer import register_analyzer_routes
        register_analyzer_routes(app, system)
    except ImportError:
        pass  # code_analyzer 可选

    return app


def run_server(system: "LearningSystem", port: int = 5000,
               host: str = "0.0.0.0", debug: bool = False):
    if not HAS_FLASK:
        print("[ERROR] Flask not installed: pip install flask")
        return

    app = create_app(system)
    local_ip = get_local_ip()

    print(f"""
+============================================================+
|  [WEB] VisionLearner Web UI Started                        |
+------------------------------------------------------------+
|  Local:   http://localhost:{port:<42}|
|  Network: http://{local_ip}:{port:<{50-len(local_ip)}}|
|  Make sure phone is on same WiFi                          |
+============================================================+""")

    # 启用访问日志显示每个请求
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)
    
    app.run(host=host, port=port, debug=debug,
            threaded=True, use_reloader=False)
